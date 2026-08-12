"""
RadAssist AI — Report Quality Checker (Phase 5, Step 3)

Detects missing sections, ambiguous wording, and internal contradictions in a
report draft.

════════════════════════════════════════════════════════════════════
DETERMINISTIC RULES, NOT A MODEL
════════════════════════════════════════════════════════════════════
Everything here is a regex or a set operation. That is a deliberate choice,
not a shortcut.

A quality checker is a tool a radiologist looks at dozens of times a day. It
earns trust by being predictable: the same draft must produce the same flags
every time, each flag must point at a specific line, and a clean report must
produce silence. A model would catch subtler problems and would also fire
differently on identical input, flag things it cannot explain, and cost a
second per check.

The measurable claim in the project document is *"reduction in
missing-section / inconsistent-terminology flags over time"*. That is a
COUNT, and counting requires a stable detector. A model whose sensitivity
drifts between versions makes the metric meaningless — the number would move
without the reports changing.

⚠️  FALSE POSITIVES ARE THE FAILURE MODE THAT MATTERS.
A checker that flags correct reports gets switched off within a week, and then
catches nothing at all. Every rule here is written to stay silent when
uncertain, and every rule is tested against a known-good report as well as a
broken one.

Model-based checks (terminology drift, subtle contradiction) belong on top of
this layer later — never underneath it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class Severity:
    """
    How much attention an issue deserves.

    ERROR   — the report is unsafe or unusable as written
    WARNING — likely wrong, needs a human decision
    INFO    — style and consistency
    """
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    ALL = (ERROR, WARNING, INFO)
    # Sort order for display: worst first.
    RANK = {ERROR: 0, WARNING: 1, INFO: 2}


@dataclass
class Issue:
    """One problem found in a draft."""
    code: str          # stable identifier — the thing you count over time
    severity: str
    message: str       # what is wrong, in the reviewer's language
    line: int | None = None    # 1-based, for highlighting
    excerpt: str = ""          # the offending text


@dataclass
class QualityReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def is_clean(self) -> bool:
        return not self.issues


# ══════════════════════════════════════════════════════════════
# PATTERNS
# ══════════════════════════════════════════════════════════════

_FINDINGS_HEADING = re.compile(r"^\W*\**\s*findings\s*\**\W*$", re.I | re.M)
_IMPRESSION_HEADING = re.compile(
    r"^\W*\**\s*(impression|conclusion)s?\s*\**\W*$", re.I | re.M
)

# Template text nobody meant to ship. Cheap to detect, embarrassing to miss.
_PLACEHOLDERS = re.compile(
    r"\b(TODO|TBD|FIXME|XXX+|LOREM IPSUM|INSERT\s+\w+|PATIENT\s*NAME"
    r"|\[\s*\]|_{3,}|\.{4,})",
    re.I,
)

# A measurement with no unit. "a 12 mm nodule" is fine; "a 12 nodule" is not,
# and "measuring 8" leaves the reader guessing mm or cm — a tenfold error.
_UNITLESS_MEASUREMENT = re.compile(
    r"\b(measur\w+|approximately|approx\.?|about|up to|size of)\s+"
    r"(\d+(?:\.\d+)?)\s*(?!\s*(mm|cm|m\b|millimet|centimet|%|percent|degrees?|°))"
    r"([a-z]|$)",
    re.I,
)

# Hedging stacked on hedging. One hedge is clinical caution; three in a row
# means the sentence carries no information at all.
_HEDGES = (
    "possibly", "possible", "may represent", "might represent", "could represent",
    "cannot be excluded", "can not be excluded", "cannot be ruled out",
    "suspicious for", "questionable", "probably", "likely", "apparent",
    "seems", "appears to possibly",
)

# Vague where a number belongs.
_VAGUE_SIZE = re.compile(
    r"\b(somewhat|fairly|rather|slightly)?\s*"
    r"(enlarged|prominent|increased|decreased|small|large|big)\b"
    r"(?![^.]{0,40}\b\d)",
    re.I,
)

_LEFT = re.compile(r"\bleft\b|\bl\.?[\s-]?sided\b", re.I)
_RIGHT = re.compile(r"\bright\b|\br\.?[\s-]?sided\b", re.I)

# Numbers that carry clinical weight. Excludes list numbering ("1." at the
# start of a line) and section numbers.
#
# ⚠️  '%' CANNOT TAKE A TRAILING \b.
# A word boundary needs a word character on one side. In "50% height" the
# character after '%' is a space, so `\b\d+\s*%\b` never matches — which
# silently disabled the check that exists to catch the Phase 4 failure of an
# impression inventing a percentage. The unit alternatives are split so only
# the alphabetic ones carry the boundary.
_MEASUREMENT_TOKEN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:(?:mm|cm|percent)\b|%)", re.I
)

# The draft label the report prompt appends. It is boilerplate, not clinical
# content, and counting it as part of the impression made every correct report
# fail with "terms appear in the impression but nowhere in the findings:
# draft, final, radiologist, report, review".
_DISCLAIMER = re.compile(r"\*?\s*draft for radiologist review[^\n]*", re.I)

_STOPWORDS = frozenset("""
a an and are as at be but by for from has have in into is it its no not of on
or that the there these this to was were with without within
""".split())


def _lines(text: str) -> list[str]:
    return text.splitlines()


def _line_of(text: str, index: int) -> int:
    """1-based line number for a character offset."""
    return text.count("\n", 0, index) + 1


def _split_sections(text: str) -> tuple[str, str]:
    """
    Return (findings_text, impression_text).

    Either may be empty. Split on the impression heading rather than parsing
    the whole document: reports vary wildly in what else they contain
    (technique, comparison, clinical history), and a strict parser would
    reject valid reports — which is the failure mode this module is most
    trying to avoid.
    """
    match = _IMPRESSION_HEADING.search(text)
    if not match:
        return text, ""
    return text[: match.start()], text[match.end():]


# ══════════════════════════════════════════════════════════════
# RULES
# ══════════════════════════════════════════════════════════════


def _check_sections(text: str) -> list[Issue]:
    issues: list[Issue] = []
    findings, impression = _split_sections(text)

    if not _FINDINGS_HEADING.search(text):
        issues.append(Issue(
            code="missing_findings",
            severity=Severity.ERROR,
            message="No FINDINGS section. The report has no observations to support its conclusions.",
        ))
    elif not findings.strip() or len(_strip_headings(findings)) < 10:
        issues.append(Issue(
            code="empty_findings",
            severity=Severity.ERROR,
            message="The FINDINGS section is empty.",
        ))

    if not _IMPRESSION_HEADING.search(text):
        issues.append(Issue(
            code="missing_impression",
            severity=Severity.ERROR,
            message=(
                "No IMPRESSION section. This is the part the referring "
                "clinician reads first."
            ),
        ))
    elif len(_strip_headings(impression)) < 10:
        issues.append(Issue(
            code="empty_impression",
            severity=Severity.ERROR,
            message="The IMPRESSION section is empty.",
        ))

    return issues


def _strip_headings(section: str) -> str:
    out = _FINDINGS_HEADING.sub("", section)
    out = _IMPRESSION_HEADING.sub("", out)
    # Drop the draft disclaimer and bullet punctuation so an "empty" section
    # containing only a dash is still recognised as empty.
    out = _DISCLAIMER.sub("", out)
    return re.sub(r"[\s\-*•·]", "", out)


def _check_placeholders(text: str) -> list[Issue]:
    return [
        Issue(
            code="placeholder_text",
            severity=Severity.ERROR,
            message=f"Template placeholder left in the report: {m.group(0)!r}",
            line=_line_of(text, m.start()),
            excerpt=m.group(0),
        )
        for m in _PLACEHOLDERS.finditer(text)
    ]


def _check_unitless_measurements(text: str) -> list[Issue]:
    """
    ⚠️  A MISSING UNIT IS A TENFOLD ERROR.
    "measuring 8" could be 8mm or 8cm. For a nodule that is the difference
    between routine follow-up and urgent work-up.
    """
    issues = []
    for m in _UNITLESS_MEASUREMENT.finditer(text):
        issues.append(Issue(
            code="measurement_without_unit",
            severity=Severity.ERROR,
            message=(
                f"Measurement {m.group(2)!r} has no unit. mm and cm differ by "
                f"a factor of ten."
            ),
            line=_line_of(text, m.start()),
            excerpt=m.group(0).strip(),
        ))
    return issues


def _check_stacked_hedging(text: str) -> list[Issue]:
    issues = []
    for i, line in enumerate(_lines(text), start=1):
        lowered = line.lower()
        hits = [h for h in _HEDGES if h in lowered]
        if len(hits) >= 2:
            issues.append(Issue(
                code="stacked_hedging",
                severity=Severity.WARNING,
                message=(
                    f"Multiple hedges in one statement ({', '.join(hits[:3])}). "
                    f"Stacked qualifiers leave the reader without a usable "
                    f"conclusion."
                ),
                line=i,
                excerpt=line.strip()[:120],
            ))
    return issues


def _check_vague_sizing(text: str) -> list[Issue]:
    """
    A size word with no number nearby. INFO, not WARNING: "mild cardiomegaly"
    is standard, accepted phrasing and flagging it as a problem would make the
    checker noise.
    """
    issues = []
    findings, _ = _split_sections(text)
    for m in _VAGUE_SIZE.finditer(findings):
        word = m.group(0).strip()
        issues.append(Issue(
            code="unquantified_size",
            severity=Severity.INFO,
            message=f"{word!r} is not quantified. Consider a measurement.",
            line=_line_of(findings, m.start()),
            excerpt=word,
        ))
    return issues


def _check_laterality(text: str) -> list[Issue]:
    """
    ⚠️  LATERALITY IS THE CLASSIC REPORTING ERROR.
    Flagged only when the impression states a side the findings never
    mention — that is a contradiction the reviewer must resolve. A report that
    simply never mentions a side is not flagged; plenty of findings have no
    laterality.
    """
    findings, impression = _split_sections(text)
    if not impression.strip():
        return []

    issues = []
    f_left, f_right = bool(_LEFT.search(findings)), bool(_RIGHT.search(findings))
    i_left, i_right = bool(_LEFT.search(impression)), bool(_RIGHT.search(impression))

    if i_left and not f_left and f_right:
        issues.append(Issue(
            code="laterality_mismatch",
            severity=Severity.ERROR,
            message=(
                "The impression says LEFT but the findings describe only the "
                "RIGHT side."
            ),
        ))
    if i_right and not f_right and f_left:
        issues.append(Issue(
            code="laterality_mismatch",
            severity=Severity.ERROR,
            message=(
                "The impression says RIGHT but the findings describe only the "
                "LEFT side."
            ),
        ))
    return issues


def _check_impression_measurements(text: str) -> list[Issue]:
    """
    ⚠️  A FIGURE IN THE IMPRESSION THAT IS NOT IN THE FINDINGS.
    This is the exact failure that started Phase 4: a report stating 50% was
    summarised as "25-50%", a number borrowed from background literature. An
    impression summarises the findings, so every measurement in it must
    already appear above.
    """
    findings, impression = _split_sections(text)
    if not impression.strip():
        return []

    in_findings = {m.group(0).lower().replace(" ", "")
                   for m in _MEASUREMENT_TOKEN.finditer(findings)}

    issues = []
    for m in _MEASUREMENT_TOKEN.finditer(impression):
        token = m.group(0).lower().replace(" ", "")
        if token not in in_findings:
            issues.append(Issue(
                code="impression_measurement_not_in_findings",
                severity=Severity.ERROR,
                message=(
                    f"The impression states {m.group(0)!r}, which does not "
                    f"appear in the findings. An impression summarises; it "
                    f"does not introduce measurements."
                ),
                excerpt=m.group(0),
            ))
    return issues


def _content_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]{4,}", text.lower())
        if w not in _STOPWORDS
    }


def _check_impression_introduces_findings(text: str) -> list[Issue]:
    """
    A clinical term appearing only in the impression.

    WARNING rather than ERROR, and deliberately conservative: an impression
    legitimately uses synonyms and umbrella terms ("degenerative change" for
    "osteophytes"). Only terms sharing no word stem with anything in the
    findings are reported, and only a handful at a time.
    """
    findings, impression = _split_sections(text)
    # ⚠️  STRIP THE DISCLAIMER FIRST.
    # It sits below the impression heading, so it lands in the impression
    # section — and its words ("draft", "radiologist", "review", "report",
    # "final") appear nowhere in the findings. Every correct report therefore
    # failed this rule. Boilerplate the system itself appended must never be
    # treated as the clinician's wording.
    impression = _DISCLAIMER.sub("", impression)

    if not findings.strip() or not impression.strip():
        return []

    f_words = _content_words(findings)

    # ⚠️  DON'T REPORT ONE PROBLEM TWICE.
    # A laterality contradiction already has its own dedicated ERROR. Letting
    # "left" also appear here listed it a second time as an "introduced term",
    # which reads as two separate defects and pads the count the reviewer
    # sees. Duplicate flags for a single problem are how a checker starts
    # becoming noise, and noise is what gets it switched off.
    _COVERED_ELSEWHERE = {"left", "right", "bilateral", "sided"}

    novel = []
    for word in sorted(_content_words(impression) - f_words - _COVERED_ELSEWHERE):
        # A shared five-character prefix covers most inflections
        # (cardiomegaly/cardiomegalic, degenerative/degeneration).
        if not any(w.startswith(word[:5]) or word.startswith(w[:5]) for w in f_words):
            novel.append(word)

    if not novel:
        return []

    return [Issue(
        code="impression_introduces_term",
        severity=Severity.WARNING,
        message=(
            "Terms appear in the impression but nowhere in the findings: "
            + ", ".join(novel[:6])
            + ". Check these are summaries rather than new conclusions."
        ),
    )]


_RULES = (
    _check_sections,
    _check_placeholders,
    _check_unitless_measurements,
    _check_stacked_hedging,
    _check_laterality,
    _check_impression_measurements,
    _check_impression_introduces_findings,
    _check_vague_sizing,
)


def check_report(text: str) -> QualityReport:
    """
    Run every rule. Never raises — a checker that can crash on a draft is
    worse than no checker, because it fails at exactly the moment someone
    typed something unusual.
    """
    if not text or not text.strip():
        return QualityReport(issues=[Issue(
            code="empty_report",
            severity=Severity.ERROR,
            message="The report is empty.",
        )])

    issues: list[Issue] = []
    for rule in _RULES:
        try:
            issues.extend(rule(text))
        except Exception:  # noqa: BLE001
            # One broken rule must not suppress the others.
            continue

    issues.sort(key=lambda i: (Severity.RANK.get(i.severity, 9), i.line or 0))
    return QualityReport(issues=issues)
