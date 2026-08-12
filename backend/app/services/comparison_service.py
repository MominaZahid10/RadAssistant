"""
RadAssist AI — Prior-Report Comparison (Phase 5, Step 4)

Pairs measurements between a prior study and the current one, so a model
summarising the two cannot invent or alter a figure.

Closes the outstanding Phase 4 deliverable — *"prior-report PDF/DOCX upload and
comparison against current findings"* — as well as Phase 5's patient timeline
item.

════════════════════════════════════════════════════════════════════
⚠️  THE ONE THING THIS MUST NOT DO
════════════════════════════════════════════════════════════════════
A nodule reported as 8mm previously and 9mm now is EITHER interval growth OR
inter-reader variation OR a different measurement axis on a different slice.
The two reports cannot distinguish these, and neither can a model reading them.

    "8mm previously, 9mm now"        ← what the documents support
    "1mm interval growth"            ← a clinical judgement they do not
    "mild progression"               ← worse: adds a trajectory

1mm on a small nodule is within measurement variability. Calling it growth can
trigger a biopsy. Calling stable disease "progression" changes treatment. The
system states both numbers and stops.

════════════════════════════════════════════════════════════════════
WHY MEASUREMENTS ARE EXTRACTED BY REGEX BEFORE THE MODEL SEES THEM
════════════════════════════════════════════════════════════════════
This is the same failure that opened Phase 4: a report stating 50% came back
as "25-50%", a number borrowed from surrounding literature. Numbers are the
part of a report a language model is least reliable with and where an error
is most consequential.

So the arithmetic is done here, deterministically, and handed to the model as
established fact. The model's job is to narrate what is already computed, not
to compute it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Units we can compare. Anything else is reported but never differenced —
# guessing at a conversion is exactly the sort of silent transformation this
# module exists to prevent.
_CONVERTIBLE_TO_MM = {
    "mm": 1.0,
    "millimetre": 1.0,
    "millimeter": 1.0,
    "cm": 10.0,
    "centimetre": 10.0,
    "centimeter": 10.0,
}

# ⚠️  '%' CANNOT CARRY A TRAILING \b — the same trap as in quality_service.
# A word boundary needs a word character on one side, and the character after
# '%' is almost always a space. `(?:mm|cm|%)\b` therefore never matched a
# percentage, so "50% height loss" extracted nothing at all. Only the
# alphabetic units take the boundary.
_MEASUREMENT = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>(?:mm|cm|millimet(?:re|er)s?|centimet(?:re|er)s?|percent)\b|%)",
    re.I,
)

# ⚠️  A DECIMAL POINT IS NOT A SENTENCE BOUNDARY.
# Splitting on a bare "." truncated the context of every decimal measurement:
# "Nodule measuring 1.2 cm in the right upper lobe" became "Nodule measuring
# 1". That collapsed the context words to {"nodule"}, dropped the similarity
# score below the pairing threshold, and silently stopped 1.2cm lesions from
# ever matching their follow-up — on a corpus where decimals are everywhere.
_SENTENCE_END = re.compile(r"(?<!\d)[.;](?!\d)|\n")

# Words that carry the anatomy a measurement belongs to. Used to decide
# whether two measurements describe the same thing.
_STOPWORDS = frozenset("""
a an and are as at be by for from has have in into is it its measuring measures
no not of on or that the there these this to up was were with within about
approximately size seen noted demonstrates shows there compared previously now
previous prior current interval study exam examination
""".split())


@dataclass
class Measurement:
    """One measurement, with the words around it."""
    value: float
    unit: str
    raw: str            # exactly as written
    context: str        # the sentence it came from
    line: int


@dataclass
class MeasurementPair:
    """
    The same finding measured twice.

    ⚠️  `delta` IS ARITHMETIC, NOT INTERPRETATION.
    It is the difference between two printed numbers. It is deliberately NOT
    called growth, progression, response, or change — those are clinical
    judgements the documents do not support.
    """
    prior: Measurement
    current: Measurement
    delta_mm: float | None = None       # None when units aren't comparable
    comparable: bool = True
    note: str = ""

    @property
    def identical(self) -> bool:
        return self.prior.raw.lower().replace(" ", "") == \
               self.current.raw.lower().replace(" ", "")


@dataclass
class ComparisonFacts:
    """Everything established without a model."""
    pairs: list[MeasurementPair] = field(default_factory=list)
    prior_only: list[Measurement] = field(default_factory=list)
    current_only: list[Measurement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Set when the two documents look like different studies entirely.
    mismatch_warning: str = ""


def _context_words(sentence: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]{3,}", sentence.lower())
        if w not in _STOPWORDS
    }


def _sentence_around(text: str, index: int) -> str:
    """
    The sentence containing the character at `index`.

    Boundaries come from _SENTENCE_END, which ignores periods sitting between
    digits — see the note there. This context is what decides whether two
    measurements describe the same finding, so truncating it silently breaks
    pairing rather than raising anything.
    """
    start = 0
    for m in _SENTENCE_END.finditer(text, 0, index):
        start = m.end()

    end = len(text)
    tail = _SENTENCE_END.search(text, index)
    if tail:
        end = tail.start()

    return text[start:end].strip()


def extract_measurements(text: str) -> list[Measurement]:
    """Every measurement in a report, with the sentence it sits in."""
    out: list[Measurement] = []
    for m in _MEASUREMENT.finditer(text or ""):
        unit = m.group("unit").lower()
        if unit == "percent":
            unit = "%"
        out.append(Measurement(
            value=float(m.group("value")),
            unit=unit,
            raw=m.group(0).strip(),
            context=_sentence_around(text, m.start()),
            line=text.count("\n", 0, m.start()) + 1,
        ))
    return out


def _to_mm(m: Measurement) -> float | None:
    factor = _CONVERTIBLE_TO_MM.get(m.unit)
    return m.value * factor if factor else None


def _similarity(a: Measurement, b: Measurement) -> float:
    """Jaccard overlap of the words around each measurement."""
    wa, wb = _context_words(a.context), _context_words(b.context)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# Below this, two measurements are assumed to describe different findings.
# Set conservatively: pairing a liver lesion with a lung nodule and reporting
# a "delta" between them would be far worse than leaving both unpaired, where
# they simply appear as prior-only and current-only.
_MIN_CONTEXT_OVERLAP = 0.30


def compare_measurements(prior_text: str, current_text: str) -> ComparisonFacts:
    """
    Pair measurements across two reports and compute literal differences.

    Greedy best-match on surrounding wording. Anything that cannot be matched
    confidently is listed separately rather than forced into a pair — an
    unmatched measurement is a prompt for the radiologist to look, whereas a
    wrong pair is a fabricated comparison.
    """
    prior = extract_measurements(prior_text)
    current = extract_measurements(current_text)

    facts = ComparisonFacts()
    used_current: set[int] = set()

    for p in prior:
        best_index, best_score = -1, 0.0
        for i, c in enumerate(current):
            if i in used_current:
                continue
            score = _similarity(p, c)
            if score > best_score:
                best_index, best_score = i, score

        if best_index == -1 or best_score < _MIN_CONTEXT_OVERLAP:
            facts.prior_only.append(p)
            continue

        c = current[best_index]
        used_current.add(best_index)

        pair = MeasurementPair(prior=p, current=c)

        if p.unit == "%" or c.unit == "%":
            # Percentages are only comparable to percentages, and even then
            # only if they describe the same quantity.
            pair.comparable = p.unit == c.unit
            if pair.comparable:
                pair.delta_mm = None
                pair.note = f"{p.raw} → {c.raw}"
            else:
                pair.note = "Units differ; not compared."
        else:
            p_mm, c_mm = _to_mm(p), _to_mm(c)
            if p_mm is None or c_mm is None:
                pair.comparable = False
                pair.note = "Unrecognised unit; not compared."
            else:
                pair.delta_mm = round(c_mm - p_mm, 2)
                pair.note = f"{p.raw} → {c.raw}"
                if p.unit != c.unit:
                    # Real and common: 1.2cm previously, 13mm now. Converting
                    # is safe here because both units are explicit — but the
                    # reader is told, because a silent conversion is where a
                    # tenfold error hides.
                    facts.warnings.append(
                        f"Units differ between studies ({p.raw} vs {c.raw}); "
                        f"compared in millimetres."
                    )

        facts.pairs.append(pair)

    facts.current_only = [c for i, c in enumerate(current) if i not in used_current]

    # ⚠️  ARE THESE EVEN THE SAME BODY PART?
    # Comparing against the wrong prior is a real clinical error — studies get
    # mis-selected from a worklist, and a comparison that silently produces
    # "everything new, everything gone" looks like a dramatic interval change
    # rather than a mismatched pair of documents.
    #
    # Checked on shared vocabulary, which is blunt but sufficient: a lumbar
    # spine report and a chest report have almost no anatomical words in
    # common, whereas two studies of the same region share many.
    facts.mismatch_warning = _study_mismatch(prior_text, current_text)
    if facts.mismatch_warning:
        facts.warnings.append(facts.mismatch_warning)

    return facts


# Below this overlap the two documents probably describe different studies.
# Deliberately low: two reports of the same region written months apart by
# different readers still share anatomy, but a false "different study" warning
# on a legitimate comparison would undermine the whole feature.
_MIN_STUDY_OVERLAP = 0.08


def _study_mismatch(prior_text: str, current_text: str) -> str:
    """Warn when the two documents look like different studies. '' if fine."""
    p_words = _context_words(prior_text or "")
    c_words = _context_words(current_text or "")

    # Too little text to judge. Silence beats a guess.
    if len(p_words) < 5 or len(c_words) < 5:
        return ""

    overlap = len(p_words & c_words) / len(p_words | c_words)
    if overlap >= _MIN_STUDY_OVERLAP:
        return ""

    return (
        f"These two reports share almost no vocabulary (overlap "
        f"{overlap:.0%}) and may describe DIFFERENT studies or body regions. "
        f"Check the prior report is the right one before reading anything "
        f"into what appears new or missing."
    )


def format_facts(facts: ComparisonFacts) -> str:
    """
    Render the established numbers for the prompt.

    Handed to the model as settled fact so it narrates rather than computes.
    Phrased throughout as "reported as X, reported as Y" — never as a
    characterisation — so there is no wording in the prompt for the model to
    copy that would imply a clinical judgement.
    """
    # The mismatch warning must survive even when neither report contains a
    # measurement — that is precisely the case where the reader has no other
    # signal that the wrong prior was selected.
    if not (facts.pairs or facts.prior_only or facts.current_only):
        if facts.mismatch_warning:
            return (
                "⚠️  STUDY MISMATCH — SAY THIS FIRST, BEFORE ANY COMPARISON:\n"
                f"  {facts.mismatch_warning}"
            )
        return ""

    lines = []
    if facts.mismatch_warning:
        lines.append("⚠️  STUDY MISMATCH — SAY THIS FIRST, BEFORE ANY COMPARISON:")
        lines.append(f"  {facts.mismatch_warning}")
        lines.append("")

    lines.append("MEASUREMENTS ESTABLISHED BY DIRECT COMPARISON (do not recompute):")

    for pair in facts.pairs:
        if pair.identical:
            lines.append(
                f"  - Reported as {pair.prior.raw} previously and "
                f"{pair.current.raw} now — the same value."
            )
        elif pair.comparable and pair.delta_mm is not None:
            direction = "larger" if pair.delta_mm > 0 else "smaller"
            lines.append(
                f"  - Reported as {pair.prior.raw} previously and "
                f"{pair.current.raw} now "
                f"({abs(pair.delta_mm):g}mm {direction} as printed)."
            )
        else:
            lines.append(
                f"  - Reported as {pair.prior.raw} previously and "
                f"{pair.current.raw} now. {pair.note}"
            )
        lines.append(f"    prior context:   {pair.prior.context[:110]}")
        lines.append(f"    current context: {pair.current.context[:110]}")

    for m in facts.prior_only:
        lines.append(f"  - {m.raw} appears only in the PRIOR report: {m.context[:110]}")
    for m in facts.current_only:
        lines.append(f"  - {m.raw} appears only in the CURRENT report: {m.context[:110]}")

    for w in facts.warnings:
        lines.append(f"  ! {w}")

    lines.append(
        "  NOTE: these are printed values. Whether a difference represents "
        "true interval change or measurement variation is NOT established by "
        "these documents."
    )
    return "\n".join(lines)
