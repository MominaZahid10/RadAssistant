"use client";

/**
 * ReportEditor — the Review & Sign-off Layer (Phase 5, Step 0.5)
 *
 * The project document specifies it directly: *"the radiologist edits,
 * approves, or rejects"*, plus a design constraint that the system produces
 * drafts only and the UI must make the human-in-the-loop explicit.
 *
 * Before this, a draft arrived as a chat bubble. It could not be edited,
 * approved, or retrieved, and the only trace of human review was a line of
 * text the model had been told to print at the bottom. That is a claim, not a
 * mechanism.
 *
 * ⚠️  THE DRAFT IS PRE-SELECTED FOR EDITING, NOT LOCKED BEHIND A BUTTON.
 * A read-only pane with an "Edit" button implies the default is to accept.
 * For a document going into a patient record the default should be to read it
 * properly, so the text is editable from the moment it appears.
 *
 * ⚠️  APPROVAL SENDS THE TEXT WITH IT.
 * Not two requests. Approving is a claim about a specific wording, and
 * splitting them lets a slow save record a signature against text the
 * reviewer never saw.
 */

import { useEffect, useRef, useState } from "react";
import {
  reportApi,
  type QualityIssue,
  type Report,
  type ReportStatus,
} from "@/lib/api";

interface Props {
  /** The model's original output. Never modified here. */
  draft: string;
  /** The dictated findings this draft came from — needed to regenerate. */
  findingsInput: string;
  model?: string | null;
  sources?: Record<string, unknown>[] | null;
  /** Re-run generation from the same findings. */
  onRegenerate?: () => void;
}

export default function ReportEditor({
  draft,
  findingsInput,
  model,
  sources,
  onRegenerate,
}: Props) {
  const [text, setText] = useState(draft);
  const [report, setReport] = useState<Report | null>(null);
  const [status, setStatus] = useState<ReportStatus>("draft");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const areaRef = useRef<HTMLTextAreaElement>(null);

  // Grow to fit rather than scrolling inside a small box. A reviewer reading
  // for errors should see the whole document at once.
  useEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [text]);

  // The draft arrives token by token while streaming, so keep the editor in
  // sync until the user touches it — after which their edits win.
  const [touched, setTouched] = useState(false);
  useEffect(() => {
    if (!touched) setText(draft);
  }, [draft, touched]);

  const edited = text.trim() !== draft.trim();

  // ── Quality checks ───────────────────────────────────────────
  // ⚠️  ADVISORY, NEVER BLOCKING.
  // The rules are good but not omniscient, and a radiologist who cannot sign
  // a correct report because a regex disagrees will stop using the tool
  // entirely. Issues are shown; Approve stays enabled.
  const [issues, setIssues] = useState<QualityIssue[]>([]);

  useEffect(() => {
    if (!text.trim()) {
      setIssues([]);
      return;
    }
    // Debounced so it runs on pauses, not keystrokes. The check is pure
    // regex server-side — no model, no database — so 400ms is comfortable.
    const t = setTimeout(() => {
      reportApi
        .qualityCheck(text)
        .then((r) => setIssues(r.issues))
        // A failed check must not disturb review. Silence beats an error
        // banner over a report the radiologist is trying to read.
        .catch(() => setIssues([]));
    }, 400);
    return () => clearTimeout(t);
  }, [text]);

  const errorCount = issues.filter((i) => i.severity === "error").length;

  /** Create the row on first save; update it thereafter. */
  async function persist(nextStatus?: ReportStatus, note?: string) {
    setBusy(true);
    setError(null);
    try {
      let current = report;
      if (!current) {
        current = await reportApi.create({
          findings_input: findingsInput,
          ai_draft: draft,
          model: model ?? null,
          sources: sources ?? null,
        });
        setReport(current);
      }

      const updated = await reportApi.update(current.id, {
        // Only send text the reviewer actually changed. An unchanged draft
        // must stay NULL server-side: "accepted as written" is a distinct
        // state from "edited to be identical", and conflating them would
        // inflate the edit rate, which is the one measured signal of how
        // often the model gets things wrong.
        ...(edited ? { edited_text: text } : {}),
        ...(nextStatus ? { status: nextStatus } : {}),
        ...(note ? { review_note: note } : {}),
      });

      setReport(updated);
      setStatus(updated.status);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save");
    } finally {
      setBusy(false);
    }
  }

  async function copy() {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  const signedOff = status === "approved" || status === "rejected";

  return (
    <div className="report-editor">
      <div className="report-editor-bar">
        <span className={`report-status report-status--${status}`}>
          {status === "approved"
            ? "Approved"
            : status === "rejected"
            ? "Rejected"
            : "Draft — unreviewed"}
        </span>

        {edited && !signedOff && (
          <span className="report-edited-flag">edited</span>
        )}

        {report?.reviewed_at && (
          <span className="report-meta">
            {new Date(report.reviewed_at).toLocaleString()}
          </span>
        )}
      </div>

      <textarea
        ref={areaRef}
        value={text}
        onChange={(e) => {
          setTouched(true);
          setText(e.target.value);
        }}
        // Approved text is frozen. Reopening is deliberate — a signature that
        // stays valid while the words change underneath it is worthless.
        readOnly={status === "approved"}
        className="report-editor-text"
        spellCheck
      />

      {error && <p className="report-error">{error}</p>}

      {issues.length > 0 && (
        <ul className="report-issues">
          {issues.map((issue, n) => (
            <li key={`${issue.code}-${n}`} className={`report-issue report-issue--${issue.severity}`}>
              <span className="report-issue-dot" aria-hidden />
              <span>
                {issue.line != null && (
                  <span className="report-issue-line">line {issue.line}</span>
                )}
                {issue.message}
              </span>
            </li>
          ))}
        </ul>
      )}

      <div className="report-actions">
        {status === "approved" ? (
          <button
            className="report-btn"
            disabled={busy}
            onClick={() => persist("draft")}
            title="Unlock for further edits. The previous sign-off is cleared."
          >
            Reopen
          </button>
        ) : (
          <>
            <button
              className="report-btn report-btn--approve"
              // NOT disabled by outstanding issues — see the note above the
              // quality effect. The reviewer is told; the reviewer decides.
              disabled={busy || !text.trim()}
              onClick={() => persist("approved")}
              title={
                errorCount
                  ? `Sign off despite ${errorCount} unresolved issue${
                      errorCount === 1 ? "" : "s"
                    }`
                  : "Sign off on this wording"
              }
            >
              {busy
                ? "Saving…"
                : errorCount
                ? `Approve anyway (${errorCount})`
                : "Approve"}
            </button>

            <button
              className="report-btn report-btn--reject"
              disabled={busy}
              onClick={() => persist("rejected")}
              title="Reject this draft. It is kept as a record, not deleted."
            >
              Reject
            </button>

            <button
              className="report-btn"
              disabled={busy}
              onClick={() => persist()}
              title="Save edits without signing off"
            >
              Save draft
            </button>
          </>
        )}

        {onRegenerate && status !== "approved" && (
          <button
            className="report-btn"
            disabled={busy}
            onClick={onRegenerate}
            title="Generate a new draft from the same findings"
          >
            Regenerate
          </button>
        )}

        <button className="report-btn" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      {/* The constraint the project document makes explicit. Rendered by the
          UI rather than left to the model, which can be prompted to omit it
          and occasionally will. */}
      <p className="report-disclaimer">
        Draft for radiologist review — not a final report.
      </p>
    </div>
  );
}
