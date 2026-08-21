/**
 * RadAssistant — Chat page
 *
 * RAG-powered chat with:
 * - SSE streaming (token-by-token typewriter effect)
 * - Source citations panel (collapsible evidence)
 * - Audience toggle (attending / resident)
 * - Error handling for LLM failures
 *
 * The transcript itself is NOT local state. It lives in AppShell so the
 * sidebar and this page are looking at the same conversations — which is
 * what makes "new chat" and "go back to a previous chat" work at all.
 */
"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  api,
  imageApi,
  type Audience,
  type ChatMode,
  type MedicalImage,
} from "@/lib/api";
import { useRouter } from "next/navigation";
import ImageViewer from "@/components/ImageViewer";
import AuthedImage from "@/components/AuthedImage";
import ReportEditor from "@/components/ReportEditor";
import Brand from "@/components/Brand";
import { useApp } from "@/components/AppShell";
import type { StoredMessage } from "@/lib/chats";
import { AUTH_EXPIRED_EVENT, isSignedIn } from "@/lib/auth";

/**
 * A turn as this page renders it. Identical to the persisted shape — the
 * transcript that reloads from storage must render the same way the live one
 * does, so there is deliberately no second in-memory type to keep in sync.
 */
type Message = StoredMessage;

/**
 * Splits answer text on inline [N] citations and renders each as a clickable
 * chip that jumps to the matching source card.
 *
 * WHY THIS MATTERS:
 * The grounding prompt requires every factual claim to carry a citation.
 * Rendering them as plain text makes that requirement decorative — the
 * radiologist can see "[3]" but can't get to what [3] actually says. Being
 * able to trace a claim to its source in one click is the entire premise of
 * the product.
 *
 * ROBUSTNESS NOTES:
 * - Matches [1] and grouped forms like [1,2] or [1, 3].
 * - Also accepts 【N】 as a fallback. The backend normalises these (some
 *   models, notably gpt-oss, emit CJK lenticular brackets), but accepting
 *   both here means a normalisation gap degrades to "citation still works"
 *   rather than "evidence panel silently dead".
 * - Only linkifies numbers within range of the sources actually received.
 *   A model that hallucinates [9] against 5 sources renders it as plain
 *   text rather than a dead link.
 * - Runs mid-stream on partial text, so a half-written "[2" simply stays
 *   literal until the closing bracket arrives.
 */
/**
 * Minimal markdown renderer with inline citation support.
 *
 * WHY NOT react-markdown?
 * Two reasons. Citations need to render *inside* bold text, list items and
 * table cells — wiring that into a third-party renderer means custom AST
 * handlers and fighting its escaping. And this must tolerate half-finished
 * markdown, because it runs on every streamed token: a table with two of its
 * five rows written, or an unclosed `**`, must render sensibly rather than
 * throw. A purpose-built renderer handles both directly.
 *
 * Supports: headings, **bold**, *italic*, `code`, bullet and numbered lists,
 * GFM tables, blockquotes, horizontal rules — plus [N] citation chips.
 */

/** Inline spans: **bold**, *italic*, `code`, and [N] / 【N】 citations. */
function renderInline(
  text: string,
  sourceCount: number,
  onCitationClick: (n: number) => void,
  keyPrefix: string
): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const re =
    /(\*\*[^*]+\*\*)|(__[^_]+__)|(\*[^*\n]+\*)|(`[^`\n]+`)|([[【]\s*\d+(?:\s*,\s*\d+)*\s*[\]】])/g;

  let last = 0;
  let i = 0;
  let m: RegExpExecArray | null;

  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const tok = m[0];

    if (tok.startsWith("**") || tok.startsWith("__")) {
      nodes.push(<strong key={`${keyPrefix}b${i++}`}>{tok.slice(2, -2)}</strong>);
    } else if (tok.startsWith("`")) {
      nodes.push(
        <code key={`${keyPrefix}c${i++}`} className="md-code">
          {tok.slice(1, -1)}
        </code>
      );
    } else if (tok.startsWith("*")) {
      nodes.push(<em key={`${keyPrefix}i${i++}`}>{tok.slice(1, -1)}</em>);
    } else {
      // Citation. Only linkify numbers that map to a source we actually have —
      // a hallucinated [9] against 4 sources stays as plain text rather than
      // becoming a dead link.
      const nums = tok
        .replace(/[[\]【】\s]/g, "")
        .split(",")
        .map((n) => parseInt(n, 10))
        .filter((n) => Number.isFinite(n));

      const valid = nums.length > 0 && nums.every((n) => n >= 1 && n <= sourceCount);

      if (!valid) {
        nodes.push(tok);
      } else {
        nums.forEach((n) =>
          nodes.push(
            <button
              key={`${keyPrefix}cite${i++}`}
              type="button"
              className="citation-chip"
              title={`Jump to source ${n}`}
              aria-label={`Show source ${n}`}
              onClick={() => onCitationClick(n)}
            >
              {n}
            </button>
          )
        );
      }
    }
    last = m.index + tok.length;
  }

  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

const isTableRow = (l: string) => /^\s*\|.*\|\s*$/.test(l);
const isTableDivider = (l: string) => /^\s*\|[\s:|-]+\|\s*$/.test(l);
const splitRow = (l: string) =>
  l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

function MarkdownAnswer({
  text,
  sourceCount,
  onCitationClick,
}: {
  text: string;
  sourceCount: number;
  onCitationClick: (n: number) => void;
}) {
  if (!text) return null;

  const lines = text.split("\n");
  const blocks: React.ReactNode[] = [];
  let k = 0;
  let i = 0;

  const inline = (s: string, p: string) =>
    renderInline(s, sourceCount, onCitationClick, p);

  while (i < lines.length) {
    const line = lines[i];

    // ── blank ──
    if (!line.trim()) {
      i++;
      continue;
    }

    // ── horizontal rule ──
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      blocks.push(<hr key={`hr${k++}`} className="md-hr" />);
      i++;
      continue;
    }

    // ── heading ──
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      const Tag = (["h3", "h4", "h5", "h6"] as const)[level - 1];
      blocks.push(
        <Tag key={`h${k++}`} className={`md-h md-h${level}`}>
          {inline(h[2], `h${k}`)}
        </Tag>
      );
      i++;
      continue;
    }

    // ── table ──
    if (isTableRow(line) && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
      const header = splitRow(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && isTableRow(lines[i])) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      blocks.push(
        <div key={`tw${k++}`} className="md-table-wrap">
          <table className="md-table">
            <thead>
              <tr>
                {header.map((c, ci) => (
                  <th key={ci}>{inline(c, `th${k}${ci}`)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, ri) => (
                <tr key={ri}>
                  {r.map((c, ci) => (
                    <td key={ci}>{inline(c, `td${k}${ri}${ci}`)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      continue;
    }

    // A table still being streamed — header written, divider not yet.
    // Render as plain text for now; it becomes a table on the next token.
    if (isTableRow(line) && i + 1 >= lines.length) {
      blocks.push(
        <p key={`p${k++}`} className="md-p md-partial">
          {inline(line, `pp${k}`)}
        </p>
      );
      i++;
      continue;
    }

    // ── blockquote ──
    if (/^\s*>\s?/.test(line)) {
      const buf: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      blocks.push(
        <blockquote key={`bq${k++}`} className="md-quote">
          {inline(buf.join(" "), `bq${k}`)}
        </blockquote>
      );
      continue;
    }

    // ── unordered list ──
    if (/^\s*[-*+]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*+]\s+/, ""));
        i++;
      }
      blocks.push(
        <ul key={`ul${k++}`} className="md-ul">
          {items.map((it, ii) => (
            <li key={ii}>{inline(it, `li${k}${ii}`)}</li>
          ))}
        </ul>
      );
      continue;
    }

    // ── ordered list ──
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+[.)]\s+/, ""));
        i++;
      }
      blocks.push(
        <ol key={`ol${k++}`} className="md-ol">
          {items.map((it, ii) => (
            <li key={ii}>{inline(it, `oli${k}${ii}`)}</li>
          ))}
        </ol>
      );
      continue;
    }

    // ── paragraph (consume until blank line or a new block starts) ──
    const buf: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^\s*[-*+]\s+/.test(lines[i]) &&
      !/^\s*\d+[.)]\s+/.test(lines[i]) &&
      !/^\s*>/.test(lines[i]) &&
      !/^#{1,4}\s/.test(lines[i]) &&
      !isTableRow(lines[i])
    ) {
      buf.push(lines[i].trim());
      i++;
    }
    if (buf.length) {
      blocks.push(
        <p key={`p${k++}`} className="md-p">
          {inline(buf.join(" "), `pi${k}`)}
        </p>
      );
    } else {
      i++;
    }
  }

  return <div className="markdown-body">{blocks}</div>;
}

/**
 * A file staged in the composer but not yet sent.
 *
 * Uploading happens on SEND rather than on selection, so a user who attaches
 * something and then changes their mind hasn't written anything to the server.
 */
interface PendingAttachment {
  key: string;
  file: File;
  /** Local object URL for the preview — no round trip needed. */
  previewUrl: string;
}


/**
 * Figures belonging to a cited article, shown inside its source card.
 *
 * THE FIRST GENUINELY MULTIMODAL MOMENT IN THE PRODUCT: a citation you can
 * look at, not just read. The figure and its caption were written together by
 * the article's authors, so the caption is real alt text rather than a guess.
 *
 * WHY IT FETCHES ITSELF RATHER THAN ARRIVING WITH THE ANSWER:
 * Most answers cite papers with no figures, and the SSE `sources` event is on
 * the critical path — the evidence panel renders from it while tokens are
 * still streaming. Loading figures here means the answer is never delayed by
 * an image lookup, and articles the user never expands are never queried.
 *
 * Renders nothing at all on failure or when there are no figures. A missing
 * illustration must never disturb an answer that is already on screen.
 */
const figureCache = new Map<string, MedicalImage[]>();

function SourceFigures({
  documentId,
  onView,
}: {
  documentId: string | null;
  onView: (img: MedicalImage) => void;
}) {
  const [figures, setFigures] = useState<MedicalImage[]>(
    () => (documentId ? figureCache.get(documentId) ?? [] : [])
  );

  useEffect(() => {
    if (!documentId || figureCache.has(documentId)) return;

    let cancelled = false;
    imageApi.forDocument(documentId).then((imgs) => {
      const withThumbs = imgs.filter((i) => i.thumbnail_url);
      figureCache.set(documentId, withThumbs);
      // Guard against setting state after the panel is collapsed again —
      // React warns, and on a slow connection this fires constantly.
      if (!cancelled) setFigures(withThumbs);
    });

    return () => {
      cancelled = true;
    };
  }, [documentId]);

  if (figures.length === 0) return null;

  return (
    <div className="source-figures">
      {figures.map((fig) => (
        <button
          key={fig.id}
          type="button"
          className="source-figure"
          onClick={() => onView(fig)}
          // The caption IS the alt text — authored, not inferred.
          title={fig.caption ?? "Figure"}
        >
          {/* Fetched with the auth header — a plain <img src> cannot send
              one, so protecting the image routes broke every thumbnail. */}
          <AuthedImage
            path={fig.thumbnail_url}
            alt={fig.caption ?? "Figure from the cited article"}
          />
        </button>
      ))}
    </div>
  );
}

export default function ChatPage() {
  const {
    collapsed,
    setCollapsed,
    messages,
    setMessages,
    ensureConversation,
    activeConversation,
  } = useApp();

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [audience, setAudience] = useState<Audience>("radiologist");
  // Ask a question, or dictate findings and get a structured draft.
  const [mode, setMode] = useState<ChatMode>("qa");
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  // Files staged in the composer, not yet uploaded.
  const [pending, setPending] = useState<PendingAttachment[]>([]);
  const [viewing, setViewing] = useState<MedicalImage | null>(null);
  const router = useRouter();
  // null while we have not yet checked — avoids flashing the chat UI to a
  // signed-out user before the redirect lands.
  const [signedIn, setSignedIn] = useState<boolean | null>(null);
  // Surfaced under the composer. Attachment used to fail with no feedback at
  // all, which is indistinguishable from the click not registering.
  const [attachError, setAttachError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // Lets regenerateReport call handleSend, which is defined further down.
  const sendRef = useRef<(() => void) | null>(null);

  // ── Session guard ─────────────────────────────────────────
  // ⚠️  A CLIENT-SIDE REDIRECT IS NOT THE SECURITY BOUNDARY.
  // It is a courtesy: without it a signed-out user sees a chat window where
  // every action fails with a 401 they cannot interpret. The actual control
  // is server-side — every clinical route requires a token, so bypassing this
  // redirect gets you an empty interface and nothing else.
  useEffect(() => {
    const present = isSignedIn();
    setSignedIn(present);
    if (!present) router.replace("/login");

    // Fired by lib/auth when a 401 clears the token mid-session — twelve
    // hours in, halfway through a report.
    const onExpired = () => {
      setSignedIn(false);
      router.replace("/login");
    };
    window.addEventListener(AUTH_EXPIRED_EVENT, onExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, onExpired);
  }, [router]);

  // Auto-scroll to the bottom as messages arrive, and when switching
  // conversations — reopening a chat should land at the end of it.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 180) + "px";
    }
  }, [input]);

  /**
   * Re-run generation from the same dictated findings.
   *
   * Sets the composer and defers the send by a tick so `input` and `mode` are
   * committed before handleSend reads them — calling it synchronously would
   * send the previous input, which is the kind of bug that only shows up when
   * someone regenerates twice.
   */
  const regenerateReport = useCallback((findings: string) => {
    if (!findings.trim()) return;
    setMode("report");
    setInput(findings);
    setTimeout(() => sendRef.current?.(), 0);
  }, []);

  const toggleSources = useCallback((messageId: string) => {
    setExpandedSources((prev) => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  }, []);

  // ── Attachments ────────────────────────────────────────────

  const addFiles = (files: FileList | File[] | null) => {
    // ⚠️  THIS PATH USED TO FAIL SILENTLY, TWICE OVER.
    // Nothing here throws visibly: if staging drops a file, the user sees an
    // unchanged composer and no explanation. Both known causes are handled,
    // and anything left surfaces as a message rather than nothing.
    if (!files) {
      setAttachError("No file was received from the picker.");
      return;
    }

    // ⚠️  MATERIALISE THE LIST *BEFORE* setPending, NOT INSIDE THE UPDATER.
    // `input.files` is a LIVE FileList bound to the element. The onChange
    // handler resets `e.target.value = ""` so the same file can be chosen
    // twice — and that reset EMPTIES the FileList. React runs a functional
    // updater asynchronously, after the handler returns, so `Array.from()`
    // inside the updater ran against a list that had already been cleared.
    const list = Array.from(files as ArrayLike<File>);
    if (!list.length) {
      setAttachError("The picker returned no files.");
      return;
    }

    try {
      const staged = list.map((file) => ({
        key: `${file.name}-${file.size}-${Date.now()}-${Math.random()}`,
        file,
        previewUrl: URL.createObjectURL(file),
      }));
      setPending((prev) => [...prev, ...staged]);
      setAttachError(null);
    } catch (e) {
      setAttachError(
        e instanceof Error
          ? `Could not attach: ${e.message}`
          : "Could not attach that file."
      );
    }
  };

  // Drag a file straight onto the composer — an independent path to the same
  // staging code, so a broken file picker is not a dead end.
  const [dragging, setDragging] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  };

  const removePending = (key: string) => {
    setPending((prev) => {
      const target = prev.find((p) => p.key === key);
      // Object URLs leak until revoked — the browser holds the whole file
      // in memory otherwise.
      if (target) URL.revokeObjectURL(target.previewUrl);
      return prev.filter((p) => p.key !== key);
    });
  };

  // Paste an image straight from the clipboard, like ChatGPT/Claude.
  const handlePaste = (e: React.ClipboardEvent) => {
    const files = Array.from(e.clipboardData.files);
    if (files.length) {
      e.preventDefault();
      addFiles(e.clipboardData.files);
    }
  };

  const handleSend = async (): Promise<void> => {
    const trimmed = input.trim();
    const attachments = pending;

    // Either text or an attachment is enough to send.
    if ((!trimmed && attachments.length === 0) || isLoading) return;

    // ⚠️  BEFORE THE FIRST setMessages, AND CAPTURED.
    // Two things depend on this. On a fresh install there is no conversation
    // yet, so one has to exist before any write lands. And every write below
    // is pinned to THIS id rather than to whatever is active when it fires —
    // a send outlives the click that started it, and the user is free to open
    // another conversation while the answer is still streaming. Writing to
    // "the active chat" would drop those tokens on the floor.
    const convId = ensureConversation();

    // Bound once rather than passing convId to each of the writes below —
    // eight call sites that must not disagree about their destination is
    // eight chances to miss one.
    const write = (update: Parameters<typeof setMessages>[0]): void =>
      setMessages(update, convId);

    const userMsgId = Date.now().toString();
    const userMsg: Message = {
      id: userMsgId,
      role: "user",
      content: trimmed,
    };

    const aiMsgId = (Date.now() + 1).toString();
    const aiMsg: Message = {
      id: aiMsgId,
      role: "assistant",
      content: "",
      isStreaming: true,
      // Captured now, not read back from the preceding bubble later — the
      // editor needs the exact input to regenerate from.
      mode,
      findingsInput: trimmed,
    };

    write((prev) => [...prev, userMsg, aiMsg]);
    setInput("");
    setPending([]);
    setIsLoading(true);

    // ── Upload attachments before asking the question ──
    // Uploading on SEND rather than on selection means a user who attaches
    // something and then changes their mind hasn't written to the server.
    let query = trimmed;
    let attachedText = "";
    let attachedWarnings: string[] = [];
    // Populated in Compare mode from the attached prior study.
    let priorText = "";

    if (attachments.length) {
      write((prev) =>
        prev.map((m) =>
          m.id === aiMsgId
            ? { ...m, content: `Reading ${attachments.length} attachment(s)…` }
            : m
        )
      );

      const uploaded: MedicalImage[] = [];
      for (const att of attachments) {
        try {
          // Treat every attachment as a report photo so its text is
          // extracted. The backend still detects DICOM from magic bytes and
          // overrides this — a clinician shouldn't have to classify the file.
          const accepted = await imageApi.upload(att.file, {
            sourceType: "report_upload",
          });
          uploaded.push(await imageApi.waitForProcessing(accepted.id));
        } catch (e) {
          console.error("Attachment failed:", e);
        } finally {
          URL.revokeObjectURL(att.previewUrl);
        }
      }

      write((prev) =>
        prev.map((m) => (m.id === userMsgId ? { ...m, images: uploaded } : m))
      );

      // ⚠️  SENT SEPARATELY, NOT CONCATENATED INTO THE QUESTION.
      // Appending the report to the query made the model treat retrieved
      // papers as authoritative and the patient's own report as loose
      // material — it inverted 'hyperlordotic' to 'hypolordotic' and
      // changed a stated 50% to '25-50%'. As a separate field the backend
      // places it above the literature and names it the primary source.
      attachedText = uploaded
        .filter((u) => u.ocr_text)
        .map((u) => `--- ${u.filename} ---\n${u.ocr_text}`)
        .join("\n\n");

      // OCR caveats travel with the text so the model flags unreliable
      // passages instead of stating misread words as fact.
      attachedWarnings = uploaded
        .filter((u) => u.description)
        .map((u) => u.description as string);

      if (!trimmed) {
        query = attachedText
          ? "Please review the attached report."
          : "I've attached an image. What can you tell me about it?";
      }

      // ── Compare mode: the attachment is the PRIOR study ──
      // This mirrors the actual workflow. The radiologist has the previous
      // report on file and is dictating the current one; they are not
      // uploading both. So the attachment becomes prior_text and the typed
      // findings stay as the query, which is what the backend expects.
      if (mode === "comparison" && attachedText) {
        priorText = attachedText;
        attachedText = "";
        attachedWarnings = [];
      }

      // Clear the placeholder before streaming begins.
      write((prev) =>
        prev.map((m) => (m.id === aiMsgId ? { ...m, content: "" } : m))
      );
    }

    try {
      // Stream the response from the RAG pipeline
      for await (const event of api.streamChat(query, {
        audience,
        mode,
        includeSources: true,
        attachedText: attachedText || undefined,
        attachedWarnings: attachedWarnings.length ? attachedWarnings : undefined,
        priorText: priorText || undefined,
      })) {
        switch (event.type) {
          case "sources":
            // Sources arrive first — attach them to the AI message
            write((prev) =>
              prev.map((msg) =>
                msg.id === aiMsgId ? { ...msg, sources: event.sources } : msg
              )
            );
            break;

          case "token":
            // Append each token to the AI message content
            write((prev) =>
              prev.map((msg) =>
                msg.id === aiMsgId
                  ? { ...msg, content: msg.content + event.token }
                  : msg
              )
            );
            break;

          case "done":
            // Mark streaming as complete, record the model used
            write((prev) =>
              prev.map((msg) =>
                msg.id === aiMsgId
                  ? { ...msg, isStreaming: false, model: event.model }
                  : msg
              )
            );
            break;

          case "error":
            // Show the error in the AI message
            write((prev) =>
              prev.map((msg) =>
                msg.id === aiMsgId
                  ? {
                      ...msg,
                      content: event.error,
                      isStreaming: false,
                      isError: true,
                    }
                  : msg
              )
            );
            break;
        }
      }
    } catch (err) {
      // Network error or failed to connect
      const errorMessage =
        err instanceof Error ? err.message : "Failed to connect to the server";
      write((prev) =>
        prev.map((msg) =>
          msg.id === aiMsgId
            ? {
                ...msg,
                content: errorMessage,
                isStreaming: false,
                isError: true,
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Kept current so Regenerate calls the latest closure rather than the one
  // captured when the component first mounted.
  sendRef.current = handleSend;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Nothing is rendered until the session check has run. Showing the chat to
  // a signed-out user for a frame and then redirecting looks like a glitch,
  // and briefly displays an interface where every action would 401.
  if (signedIn === null) return null;
  if (!signedIn) return null;

  const canSend = (input.trim().length > 0 || pending.length > 0) && !isLoading;
  const isEmpty = messages.length === 0;

  return (
    <>
      {/* ── Top bar ──
          The Ask/Draft/Compare control lives up here rather than inside the
          composer. It selects what the assistant PRODUCES, which is a
          property of the conversation, not of the message being typed —
          and moving it out is most of what un-crowded the composer. */}
      <header className="topbar">
        {collapsed && (
          <button
            className="topbar-expand"
            onClick={() => setCollapsed(false)}
            title="Open sidebar"
            aria-label="Open sidebar"
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
            >
              <rect x="3" y="4" width="18" height="16" rx="2.5" />
              <path d="M9.5 4v16" />
            </svg>
          </button>
        )}

        <h1 className="topbar-title">
          {activeConversation && !isEmpty ? activeConversation.title : "New chat"}
        </h1>

        <div className="segmented">
          <button
            className={mode === "qa" ? "active" : ""}
            onClick={() => setMode("qa")}
            title="Answer a question from the knowledge base"
          >
            Ask
          </button>
          <button
            className={mode === "report" ? "active" : ""}
            onClick={() => setMode("report")}
            title="Draft a structured report from dictated findings"
          >
            Draft
          </button>
          <button
            className={mode === "comparison" ? "active" : ""}
            onClick={() => setMode("comparison")}
            title="Attach the prior study, dictate the current findings"
          >
            Compare
          </button>
        </div>
      </header>

      {/* ── Thread ── */}
      <div className="thread">
        {isEmpty ? (
          <div className="welcome">
            <Brand size={52} className="welcome-mark" />
            <h2>How can I help?</h2>
            <p>Ask a question, draft a report, or compare against a prior study.</p>

            <div className="welcome-prompts">
              {[
                "Radiographic findings of pneumothorax",
                "Fleischner criteria for pulmonary nodules",
                "CTPA findings for pulmonary embolism",
                "ABCDE approach to chest X-ray",
              ].map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => {
                    setInput(prompt);
                    textareaRef.current?.focus();
                  }}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="thread-inner">
            {messages.map((msg) =>
              msg.role === "user" ? (
                /* ── The user's own turn: filled, right-aligned ──
                   No "You" caption. Alignment and fill already say who is
                   speaking; the label was a third piece of furniture
                   repeating what the shape communicates. */
                <div key={msg.id} className="msg-row msg-row--user">
                  <div style={{ maxWidth: "78%" }}>
                    {msg.content && (
                      <div className="bubble-user">{msg.content}</div>
                    )}

                    {msg.images && msg.images.length > 0 && (
                      <div className="msg-attachments" style={{ justifyContent: "flex-end" }}>
                        {msg.images.map((img) => (
                          <button
                            key={img.id}
                            className="msg-attachment"
                            onClick={() => setViewing(img)}
                            title={
                              img.ocr_text
                                ? `${img.filename} — ${img.ocr_text.length} chars of text extracted`
                                : img.filename
                            }
                          >
                            {img.thumbnail_url ? (
                              <AuthedImage
                                path={img.thumbnail_url}
                                alt={img.filename}
                              />
                            ) : (
                              <span className="msg-attachment-fallback">🩻</span>
                            )}
                            {img.ocr_text && (
                              <span
                                className="msg-attachment-ocr"
                                title="Text was extracted and included in the question"
                              >
                                T
                              </span>
                            )}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                /* ── The assistant's turn ── */
                <div key={msg.id} className="msg-assistant">
                  <div className="assistant-mark">
                    <Brand size={17} />
                  </div>

                  <div className="assistant-body">
                    {/* A finished report draft becomes an editable, signable
                        document rather than a chat bubble — so it supplies
                        its own frame and the bubble goes bare. Still
                        streaming, it renders as text: an editor whose
                        contents rewrite themselves mid-keystroke is not
                        usable. */}
                    <div
                      className={`bubble-assistant${
                        msg.isError ? " bubble-assistant--error" : ""
                      }${
                        msg.mode === "report" && !msg.isStreaming && !msg.isError
                          ? " bubble-assistant--bare"
                          : ""
                      }`}
                    >
                      {msg.mode === "report" &&
                      !msg.isStreaming &&
                      !msg.isError &&
                      msg.content.trim() ? (
                        <ReportEditor
                          draft={msg.content}
                          findingsInput={msg.findingsInput ?? ""}
                          model={msg.model}
                          sources={
                            msg.sources as unknown as Record<string, unknown>[]
                          }
                          onRegenerate={() =>
                            regenerateReport(msg.findingsInput ?? "")
                          }
                        />
                      ) : (
                        <MarkdownAnswer
                          text={msg.content}
                          sourceCount={msg.sources?.length ?? 0}
                          onCitationClick={(n) => {
                            // Make sure the panel is open, then scroll to it.
                            setExpandedSources((prev) => {
                              const next = new Set(prev);
                              next.add(msg.id);
                              return next;
                            });
                            // Wait a frame so the panel exists in the DOM.
                            requestAnimationFrame(() => {
                              const el = document.getElementById(
                                `source-${msg.id}-${n}`
                              );
                              el?.scrollIntoView({
                                behavior: "smooth",
                                block: "center",
                              });
                              el?.classList.add("source-card--flash");
                              setTimeout(
                                () => el?.classList.remove("source-card--flash"),
                                1200
                              );
                            });
                          }}
                        />
                      )}

                      {msg.isStreaming && <span className="stream-caret" />}

                      {/* Sources sit inside the answer card: the evidence
                          belongs to the claim, not beside it. */}
                      {msg.sources && msg.sources.length > 0 && !msg.isStreaming && (
                        <>
                          <button
                            onClick={() => toggleSources(msg.id)}
                            className="sources-toggle"
                          >
                            <svg
                              width="12"
                              height="12"
                              viewBox="0 0 12 12"
                              fill="none"
                              className={`transition-transform duration-200 ${
                                expandedSources.has(msg.id) ? "rotate-90" : ""
                              }`}
                            >
                              <path
                                d="M4.5 2.5L8 6L4.5 9.5"
                                stroke="currentColor"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                            </svg>
                            <span>
                              {msg.sources.length} source
                              {msg.sources.length !== 1 ? "s" : ""}
                            </span>
                          </button>

                          {expandedSources.has(msg.id) && (
                            <div className="sources-panel">
                              {msg.sources.map((src) => (
                                <div
                                  key={src.chunk_id}
                                  id={`source-${msg.id}-${src.chunk_id}`}
                                  className="source-card"
                                >
                                  <div className="source-header">
                                    <span className="source-badge">
                                      [{src.chunk_id}]
                                    </span>
                                    <span className="source-title">
                                      {src.document_title || "Unknown source"}
                                    </span>
                                    <span className="source-score">
                                      {(src.score * 100).toFixed(0)}%
                                    </span>
                                  </div>
                                  {src.source_type && (
                                    <span className="source-type">
                                      {src.source_type}
                                    </span>
                                  )}
                                  <p className="source-text">{src.text}</p>

                                  <SourceFigures
                                    documentId={src.document_id ?? null}
                                    onView={setViewing}
                                  />
                                </div>
                              ))}
                            </div>
                          )}
                        </>
                      )}
                    </div>

                    {msg.model && !msg.isStreaming && (
                      <p className="msg-meta">via {msg.model}</p>
                    )}
                  </div>
                </div>
              )
            )}

            {/* Waiting for the first token */}
            {isLoading &&
              messages.length > 0 &&
              messages[messages.length - 1].role === "assistant" &&
              messages[messages.length - 1].content === "" &&
              !messages[messages.length - 1].isError && (
                <div className="msg-assistant" style={{ marginTop: -14 }}>
                  <div style={{ width: 30, flexShrink: 0 }} />
                  <div className="flex gap-1 items-center text-xs text-foreground-muted">
                    <span className="flex gap-1 mr-2">
                      <span className="w-1.5 h-1.5 bg-foreground-muted rounded-full dot-1" />
                      <span className="w-1.5 h-1.5 bg-foreground-muted rounded-full dot-2" />
                      <span className="w-1.5 h-1.5 bg-foreground-muted rounded-full dot-3" />
                    </span>
                    Searching the knowledge base…
                  </div>
                </div>
              )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* ── Composer ── */}
      <div className="composer-wrap">
        {/* Staged attachments, above the input */}
        {pending.length > 0 && (
          <div className="attach-tray">
            {pending.map((att) => (
              <div key={att.key} className="attach-chip">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={att.previewUrl} alt={att.file.name} />
                <button
                  onClick={() => removePending(att.key)}
                  className="attach-remove"
                  aria-label={`Remove ${att.file.name}`}
                >
                  ✕
                </button>
                <span className="attach-name">{att.file.name}</span>
              </div>
            ))}
          </div>
        )}

        {attachError && (
          <p className="attach-error">
            {attachError}{" "}
            <button onClick={() => setAttachError(null)}>dismiss</button>
          </p>
        )}

        <div
          className="composer"
          data-dragging={dragging}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={
              pending.length
                ? "Ask about the attached file, or just send…"
                : mode === "report"
                ? "Dictate findings — e.g. Mild cardiomegaly. No pleural effusion."
                : mode === "comparison"
                ? "Attach the prior study, then dictate the current findings…"
                : "Ask a radiology question…"
            }
            rows={1}
          />

          <div className="composer-bar">
            {/* Attach — DICOM, report photos, any image */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              // ⚠️  POSITIONED OFF-SCREEN, NOT `hidden`.
              // `hidden` sets display:none, and a display:none input is the
              // least reliable way to do this — browsers and extensions treat
              // it inconsistently, and a change event that never fires looks
              // exactly like a click that never registered. An off-screen but
              // rendered input is the standard robust pattern.
              style={{
                position: "absolute",
                width: 1,
                height: 1,
                opacity: 0,
                pointerEvents: "none",
              }}
              tabIndex={-1}
              // ⚠️  NO `accept` FILTER.
              // It was `.dcm,.dicom,image/*,application/octet-stream`. On
              // Windows the picker then silently refuses files whose MIME type
              // the OS reports differently — you select the file, the dialog
              // closes, and nothing arrives. The backend already sniffs the
              // CONTENT rather than trusting the extension, so filtering here
              // only ever rejects files the server would have handled.
              onChange={(e) => {
                const chosen = e.target.files ? Array.from(e.target.files) : [];
                // Copy out BEFORE resetting — the reset empties the live list.
                e.target.value = "";
                addFiles(chosen);
              }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="attach-button"
              title="Attach an image, report photo, or DICOM file"
              aria-label="Attach a file"
            >
              <svg
                width="17"
                height="17"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
              </svg>
            </button>

            {/* Register. Only meaningful when answering a question — a report
                goes into a medical record, and its register is fixed by
                reporting convention rather than by who is reading it. One
                button that toggles, rather than a two-option segmented
                control, because it is a binary the user rarely touches. */}
            <button
              className="composer-pill"
              disabled={mode === "report"}
              onClick={() =>
                setAudience((a) =>
                  a === "radiologist" ? "resident" : "radiologist"
                )
              }
              title={
                mode === "report"
                  ? "Not applicable when drafting a report"
                  : audience === "radiologist"
                  ? "Concise, standard terminology. Click for step-by-step."
                  : "Step-by-step reasoning, defines terms. Click for concise."
              }
            >
              {audience === "radiologist" ? "Attending" : "Resident"}
            </button>

            <button
              onClick={handleSend}
              disabled={!canSend}
              className="composer-send"
              title="Send"
              aria-label="Send message"
            >
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            </button>
          </div>
        </div>

        <p className="composer-note">
          Decision support only — verify before clinical use.
        </p>
      </div>

      {/* Full-screen viewer for an attached image */}
      {viewing && (
        <ImageViewer image={viewing} onClose={() => setViewing(null)} />
      )}
    </>
  );
}
