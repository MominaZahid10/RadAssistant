/**
 * RadAssistant — Conversation store (client-side)
 *
 * "New chat" used to create a row in a list that nothing was attached to:
 * the sidebar tracked titles, the chat page tracked messages, and the two
 * never met. Clicking a previous conversation did nothing because there was
 * nothing to go back to. This module is the missing half — the actual
 * transcript, keyed by conversation.
 *
 * ⚠️  WHY localStorage, WHEN auth.ts DELIBERATELY USES sessionStorage.
 * That difference is intentional and worth stating, because it looks like an
 * inconsistency. A *token* in localStorage means the next person at a shared
 * reading-room workstation is still signed in as the last one, and signs
 * reports in their name. That is an authorisation failure. A *transcript* in
 * localStorage is not: it is inert text that grants no access, and it is
 * partitioned per account (see storageKey) so signing in as someone else
 * shows their history, not yours. The token still expires with the tab.
 *
 * The practical consequence: on a shared workstation the previous user's
 * chat titles remain on disk under their own key. For anonymised and
 * synthetic material that is the intended trade — history that survives a
 * browser restart. `clearAllChats` exists for deployments that would rather
 * not keep it, and can be wired into sign-out.
 *
 * ⚠️  NOT THE SAME THING AS THE REPORTS TABLE.
 * Approved reports are persisted server-side, owned, and audited. This is
 * scratch conversation state. Nothing here is a clinical record, and a
 * cleared browser losing it is acceptable in a way that losing a signed
 * report would not be.
 */

import type { SourceReference, ChatMode, MedicalImage } from "./api";

/** One turn in a conversation, in the shape the chat page renders. */
export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  images?: MedicalImage[];
  sources?: SourceReference[];
  model?: string;
  isStreaming?: boolean;
  isError?: boolean;
  mode?: ChatMode;
  findingsInput?: string;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: StoredMessage[];
}

const VERSION = "v1";
const MAX_CONVERSATIONS = 60;

/** Placeholder title for a conversation with nothing in it yet. */
export const UNTITLED = "New chat";

/**
 * Storage is partitioned by account.
 *
 * Without this, signing out and signing in as a colleague on the same
 * workstation shows you their reading list — or rather, shows them yours.
 * The email is not a secret and is only used as a namespace.
 */
function storageKey(email: string | null): string {
  return `radassist.chats.${VERSION}.${(email ?? "anon").toLowerCase()}`;
}

function isConversation(value: unknown): value is Conversation {
  if (!value || typeof value !== "object") return false;
  const c = value as Partial<Conversation>;
  return (
    typeof c.id === "string" &&
    typeof c.title === "string" &&
    Array.isArray(c.messages)
  );
}

/**
 * Read the stored conversations for an account.
 *
 * Returns [] rather than throwing on anything unexpected. A corrupt or
 * half-written entry — a quota error mid-write, a schema from an older build
 * — must degrade to "no history", never to a blank screen where the chat
 * used to be.
 */
export function loadConversations(email: string | null): Conversation[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(storageKey(email));
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(isConversation)
      .map((c) => ({
        ...c,
        // A message left mid-stream when the tab was closed would otherwise
        // reload with a blinking caret and no way to finish it.
        messages: c.messages.map((m) => ({ ...m, isStreaming: false })),
      }))
      .sort((a, b) => b.updatedAt - a.updatedAt);
  } catch {
    return [];
  }
}

/**
 * Persist conversations, newest first, capped.
 *
 * ⚠️  THE CAP IS NOT COSMETIC. localStorage is a hard ~5MB per origin and a
 * transcript carrying source excerpts is not small. Past the quota every
 * write throws, and the failure mode is that history silently stops updating
 * while the app looks fine. Trimming to the most recent conversations keeps
 * the store well inside the budget; if a write still fails we drop the
 * oldest half and try once more before giving up.
 */
export function saveConversations(
  email: string | null,
  conversations: Conversation[]
): void {
  if (typeof window === "undefined") return;
  const key = storageKey(email);
  const trimmed = [...conversations]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, MAX_CONVERSATIONS);

  try {
    window.localStorage.setItem(key, JSON.stringify(trimmed));
  } catch {
    try {
      window.localStorage.setItem(
        key,
        JSON.stringify(trimmed.slice(0, Math.ceil(trimmed.length / 2)))
      );
    } catch {
      /* Out of room and the halved set still doesn't fit. History stops
         persisting; the in-memory session continues unaffected. */
    }
  }
}

export function clearAllChats(email: string | null): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(storageKey(email));
  } catch {
    /* nothing to do */
  }
}

export function newConversation(): Conversation {
  const now = Date.now();
  return {
    id: `c_${now}_${Math.random().toString(36).slice(2, 8)}`,
    title: UNTITLED,
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

/**
 * Derive a sidebar title from the first thing the user actually said.
 *
 * Cuts on a word boundary so the list doesn't fill with "What are the radio…"
 * style truncations mid-syllable. Falls back to the placeholder for an
 * image-only message, which has no text to name it by.
 */
export function titleFromMessage(text: string): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return UNTITLED;
  if (clean.length <= 42) return clean;
  const cut = clean.slice(0, 42);
  const lastSpace = cut.lastIndexOf(" ");
  return (lastSpace > 20 ? cut.slice(0, lastSpace) : cut) + "…";
}
