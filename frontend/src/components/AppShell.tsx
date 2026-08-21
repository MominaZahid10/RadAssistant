"use client";

/**
 * RadAssistant — application shell and conversation state.
 *
 * TWO JOBS, ONE COMPONENT, ON PURPOSE:
 *
 * 1. LAYOUT. The sidebar and the main column are flex siblings. The previous
 *    layout pinned the sidebar with `position: fixed` and pushed the content
 *    over with a hard-coded `ml-[260px]` — the width was written down twice,
 *    and collapsing it would have meant animating a margin in lockstep with
 *    a width. As siblings, one transition moves both.
 *
 * 2. STATE. The sidebar lists conversations; the chat page owns the messages
 *    inside one. They must be the same data or "go back to a previous chat"
 *    cannot work. Rather than have the sidebar own titles and the page own
 *    transcripts and try to keep the two in step, both read from this context.
 *
 * WHY THE SHELL IS SKIPPED ON /login: the sidebar previously rendered on the
 * sign-in page too — a chat list, a New Chat button and a sign-out control,
 * shown to somebody who is not signed in.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { usePathname } from "next/navigation";
import Sidebar from "./Sidebar";
import {
  loadConversations,
  saveConversations,
  newConversation,
  titleFromMessage,
  UNTITLED,
  type Conversation,
  type StoredMessage,
} from "@/lib/chats";
import { AUTH_EXPIRED_EVENT, getStoredEmail } from "@/lib/auth";

interface AppContextValue {
  /** Sidebar collapsed, like Claude/ChatGPT. Persisted. */
  collapsed: boolean;
  setCollapsed: (v: boolean) => void;

  conversations: Conversation[];
  activeId: string | null;
  activeConversation: Conversation | null;
  messages: StoredMessage[];

  startNewChat: () => void;
  selectChat: (id: string) => void;
  deleteChat: (id: string) => void;
  /**
   * Guarantee there is a conversation to write into, and return its id.
   * Call this once before the first setMessages of a send.
   */
  ensureConversation: () => string;
  /**
   * Replace a conversation's messages (accepts an updater).
   *
   * `conversationId` pins the write to a specific conversation. Pass the id
   * returned by ensureConversation for anything that spans time — a stream,
   * an upload — so it keeps writing where it started even if the user has
   * since switched chats. Omit it for immediate edits to whatever is on
   * screen.
   */
  setMessages: (
    update: StoredMessage[] | ((prev: StoredMessage[]) => StoredMessage[]),
    conversationId?: string
  ) => void;
  /** True once localStorage has been read — guards the first paint. */
  hydrated: boolean;
}

const AppContext = createContext<AppContextValue | null>(null);

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside <AppShell>");
  return ctx;
}

const COLLAPSE_KEY = "radassist.sidebar.collapsed";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isAuthRoute = pathname?.startsWith("/login") ?? false;

  const [collapsed, setCollapsedState] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveIdState] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [email, setEmail] = useState<string | null>(null);

  /**
   * ⚠️  A REF ALONGSIDE THE STATE, DELIBERATELY.
   *
   * Streaming calls setMessages once per token. Those calls need to know
   * which conversation to write into *now*, not at the last render — and
   * several tokens can arrive inside one React batch, before `activeId` has
   * been re-read. The ref is the synchronous answer; the state drives
   * rendering. Every path that changes the active chat writes both.
   */
  const activeIdRef = useRef<string | null>(null);

  const setActiveId = useCallback((id: string | null) => {
    activeIdRef.current = id;
    setActiveIdState(id);
  }, []);

  // ── Hydration ────────────────────────────────────────────────
  // Reading localStorage during render would make the server-rendered markup
  // disagree with the first client render, and React would throw the tree
  // away. Everything storage-backed is read here instead, after mount.
  //
  //
  // The disable below is deliberate. Seeding state from a browser store on
  // mount is the one case this rule cannot express: useSyncExternalStore
  // (used for the session email in Sidebar) is the right answer for values
  // React only READS, but conversations are written and edited here, so they
  // have to be real state. The cost is one extra render on mount, once.
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (isAuthRoute) return;

    const who = getStoredEmail();
    setEmail(who);

    const stored = loadConversations(who);
    setConversations(stored);
    setActiveId(stored.length ? stored[0].id : null);

    try {
      setCollapsedState(window.localStorage.getItem(COLLAPSE_KEY) === "1");
    } catch {
      /* storage blocked — default to expanded */
    }
    setHydrated(true);
  }, [isAuthRoute, setActiveId]);
  /* eslint-enable react-hooks/set-state-in-effect */

  // Signing out must not leave the previous account's titles on screen while
  // the redirect lands.
  useEffect(() => {
    const onExpired = () => {
      setConversations([]);
      setActiveId(null);
      setEmail(null);
    };
    window.addEventListener(AUTH_EXPIRED_EVENT, onExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, onExpired);
  }, [setActiveId]);

  // ── Persistence ──────────────────────────────────────────────
  // ⚠️  DEBOUNCED, BECAUSE STREAMING WRITES ONCE PER TOKEN.
  // Serialising the whole conversation on every token is thousands of
  // JSON.stringify calls on the main thread while text is animating.
  // Coalescing to one write every 600ms keeps the transcript safe without
  // the stutter.
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!hydrated) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      saveConversations(email, conversations);
    }, 600);
    return () => {
      if (saveTimer.current) clearTimeout(saveTimer.current);
    };
  }, [conversations, email, hydrated]);

  // A pending debounce would otherwise be dropped when the tab closes
  // mid-answer, losing the last few seconds of the transcript.
  useEffect(() => {
    if (!hydrated) return;
    const flush = () => saveConversations(email, conversations);
    window.addEventListener("beforeunload", flush);
    return () => window.removeEventListener("beforeunload", flush);
  }, [conversations, email, hydrated]);

  const setCollapsed = useCallback((v: boolean) => {
    setCollapsedState(v);
    try {
      window.localStorage.setItem(COLLAPSE_KEY, v ? "1" : "0");
    } catch {
      /* preference simply won't persist */
    }
  }, []);

  // ── Conversation actions ─────────────────────────────────────

  const startNewChat = useCallback(() => {
    // Clicking New chat twice should not stack up identical empty rows — if
    // a blank conversation is already open, that IS the new chat.
    const blank = conversations.find((c) => c.messages.length === 0);
    if (blank) {
      setActiveId(blank.id);
      return;
    }
    const fresh = newConversation();
    setConversations((prev) => [fresh, ...prev]);
    setActiveId(fresh.id);
  }, [conversations, setActiveId]);

  const selectChat = useCallback(
    (id: string) => setActiveId(id),
    [setActiveId]
  );

  const deleteChat = useCallback(
    (id: string) => {
      const next = conversations.filter((c) => c.id !== id);
      setConversations(next);
      if (activeIdRef.current === id) {
        setActiveId(next.length ? next[0].id : null);
      }
    },
    [conversations, setActiveId]
  );

  /**
   * Make sure there is somewhere to put the next message.
   *
   * A user landing on a fresh install and typing straight away has no
   * conversation yet. Creating it here — once, at send time — means
   * setMessages can stay a pure updater and never has to spawn state as a
   * side effect of a token arriving.
   */
  const ensureConversation = useCallback((): string => {
    const current = activeIdRef.current;
    if (current && conversations.some((c) => c.id === current)) return current;

    const fresh = newConversation();
    setConversations((prev) => [fresh, ...prev]);
    setActiveId(fresh.id);
    return fresh.id;
  }, [conversations, setActiveId]);

  const setMessages = useCallback<AppContextValue["setMessages"]>(
    (update, conversationId) => {
      // ⚠️  THE EXPLICIT ID IS WHAT MAKES SWITCHING CHATS MID-ANSWER SAFE.
      // Falling back to "whatever is active" was wrong for streaming: a user
      // who clicks another conversation while tokens are still arriving moves
      // the active id, and every subsequent token then looked for its message
      // in a conversation that does not contain it. The tokens were dropped
      // and the half-written answer kept its caret forever, because the
      // `done` event missed too. Callers that span time pass the id they
      // started with.
      const id = conversationId ?? activeIdRef.current;
      if (!id) return;

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== id) return c;
          const next =
            typeof update === "function" ? update(c.messages) : update;

          // Name the conversation from the first thing the user said, and
          // only then — retitling on every send would rewrite the sidebar
          // label halfway through a conversation.
          const firstUser = next.find((m) => m.role === "user");
          const title =
            c.title === UNTITLED && firstUser
              ? titleFromMessage(firstUser.content)
              : c.title;

          return { ...c, messages: next, title, updatedAt: Date.now() };
        })
      );
    },
    []
  );

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? null,
    [conversations, activeId]
  );

  const value = useMemo<AppContextValue>(
    () => ({
      collapsed,
      setCollapsed,
      conversations,
      activeId,
      activeConversation,
      messages: activeConversation?.messages ?? [],
      startNewChat,
      selectChat,
      deleteChat,
      ensureConversation,
      setMessages,
      hydrated,
    }),
    [
      collapsed,
      setCollapsed,
      conversations,
      activeId,
      activeConversation,
      startNewChat,
      selectChat,
      deleteChat,
      ensureConversation,
      setMessages,
      hydrated,
    ]
  );

  // The sign-in page is full-bleed: no sidebar, no chat furniture.
  if (isAuthRoute) {
    return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
  }

  return (
    <AppContext.Provider value={value}>
      <div className="app-shell">
        <Sidebar />
        <main className="app-main">{children}</main>
      </div>
    </AppContext.Provider>
  );
}
