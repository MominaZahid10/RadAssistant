"use client";

/**
 * RadAssistant — Sidebar
 *
 * ⚠️  WHAT WAS ACTUALLY BROKEN HERE.
 * The old sidebar held its own `chats` array in local state. "New Chat"
 * pushed a row onto that array and nothing else happened — no transcript was
 * created, the chat page never heard about it, and clicking a previous row
 * only moved a highlight. The list was a picture of a feature.
 *
 * It now renders the same conversations the chat page writes to (see
 * AppShell). New chat creates a real conversation; selecting one loads its
 * transcript; deleting one removes it.
 */

import { useApp } from "./AppShell";
import Brand from "./Brand";
import { UNTITLED } from "@/lib/chats";
import {
  clearSession,
  getStoredEmail,
  serverEmailSnapshot,
  subscribeToSession,
} from "@/lib/auth";
import { useRouter } from "next/navigation";
import { useSyncExternalStore } from "react";

export default function Sidebar() {
  const {
    collapsed,
    setCollapsed,
    conversations,
    activeId,
    startNewChat,
    selectChat,
    deleteChat,
    hydrated,
  } = useApp();
  const router = useRouter();

  // sessionStorage cannot be read during render without breaking hydration
  // (the server has no such storage). This subscribes to it properly instead
  // of reading it in an effect and re-rendering — see subscribeToSession.
  const email = useSyncExternalStore(
    subscribeToSession,
    getStoredEmail,
    serverEmailSnapshot
  );

  const initial = (email?.trim()?.[0] ?? "?").toUpperCase();

  return (
    <aside className="sidebar" data-collapsed={collapsed}>
      {/* ── Brand + collapse ── */}
      <div className="sidebar-top">
        {collapsed ? (
          <button
            className="sidebar-toggle"
            style={{ margin: 0, display: "grid" }}
            onClick={() => setCollapsed(false)}
            title="Expand sidebar"
            aria-label="Expand sidebar"
          >
            <Brand size={24} className="text-accent" />
          </button>
        ) : (
          <>
            <Brand size={26} className="text-accent" />
            <span className="sidebar-brand">RadAssistant</span>
            <button
              className="sidebar-toggle"
              onClick={() => setCollapsed(true)}
              title="Collapse sidebar"
              aria-label="Collapse sidebar"
            >
              <svg
                width="17"
                height="17"
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
          </>
        )}
      </div>

      {/* ── New chat ── */}
      <div className="sidebar-new">
        <button
          onClick={startNewChat}
          title="New chat"
          aria-label="Start a new chat"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            style={{ flexShrink: 0 }}
          >
            <path d="M12 5v14M5 12h14" />
          </svg>
          <span>New chat</span>
        </button>
      </div>

      {/* ── History ── */}
      <div className="sidebar-label">Recent</div>
      <nav className="sidebar-list">
        {hydrated && conversations.length === 0 && (
          <p className="sidebar-empty">No conversations yet.</p>
        )}

        {conversations.map((chat) => (
          <div
            key={chat.id}
            className="chat-row"
            data-active={chat.id === activeId}
          >
            <button
              className="chat-row-open"
              onClick={() => selectChat(chat.id)}
              title={chat.title}
            >
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
              >
                <path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 8.9 8.9 0 0 1-4-.9L3 21l1.9-4.9A8.4 8.4 0 0 1 12 3a8.4 8.4 0 0 1 9 8.5z" />
              </svg>
              <span className="chat-row-title">{chat.title || UNTITLED}</span>
            </button>

            <button
              className="chat-row-delete"
              onClick={() => deleteChat(chat.id)}
              title="Delete conversation"
              aria-label={`Delete ${chat.title}`}
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
              >
                <path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" />
              </svg>
            </button>
          </div>
        ))}
      </nav>

      {/* ── Account ──
          Without a way out there is no route back to the sign-in screen
          short of clearing storage by hand — and on a shared workstation
          "sign out" is what stops the next person's approvals being
          attributed to you. */}
      <div className="sidebar-foot">
        <div className="sidebar-avatar" title={email ?? undefined}>
          {initial}
        </div>
        <span className="sidebar-email">{email ?? "Signed out"}</span>
        <button
          className="sidebar-signout"
          title="Sign out"
          aria-label="Sign out"
          onClick={() => {
            // notify=false: this navigation is deliberate, so the
            // "session expired" listener must not also fire.
            clearSession(false);
            router.replace("/login");
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
          >
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" />
          </svg>
        </button>
      </div>
    </aside>
  );
}
