/**
 * RadAssist AI — Sidebar (ChatGPT-style)
 * 
 * Simple sidebar with:
 * - New Chat button at top
 * - Chat history list
 * - App branding at bottom
 */
"use client";

import { useState } from "react";

interface Chat {
  id: string;
  title: string;
  active: boolean;
}

export default function Sidebar() {
  const [chats, setChats] = useState<Chat[]>([
    { id: "1", title: "Welcome to RadAssist AI", active: true },
  ]);

  const handleNewChat = () => {
    const newChat: Chat = {
      id: Date.now().toString(),
      title: "New conversation",
      active: true,
    };
    setChats((prev) =>
      [newChat, ...prev.map((c) => ({ ...c, active: false }))]
    );
  };

  const handleSelectChat = (id: string) => {
    setChats((prev) =>
      prev.map((c) => ({ ...c, active: c.id === id }))
    );
  };

  return (
    <aside className="fixed left-0 top-0 h-screen w-[260px] bg-sidebar-bg border-r border-border flex flex-col z-50">
      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border border-border text-sm text-foreground-secondary hover:text-foreground hover:bg-surface-hover transition-colors"
        >
          <span className="text-lg">+</span>
          <span>New Chat</span>
        </button>
      </div>

      {/* Chat History */}
      <nav className="flex-1 px-3 space-y-0.5 overflow-y-auto">
        {chats.map((chat) => (
          <button
            key={chat.id}
            onClick={() => handleSelectChat(chat.id)}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm truncate transition-colors ${
              chat.active
                ? "bg-surface text-foreground"
                : "text-foreground-secondary hover:bg-surface-hover hover:text-foreground"
            }`}
          >
            {chat.title}
          </button>
        ))}
      </nav>

      {/* Bottom Branding */}
      <div className="p-3 border-t border-border">
        <div className="flex items-center gap-2 px-2">
          <span className="text-lg">🩻</span>
          <div>
            <p className="text-sm font-semibold text-gradient">RadAssist AI</p>
            <p className="text-[10px] text-foreground-muted">Radiology Assistant</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
