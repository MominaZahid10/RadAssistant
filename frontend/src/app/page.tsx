/**
 * RadAssist AI — Chat Page (Main Interface)
 * 
 * Clean chat interface like ChatGPT/Claude:
 * - Messages area in the center
 * - Input box pinned to bottom
 * - Welcome screen when no messages
 */
"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 200) + "px";
    }
  }, [input]);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    // Add user message
    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: trimmed,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    // Simulate AI response (will be replaced with real RAG in Phase 3)
    setTimeout(() => {
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "I'm RadAssist AI, your radiology reporting assistant. The RAG pipeline isn't connected yet (that's Phase 3!), but the infrastructure is ready. Once connected, I'll be able to:\n\n• Generate structured radiology reports from your findings\n• Search the knowledge base for relevant guidelines & templates\n• Find similar past cases\n• Suggest differential diagnoses with evidence\n\nEvery suggestion will come with traceable sources — no black-box answers.",
      };
      setMessages((prev) => [...prev, aiMsg]);
      setIsLoading(false);
    }, 1500);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          /* Welcome Screen */
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-md">
              <div className="text-5xl mb-4">🩻</div>
              <h1 className="text-2xl font-semibold text-gradient mb-2">
                RadAssist AI
              </h1>
              <p className="text-foreground-secondary text-sm leading-relaxed">
                Your radiology reporting assistant. Ask about findings,
                generate reports, or search the knowledge base.
              </p>
            </div>
          </div>
        ) : (
          /* Message List */
          <div className="max-w-3xl mx-auto py-6 px-4 space-y-6">
            {messages.map((msg) => (
              <div key={msg.id} className="flex gap-3">
                {/* Avatar */}
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs mt-0.5 ${
                    msg.role === "user"
                      ? "bg-accent/20 text-accent"
                      : "bg-surface text-foreground-secondary"
                  }`}
                >
                  {msg.role === "user" ? "Y" : "🩻"}
                </div>

                {/* Message Content */}
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-foreground-muted mb-1">
                    {msg.role === "user" ? "You" : "RadAssist AI"}
                  </p>
                  <div className="text-sm text-foreground leading-relaxed whitespace-pre-wrap">
                    {msg.content}
                  </div>
                </div>
              </div>
            ))}

            {/* Typing Indicator */}
            {isLoading && (
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-surface flex items-center justify-center flex-shrink-0 text-xs">
                  🩻
                </div>
                <div className="pt-2">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-foreground-muted rounded-full dot-1" />
                    <span className="w-2 h-2 bg-foreground-muted rounded-full dot-2" />
                    <span className="w-2 h-2 bg-foreground-muted rounded-full dot-3" />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area — pinned to bottom */}
      <div className="border-t border-border bg-background-secondary p-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-2 bg-surface rounded-xl border border-border focus-within:border-accent/50 transition-colors px-4 py-3">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe your findings or ask a question..."
              rows={1}
              className="flex-1 bg-transparent text-sm text-foreground placeholder:text-foreground-muted resize-none outline-none max-h-[200px]"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className={`p-2 rounded-lg transition-colors flex-shrink-0 ${
                input.trim() && !isLoading
                  ? "bg-accent text-white hover:bg-accent-hover"
                  : "text-foreground-muted cursor-not-allowed"
              }`}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="none"
                className="rotate-90"
              >
                <path
                  d="M2 8L14 2L8 14L7 9L2 8Z"
                  fill="currentColor"
                />
              </svg>
            </button>
          </div>
          <p className="text-[10px] text-foreground-muted text-center mt-2">
            RadAssist AI is a decision-support tool. Always verify AI suggestions before clinical use.
          </p>
        </div>
      </div>
    </div>
  );
}
