/**
 * RadAssist AI — Backend API Client
 *
 * WHY A SEPARATE API CLIENT?
 * Instead of writing `fetch("http://localhost:8000/api/v1/...")` 
 * everywhere in the frontend, we create ONE utility that:
 *
 * 1. Knows the backend URL (from environment variable)
 * 2. Adds common headers (Content-Type, Auth tokens later)
 * 3. Handles errors consistently
 * 4. Makes it easy to swap the URL for production
 *
 * USAGE:
 *   import { api } from "@/lib/api";
 *   const health = await api.getHealth();
 *   
 *   // Streaming chat:
 *   const stream = api.streamChat("What is pneumothorax?");
 *   for await (const event of stream) { ... }
 */

// Backend URL — reads from .env.local, falls back to Docker default
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Generic fetch wrapper with error handling.
 * All API calls go through this function.
 */
async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  // If the response isn't OK (2xx), throw a descriptive error
  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(
      `API Error ${response.status}: ${response.statusText} — ${errorBody}`
    );
  }

  return response.json();
}

// ── Health Check Types ──────────────────────────────────────
export interface HealthResponse {
  status: string;
  version: string;
  components: {
    api: string;
    database: string;
    qdrant: string;
  };
}

// ── Chat Types (Phase 3) ────────────────────────────────────

export type Audience = "radiologist" | "resident";

export interface ChatRequest {
  query: string;
  stream: boolean;
  audience: Audience;
  include_sources: boolean;
}

export interface SourceReference {
  chunk_id: number;
  text: string;
  score: number;
  document_title: string | null;
  source_type: string | null;
  chunk_index: number | null;
  document_id: string | null;
}

export interface ChatResponse {
  answer: string;
  sources: SourceReference[] | null;
  query: string;
  model: string;
}

export interface ProviderInfo {
  configured: boolean;
  default_model: string;
}

export interface ModelInfoResponse {
  active_provider: string;
  active_model: string;
  providers: Record<string, ProviderInfo>;
}

// ── SSE Event Types ─────────────────────────────────────────

export type SSEEvent =
  | { type: "sources"; sources: SourceReference[] }
  | { type: "token"; token: string }
  | { type: "done"; model: string }
  | { type: "error"; error: string };

// ── API Methods ─────────────────────────────────────────────
// Each method is a clean function call. No raw fetch() needed
// in components.
export const api = {
  /** Check if the backend and all services are healthy */
  getHealth: () => fetchAPI<HealthResponse>("/api/v1/health"),

  /** Get basic API info */
  getRoot: () => fetchAPI<{ app: string; version: string }>("/"),

  /** Get LLM provider info */
  getModels: () => fetchAPI<ModelInfoResponse>("/api/v1/chat/models"),

  /**
   * Non-streaming chat — sends a question, waits for the full response.
   * Good for programmatic use or testing.
   */
  sendChat: (
    query: string,
    options?: { audience?: Audience; includeSources?: boolean }
  ) =>
    fetchAPI<ChatResponse>("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({
        query,
        stream: false,
        audience: options?.audience ?? "radiologist",
        include_sources: options?.includeSources ?? true,
      }),
    }),

  /**
   * Streaming chat — sends a question, returns an async generator
   * that yields SSE events (sources, tokens, done/error).
   *
   * USAGE:
   *   for await (const event of api.streamChat("pneumothorax?")) {
   *     if (event.type === "token") process.stdout.write(event.token);
   *     if (event.type === "sources") showEvidence(event.sources);
   *     if (event.type === "done") console.log("Model:", event.model);
   *   }
   */
  streamChat: async function* (
    query: string,
    options?: { audience?: Audience; includeSources?: boolean }
  ): AsyncGenerator<SSEEvent> {
    const url = `${API_BASE_URL}/api/v1/chat`;

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        stream: true,
        audience: options?.audience ?? "radiologist",
        include_sources: options?.includeSources ?? true,
      }),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `API Error ${response.status}: ${response.statusText} — ${errorBody}`
      );
    }

    // Parse the SSE stream.
    // We read the response body as text chunks and parse SSE events manually.
    // This is more reliable across browsers than using EventSource (which
    // only supports GET requests).
    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by double newlines
        const events = buffer.split("\n\n");
        // The last element might be incomplete — keep it in the buffer
        buffer = events.pop() || "";

        for (const eventStr of events) {
          const parsed = _parseSSEEvent(eventStr.trim());
          if (parsed) yield parsed;
        }
      }

      // Process any remaining buffer
      if (buffer.trim()) {
        const parsed = _parseSSEEvent(buffer.trim());
        if (parsed) yield parsed;
      }
    } finally {
      reader.releaseLock();
    }
  },
};

/**
 * Parse a single SSE event string into a typed event object.
 *
 * Input format:
 *   "event: token\ndata: {"token": "The"}"
 */
function _parseSSEEvent(raw: string): SSEEvent | null {
  if (!raw) return null;

  let eventType = "";
  let dataStr = "";

  for (const line of raw.split("\n")) {
    if (line.startsWith("event: ")) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      dataStr = line.slice(6);
    }
  }

  if (!eventType || !dataStr) return null;

  try {
    const data = JSON.parse(dataStr);

    switch (eventType) {
      case "sources":
        return { type: "sources", sources: data.sources ?? [] };
      case "token":
        return { type: "token", token: data.token ?? "" };
      case "done":
        return { type: "done", model: data.model ?? "unknown" };
      case "error":
        return { type: "error", error: data.error ?? "Unknown error" };
      default:
        return null;
    }
  } catch {
    return null;
  }
}
