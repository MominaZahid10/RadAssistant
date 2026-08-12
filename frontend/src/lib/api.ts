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

/**
 * What the model is being asked to DO, as opposed to who for.
 *
 *   qa     - answer a question from the knowledge base
 *   report - treat the input as dictated FINDINGS and draft a structured
 *            report (Findings / Impression) in clinical register
 *
 * The backend validates this as an enum, so a typo returns 422 rather than
 * quietly producing a chat answer where a report was asked for.
 */
export type ChatMode = "qa" | "report";

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
    options?: {
      audience?: Audience;
      mode?: ChatMode;
      includeSources?: boolean;
      /** Text of an uploaded document. Sent SEPARATELY, never appended
       *  to the question — see attached_text on the backend schema. */
      attachedText?: string;
      attachedWarnings?: string[];
    }
  ): AsyncGenerator<SSEEvent> {
    const url = `${API_BASE_URL}/api/v1/chat`;

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        stream: true,
        mode: options?.mode ?? "qa",
        audience: options?.audience ?? "radiologist",
        include_sources: options?.includeSources ?? true,
        attached_text: options?.attachedText ?? null,
        attached_warnings: options?.attachedWarnings ?? null,
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

// ══════════════════════════════════════════════════════════════
// Images (Phase 4)
// ══════════════════════════════════════════════════════════════

export type ImageSourceType =
  | "dicom_upload"
  | "report_upload"
  | "pmc_figure"
  | "image_upload";

export interface MedicalImage {
  id: string;
  document_id: string | null;
  filename: string;
  mime_type: string;
  file_size: number | null;
  width: number | null;
  height: number | null;

  // Clinical metadata — null for figures and plain uploads.
  modality: string | null;
  body_part: string | null;
  view_position: string | null;
  study_date: string | null;

  source_type: ImageSourceType;
  source_url: string | null;
  caption: string | null;
  description: string | null;

  // True only after de-identification demonstrably ran. Never assumed —
  // the UI surfaces this rather than letting the user infer it.
  is_deidentified: boolean;
  dicom_metadata: Record<string, unknown> | null;
  ocr_text: string | null;

  status: "processing" | "completed" | "failed";
  error_message: string | null;

  created_at: string;
  updated_at: string;

  // Server-resolved URLs. The API deliberately never exposes filesystem
  // paths, so these are the only way to fetch the bytes.
  file_url: string | null;
  thumbnail_url: string | null;
}

export interface ImageListResponse {
  images: MedicalImage[];
  total: number;
  page: number;
  page_size: number;
}

export interface ImageUploadResponse {
  id: string;
  filename: string;
  status: string;
  message: string;
  detected_type: "dicom" | "image" | "report";
}

export interface ImageStats {
  total_images: number;
  completed: number;
  failed: number;
  processing: number;
  by_source_type: Record<string, number>;
  by_modality: Record<string, number>;
  deidentified_count: number;
  storage_bytes: number;
  storage_files: number;
}

/** Absolute URL for an image served by the API. */
export function imageUrl(path: string | null): string | null {
  return path ? `${API_BASE_URL}${path}` : null;
}

export const imageApi = {
  /**
   * Upload a DICOM study, a photographed report, or a plain image.
   *
   * Returns as soon as the file is accepted — parsing, de-identification and
   * thumbnailing happen in the background, so poll `get()` until status
   * leaves "processing".
   */
  upload: async (
    file: File,
    options?: {
      sourceType?: ImageSourceType;
      description?: string;
      documentId?: string;
    }
  ): Promise<ImageUploadResponse> => {
    const form = new FormData();
    form.append("file", file);
    form.append("source_type", options?.sourceType ?? "image_upload");
    if (options?.description) form.append("description", options.description);
    if (options?.documentId) form.append("document_id", options.documentId);

    // NOTE: no Content-Type header. The browser must set it itself so it can
    // include the multipart boundary — setting it manually breaks the upload.
    const res = await fetch(`${API_BASE_URL}/api/v1/images/upload`, {
      method: "POST",
      body: form,
    });

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Upload failed (${res.status}): ${body}`);
    }
    return res.json();
  },

  get: (id: string) => fetchAPI<MedicalImage>(`/api/v1/images/${id}`),

  list: (params?: {
    page?: number;
    pageSize?: number;
    sourceType?: string;
    modality?: string;
    documentId?: string;
  }) => {
    const q = new URLSearchParams();
    if (params?.page) q.set("page", String(params.page));
    if (params?.pageSize) q.set("page_size", String(params.pageSize));
    if (params?.sourceType) q.set("source_type", params.sourceType);
    if (params?.modality) q.set("modality", params.modality);
    if (params?.documentId) q.set("document_id", params.documentId);
    const qs = q.toString();
    return fetchAPI<ImageListResponse>(`/api/v1/images${qs ? `?${qs}` : ""}`);
  },

  stats: () => fetchAPI<ImageStats>("/api/v1/images/stats"),

  /**
   * Figures belonging to one article.
   *
   * Used by the evidence panel: when a cited paper has figures, they appear
   * beside the passage that cited it — a citation you can look at rather than
   * only read.
   *
   * Returns [] rather than throwing on failure. A missing figure is a
   * cosmetic loss; it must never break the answer that is already on screen.
   */
  forDocument: async (documentId: string): Promise<MedicalImage[]> => {
    try {
      return await fetchAPI<MedicalImage[]>(
        `/api/v1/knowledge/documents/${documentId}/images`
      );
    } catch {
      return [];
    }
  },

  remove: (id: string) =>
    fetchAPI<{ message: string }>(`/api/v1/images/${id}`, { method: "DELETE" }),

  /**
   * Poll until processing finishes.
   *
   * Uploads return immediately, so the UI needs to know when the thumbnail
   * exists. Gives up after `timeoutMs` rather than polling forever — a stuck
   * background task shouldn't leave a spinner running indefinitely.
   */
  waitForProcessing: async (
    id: string,
    { intervalMs = 1000, timeoutMs = 60000 } = {}
  ): Promise<MedicalImage> => {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const img = await imageApi.get(id);
      if (img.status !== "processing") return img;
      if (Date.now() > deadline) {
        throw new Error(`Still processing after ${timeoutMs / 1000}s: ${id}`);
      }
      await new Promise((r) => setTimeout(r, intervalMs));
    }
  },
};
