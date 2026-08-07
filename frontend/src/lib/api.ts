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

// ── API Methods ─────────────────────────────────────────────
// Each method is a clean function call. No raw fetch() needed
// in components.
export const api = {
  /** Check if the backend and all services are healthy */
  getHealth: () => fetchAPI<HealthResponse>("/api/v1/health"),

  /** Get basic API info */
  getRoot: () => fetchAPI<{ app: string; version: string }>("/"),

  // Future phases will add:
  // generateReport: (data) => fetchAPI("/api/v1/reports/generate", { method: "POST", body: JSON.stringify(data) }),
  // searchKnowledge: (query) => fetchAPI(`/api/v1/knowledge/search?q=${query}`),
  // uploadDocument: (file) => fetchAPI("/api/v1/documents/upload", { method: "POST", body: formData }),
};
