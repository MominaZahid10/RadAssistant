/**
 * Warm the dev server, and fail fast with a readable message if the stack
 * is not up.
 *
 * ⚠️  WHY A WARMUP EXISTS AT ALL.
 * docker-compose runs the frontend with `next dev`, which compiles a route
 * the first time it is requested. The chat page is a large client component
 * plus a full Tailwind pass, and inside Docker on Windows — with polling
 * file watchers — that first compile can take well over the 15s an assertion
 * is willing to wait. The symptom is baffling: the very first test fails
 * because a redirect "did not happen", when in fact the page had not
 * finished hydrating, so the effect that performs the redirect had not run
 * yet. Every later test passes. Requesting both routes once here moves that
 * cost outside the tests, where it belongs.
 */

import type { FullConfig } from "@playwright/test";

const BASE_URL = process.env.E2E_BASE_URL ?? "http://localhost:3000";
const API_URL = process.env.E2E_API_URL ?? "http://localhost:8000";

async function waitFor(
  url: string,
  label: string,
  timeoutMs = 180_000
): Promise<void> {
  const started = Date.now();
  let lastError = "";

  while (Date.now() - started < timeoutMs) {
    try {
      const res = await fetch(url, { redirect: "manual" });
      // Any HTTP answer means something is listening and compiled. A 3xx from
      // the app is a perfectly good sign of life.
      if (res.status < 500) return;
      lastError = `HTTP ${res.status}`;
    } catch (e) {
      lastError = e instanceof Error ? e.message : String(e);
    }
    await new Promise((r) => setTimeout(r, 2_000));
  }

  throw new Error(
    `${label} did not respond at ${url} within ${timeoutMs / 1000}s ` +
      `(last error: ${lastError}).\n\n` +
      `The E2E suite runs against the real stack. Start it with:\n` +
      `    docker-compose up -d\n` +
      `and check it is healthy with:\n` +
      `    docker-compose ps\n` +
      `    docker-compose logs --tail=50 backend`
  );
}

export default async function globalSetup(_config: FullConfig) {
  await waitFor(`${API_URL}/api/v1/health`, "The backend");
  // Compile both routes before the first assertion depends on them.
  await waitFor(`${BASE_URL}/login`, "The frontend");
  await waitFor(`${BASE_URL}/`, "The frontend");
}
