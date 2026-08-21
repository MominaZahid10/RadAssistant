/**
 * Playwright configuration — end-to-end tests against the REAL stack.
 *
 * ⚠️  THESE TESTS DO NOT MOCK THE API.
 * They drive the app the way a radiologist would, against the running
 * docker-compose stack. That is deliberate: the class of bug this suite
 * exists to catch — a conversation that stops updating, an answer written
 * into the wrong chat, a session that leaks between accounts — only appears
 * when real latency is involved. A mocked stream returns in one tick and
 * hides exactly those races.
 *
 * The cost is real. Two specs ask the LLM actual questions and so spend
 * tokens; everything else exercises the UI against the API without
 * generation. Keep it that way — a suite that costs money per run stops
 * being run.
 *
 * PREREQUISITES
 *   docker-compose up -d          # backend, frontend, postgres, qdrant
 *   ALLOW_REGISTRATION=true       # in backend/.env — the suite creates its
 *                                 # own throwaway accounts (see e2e/auth.ts)
 *
 * RUN
 *   npm run test:e2e              # headless
 *   npm run test:e2e:headed       # watch it happen
 *   npm run test:e2e -- --ui      # pick and step through individual tests
 */

import { defineConfig, devices } from "@playwright/test";

const BASE_URL = process.env.E2E_BASE_URL ?? "http://localhost:3000";
const API_URL = process.env.E2E_API_URL ?? "http://localhost:8000";

export default defineConfig({
  testDir: "./e2e",

  // Waits for the stack and compiles both routes before any test runs, so
  // the first assertion isn't racing a cold `next dev` build. See the file
  // for the failure this prevents.
  globalSetup: "./e2e/support/global-setup.ts",

  // ⚠️  NOT PARALLEL BY DEFAULT.
  // Every worker talks to one Postgres and one Qdrant. Registration and
  // report approval both write, and the LLM provider rate-limits per key —
  // parallel workers turn a real failure into an unreproducible one. Set
  // E2E_WORKERS if you have the headroom and want the speed.
  workers: process.env.E2E_WORKERS ? Number(process.env.E2E_WORKERS) : 1,
  fullyParallel: false,

  // A real answer involves retrieval, reranking and generation. The default
  // 30s expect timeout is not enough on a cold cache; 60s is.
  timeout: 120_000,
  expect: { timeout: 15_000 },

  // Fail the run if someone leaves a .only in — that silently skips the rest
  // of the suite and the run still goes green.
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,

  reporter: process.env.CI
    ? [["list"], ["html", { open: "never" }]]
    : [["list"], ["html", { open: "never" }]],

  use: {
    baseURL: BASE_URL,
    // Evidence for a failure you did not watch happen.
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    actionTimeout: 15_000,
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } },
    },
  ],

  // Surfaced to the specs via process.env so a single place decides where the
  // API lives.
  metadata: { apiUrl: API_URL },
});
