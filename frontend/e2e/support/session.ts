/**
 * Shared helpers: accounts, session seeding, storage inspection.
 *
 * ⚠️  ACCOUNTS ARE STABLE AND REUSED, NOT CREATED PER TEST.
 *
 * The first version of this file registered a throwaway account per test, to
 * keep tests independent. Against the real backend that is wrong twice over:
 *
 *   1. `REGISTER = Limit(5, 3600)` in app/core/limits.py — five registrations
 *      per hour, per IP. A 25-test suite asking for 20+ accounts exhausts
 *      that in the first few tests and every later one dies on a 429. The
 *      limiter is not an obstacle to work around; it is there because a
 *      signup form cannot hide whether an address is already registered, and
 *      rate limiting is what stops that being enumerable. A test suite does
 *      not get to disable it.
 *
 *   2. Registration is the wrong tool anyway. What each test needs is a
 *      *token*, and logging in gets one — `LOGIN = Limit(10, 60)` is far
 *      more generous, and after the first ever run no registration happens
 *      at all.
 *
 * So: two fixed accounts, created on first use and logged into thereafter.
 *
 * Sharing accounts across tests is safe HERE for a specific reason worth
 * stating, because it would not be in most suites: conversations live in
 * localStorage, and Playwright gives every test a fresh browser context.
 * Two tests using the same account still see empty, independent storage. The
 * account only supplies a bearer token. If conversations ever move to the
 * server, this assumption breaks and the fixtures have to change with it.
 */

/**
 * ⚠️  example.com, NOT example.test.
 * The API validates with pydantic's EmailStr, which uses email-validator,
 * which rejects RFC 2606 special-use TLDs — .test, .invalid, .localhost,
 * .local, .onion, .arpa. An address at example.test comes back 422, and the
 * frontend maps 422 to "Password must be at least 12 characters", so the
 * error you see names the wrong field entirely. example.com is a normal
 * registrable domain and validates cleanly.
 */
const TEST_DOMAIN = "example.com";

import { type Page, expect } from "@playwright/test";

export const API_URL = process.env.E2E_API_URL ?? "http://localhost:8000";

/** Comfortably over the server's 12-character minimum. */
export const TEST_PASSWORD = "e2e-correct-horse-battery";

export interface TestAccount {
  email: string;
  password: string;
  token: string;
}

async function postJson(path: string, body: unknown) {
  return fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function explain(status: number, action: string): string {
  if (status === 429) {
    return (
      `${action} was rate limited (429). The backend allows 5 registrations ` +
      `per hour and 10 logins per minute per IP (app/core/limits.py). If you ` +
      `have run this suite several times in the last hour, wait or restart ` +
      `the backend container to reset the in-memory counters.`
    );
  }
  if (status === 403) {
    return (
      `${action} was refused (403): registration is disabled on this backend. ` +
      `Set ALLOW_REGISTRATION=true in backend/.env and restart the stack, or ` +
      `pre-create the two e2e accounts with scripts/create_user.py.`
    );
  }
  if (status === 422) {
    return (
      `${action} failed validation (422). Check the email domain and that the ` +
      `password is at least 12 characters.`
    );
  }
  return `${action} failed (${status}). Is the backend up at ${API_URL}?`;
}

/**
 * Get a token for a stable account, creating it the first time it is needed.
 *
 * Login first, register only on failure — so a repeat run of the suite costs
 * zero registrations and cannot exhaust the hourly limit.
 */
export async function ensureAccount(label: string): Promise<TestAccount> {
  const email = `radassist-e2e-${label}@${TEST_DOMAIN}`;

  const login = await postJson("/api/v1/auth/login", {
    email,
    password: TEST_PASSWORD,
  });
  if (login.ok) {
    const body = (await login.json()) as { access_token: string };
    return { email, password: TEST_PASSWORD, token: body.access_token };
  }

  // 401 means the account does not exist yet (or the password changed).
  // Anything else is a problem worth reporting rather than papering over
  // with a registration attempt.
  if (login.status !== 401) {
    throw new Error(explain(login.status, `Signing in as ${email}`));
  }

  const register = await postJson("/api/v1/auth/register", {
    email,
    password: TEST_PASSWORD,
    full_name: "RadAssistant E2E",
  });

  if (!register.ok) {
    throw new Error(explain(register.status, `Creating ${email}`));
  }

  const body = (await register.json()) as { access_token: string };
  return { email, password: TEST_PASSWORD, token: body.access_token };
}

/**
 * Register a genuinely new account through the API.
 *
 * Only for tests that must observe a first-ever sign-up. Costs one of the
 * five hourly registrations, so use it sparingly.
 */
export async function registerFreshAccount(): Promise<TestAccount> {
  const email = `radassist-e2e-fresh-${Date.now()}@${TEST_DOMAIN}`;
  const res = await postJson("/api/v1/auth/register", {
    email,
    password: TEST_PASSWORD,
    full_name: "RadAssistant E2E",
  });
  if (!res.ok) throw new Error(explain(res.status, `Creating ${email}`));
  const body = (await res.json()) as { access_token: string };
  return { email, password: TEST_PASSWORD, token: body.access_token };
}

/** A fresh address that has NOT been registered — for driving the sign-up form. */
export function unusedEmail(): string {
  return `radassist-e2e-form-${Date.now()}@${TEST_DOMAIN}`;
}

/**
 * Put a signed-in session into the page before any app code runs.
 *
 * ⚠️  addInitScript, NOT an evaluate() after goto.
 * The chat page checks for a token on mount and redirects to /login if it is
 * missing. Writing the token after navigation loses that race about half the
 * time — the classic flaky-auth-test bug. addInitScript runs before the
 * document's own scripts, so the token is always there first.
 *
 * ⚠️  AND SEEDED ONCE PER CONTEXT, NOT ONCE PER NAVIGATION.
 * addInitScript runs before EVERY document. Writing the token unconditionally
 * reinstated it on every reload and every goto, so signing out inside a test
 * appeared not to work — the next navigation silently signed the user back
 * in, and no test could observe a sign-out at all. sessionStorage already
 * survives a reload within the tab, so seeding once is both sufficient and
 * faithful to how the app really behaves.
 */
export async function signIn(page: Page, account: TestAccount): Promise<void> {
  await page.addInitScript(
    ([token, email]) => {
      const SENTINEL = "__e2e_session_seeded";
      if (sessionStorage.getItem(SENTINEL)) return;
      sessionStorage.setItem(SENTINEL, "1");
      sessionStorage.setItem("radassist.token", token);
      sessionStorage.setItem("radassist.email", email);
    },
    [account.token, account.email] as const
  );
}

/** Read the persisted conversation list for an account. */
export async function readStoredChats(
  page: Page,
  email: string
): Promise<Array<{ id: string; title: string; messages: unknown[] }>> {
  return page.evaluate((who) => {
    const raw = localStorage.getItem(`radassist.chats.v1.${who.toLowerCase()}`);
    return raw ? JSON.parse(raw) : [];
  }, email);
}

/**
 * Wait for an answer to finish streaming.
 *
 * Keyed on the caret disappearing rather than on a timer: generation time
 * varies with the question, the provider and whether the embedding cache is
 * warm, and a fixed sleep is either flaky or slow.
 */
export async function waitForAnswer(page: Page, timeout = 90_000) {
  const answer = page.locator(".msg-assistant").last();
  await expect(answer).toBeVisible({ timeout: 20_000 });
  await expect(page.locator(".stream-caret")).toHaveCount(0, { timeout });
  return answer;
}

/** Type into the composer and send. */
export async function ask(page: Page, question: string) {
  await page.locator(".composer textarea").fill(question);
  await page.locator(".composer-send").click();
}

/** A sidebar row by its visible title. */
export function chatRow(page: Page, title: string | RegExp) {
  return page.locator(".chat-row-open").filter({ hasText: title });
}
