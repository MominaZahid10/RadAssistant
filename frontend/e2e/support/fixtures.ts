/**
 * Worker-scoped account fixtures.
 *
 * Two accounts exist for the whole run, not one per test. See the note at the
 * top of session.ts for why (the backend allows five registrations per hour,
 * and every test needs a token rather than a brand-new identity).
 *
 * Import `test` from here instead of from @playwright/test.
 */

import { test as base } from "@playwright/test";
import { ensureAccount, type TestAccount } from "./session";

interface WorkerFixtures {
  /** The account almost every test runs as. */
  primary: TestAccount;
  /**
   * A second, different account. Only needed by tests about the boundary
   * between accounts — signing out and in as somebody else.
   */
  secondary: TestAccount;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export const test = base.extend<{}, WorkerFixtures>({
  primary: [
    async ({}, use) => {
      await use(await ensureAccount("primary"));
    },
    { scope: "worker" },
  ],
  secondary: [
    async ({}, use) => {
      await use(await ensureAccount("secondary"));
    },
    { scope: "worker" },
  ],
});

export { expect } from "@playwright/test";
