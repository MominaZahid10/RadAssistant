/**
 * Sign in, sign up, and the boundary between them.
 *
 * These drive the actual form — this is the one spec that does. Everything
 * else seeds a session directly, so a copy change here doesn't cascade into
 * unrelated failures.
 */

import { test, expect } from "./support/fixtures";
import { API_URL, TEST_PASSWORD, signIn, unusedEmail } from "./support/session";

test.describe("Authentication", () => {
  test("a signed-out visitor is sent to the sign-in page", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveURL(/\/login/);
    await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();
  });

  test("the sign-in page shows no app furniture", async ({ page }) => {
    // The sidebar used to render here: a chat list, a New chat button and a
    // sign-out control, shown to somebody who is not signed in.
    await page.goto("/login");
    await expect(page.locator(".sidebar")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Start a new chat" })).toHaveCount(0);
  });

  test("a new account can be created and lands in an empty workspace", async ({
    page,
  }) => {
    // ⚠️  THE ONE TEST THAT SPENDS A REGISTRATION.
    // The backend allows five per hour, and this test cannot be written
    // without one: it exists to prove a first-ever sign-up works. Everything
    // else reuses the shared accounts.
    const email = unusedEmail();

    await page.goto("/login");
    await page.getByRole("button", { name: "Create one" }).click();

    await expect(page.getByRole("heading", { name: "Create account" })).toBeVisible();
    await page.getByLabel("Full name").fill("E2E Form User");
    await page.getByLabel("Email").fill(email);
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.getByRole("button", { name: "Create account" }).click();

    await expect(page).toHaveURL(/\/$/, { timeout: 30_000 });
    await expect(page.getByRole("heading", { name: "How can I help?" })).toBeVisible();

    // ⚠️  THE POINT OF THE TEST. Open signup is only safe because a new
    // account sees nothing. If this list is ever non-empty, ownership is
    // broken and signup has to be switched off.
    await expect(page.locator(".chat-row")).toHaveCount(0);
    await expect(page.locator(".sidebar-email")).toHaveText(email.toLowerCase());
  });

  test("an existing account can sign in", async ({ page, primary: account }) => {
    await page.goto("/login");
    await page.getByLabel("Email").fill(account.email);
    await page.locator("#password").fill(account.password);
    await page.getByRole("button", { name: "Sign in" }).click();

    await expect(page.getByRole("heading", { name: "How can I help?" })).toBeVisible({
      timeout: 30_000,
    });
  });

  test("a wrong password is rejected without revealing whether the account exists", async ({
    page,
    primary: account,
  }) => {
    await page.goto("/login");
    await page.getByLabel("Email").fill(account.email);
    await page.locator("#password").fill("definitely-not-the-password");
    await page.getByRole("button", { name: "Sign in" }).click();

    const error = page.locator(".auth-error");
    await expect(error).toBeVisible();

    // ⚠️  THIS ASSERTION IS A SECURITY CONTROL, NOT A COPY CHECK.
    // The backend returns one message for unknown-account, wrong-password
    // and disabled-account precisely so the form cannot be used to enumerate
    // who has an account. A "helpful" rewording in the UI would undo that,
    // and would look like an improvement in review.
    const text = (await error.textContent())?.toLowerCase() ?? "";
    expect(text).not.toContain("no account");
    expect(text).not.toContain("not found");
    expect(text).not.toContain("does not exist");

    // The field is cleared so a shoulder-surfer can't read the failed attempt.
    await expect(page.locator("#password")).toHaveValue("");
    await expect(page).toHaveURL(/\/login/);
  });

  test("the password can be revealed and hidden again", async ({ page }) => {
    await page.goto("/login");
    const field = page.locator("#password");
    await field.fill("hunter2-hunter2");

    await expect(field).toHaveAttribute("type", "password");
    await page.getByRole("button", { name: "Show password" }).click();
    await expect(field).toHaveAttribute("type", "text");
    await page.getByRole("button", { name: "Hide password" }).click();
    await expect(field).toHaveAttribute("type", "password");
  });

  test("signing out clears the session and blocks the back button", async ({
    page,
    primary: account,
  }) => {
    // Signs in through the FORM rather than by seeding storage. This test is
    // specifically about the session being torn down, so nothing here may
    // put a token back afterwards.
    await page.goto("/login");
    await page.getByLabel("Email").fill(account.email);
    await page.locator("#password").fill(account.password);
    await page.getByRole("button", { name: "Sign in" }).click();
    await expect(page.getByRole("heading", { name: "How can I help?" })).toBeVisible({
      timeout: 30_000,
    });

    await page.getByRole("button", { name: "Sign out" }).click();
    await expect(page).toHaveURL(/\/login/);

    // ⚠️  A SHARED READING-ROOM WORKSTATION IS THE THREAT MODEL.
    // Returning to the app must not restore the previous clinician's
    // session — anything they approve next would carry the wrong signature.
    //
    // Asserted by navigating rather than by page.goBack(): sign-out uses
    // router.replace, so there is no authed entry left in history to go back
    // to and goBack() lands on about:blank. That is the desired behaviour,
    // but it tests the history stack rather than the session, and a direct
    // navigation is the case that actually matters — someone typing the URL
    // or restoring a tab.
    await page.goto("/");
    await expect(page).toHaveURL(/\/login/);
    await expect(page.locator(".chat-row")).toHaveCount(0);
  });

  test("an expired token drops the user back to sign-in mid-session", async ({
    page,
    primary: account,
  }) => {
    await signIn(page, account);
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "How can I help?" })).toBeVisible();

    // Force the 401 path the way a twelve-hour shift would.
    await page.route(`${API_URL}/api/v1/**`, (route) =>
      route.fulfill({ status: 401, body: '{"detail":"Token expired"}' })
    );

    await page.locator(".composer textarea").fill("anything");
    await page.locator(".composer-send").click();

    await expect(page).toHaveURL(/\/login/, { timeout: 30_000 });
  });
});
