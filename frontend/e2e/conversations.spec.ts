/**
 * Conversation management — the sidebar, the store, and the boundary between
 * accounts.
 *
 * ⚠️  THIS IS THE SPEC THAT MATTERS MOST.
 * "New chat" and "go back to a previous chat" were a picture of a feature
 * before this rewrite: the sidebar kept its own array of titles, nothing was
 * ever written to, and clicking a row only moved a highlight. Every test
 * below asserts on the transcript, not on the highlight — the highlight was
 * the thing that already worked.
 *
 * Most of these never call the LLM. They seed transcripts into storage and
 * assert on how the app reads them back, which is where the logic lives.
 */

import { test, expect } from "./support/fixtures";
import { chatRow, readStoredChats, signIn } from "./support/session";

/** Seed a conversation list before the app boots. */
async function seedChats(
  page: import("@playwright/test").Page,
  email: string,
  chats: Array<{ id: string; title: string; messages?: unknown[]; age?: number }>
) {
  await page.addInitScript(
    ([who, list]) => {
      // ⚠️  SEED ONCE PER CONTEXT, NOT ONCE PER NAVIGATION.
      // addInitScript runs before EVERY document, reloads included. Seeding
      // unconditionally meant a reload silently restored the original
      // fixture — so "history survives a reload" passed without the app's
      // storage layer being involved at all, and a genuine delete looked
      // like it had failed to persist. The sentinel makes the seed a
      // starting state rather than a permanent one.
      const key = `radassist.chats.v1.${(who as string).toLowerCase()}`;
      const sentinel = `${key}.__seeded`;
      if (localStorage.getItem(sentinel)) return;
      localStorage.setItem(sentinel, "1");

      const now = Date.now();
      localStorage.setItem(
        key,
        JSON.stringify(
          (list as Array<{ id: string; title: string; messages?: unknown[]; age?: number }>).map(
            (c) => ({
              id: c.id,
              title: c.title,
              createdAt: now - (c.age ?? 0),
              updatedAt: now - (c.age ?? 0),
              messages: c.messages ?? [],
            })
          )
        )
      );
    },
    [email, chats] as const
  );
}

/** A minimal two-turn transcript. */
const transcript = (q: string, a: string) => [
  { id: `u-${q.slice(0, 6)}`, role: "user", content: q },
  { id: `a-${q.slice(0, 6)}`, role: "assistant", content: a, model: "seeded" },
];

/**
 * Every test here runs as the same shared account, and that is safe:
 * conversations live in localStorage and Playwright gives each test a fresh
 * browser context, so two tests using one account still see independent,
 * empty storage. The account supplies a token, nothing more.
 */
test.describe("Conversations", () => {
  test("New chat creates a real conversation, not just a row", async ({ page, primary: account }) => {
    await signIn(page, account);
    await seedChats(page, account.email, [
      { id: "a", title: "Pneumothorax findings", messages: transcript("q1", "answer one") },
    ]);
    await page.goto("/");

    await expect(page.locator(".chat-row")).toHaveCount(1);
    await expect(page.locator(".bubble-user")).toHaveCount(1);

    await page.getByRole("button", { name: "Start a new chat" }).click();

    // Two rows, and the thread is genuinely empty — not the previous
    // conversation with a highlight moved off it.
    await expect(page.locator(".chat-row")).toHaveCount(2);
    await expect(page.getByRole("heading", { name: "How can I help?" })).toBeVisible();
    await expect(page.locator(".bubble-user")).toHaveCount(0);
  });

  test("clicking a previous conversation restores its transcript", async ({ page, primary: account }) => {
    await signIn(page, account);
    await seedChats(page, account.email, [
      { id: "a", title: "Chat A", messages: transcript("about pneumothorax", "answer A") },
      { id: "b", title: "Chat B", age: 5000, messages: transcript("about nodules", "answer B") },
    ]);
    await page.goto("/");

    await expect(page.locator(".bubble-user")).toHaveText(["about pneumothorax"]);

    await chatRow(page, "Chat B").click();
    await expect(page.locator(".bubble-user")).toHaveText(["about nodules"]);
    await expect(page.locator(".bubble-assistant")).toContainText("answer B");
    await expect(page.locator(".topbar-title")).toHaveText("Chat B");

    await chatRow(page, "Chat A").click();
    await expect(page.locator(".bubble-user")).toHaveText(["about pneumothorax"]);
  });

  test("history survives a full page reload", async ({ page, primary: account }) => {
    await signIn(page, account);
    await seedChats(page, account.email, [
      { id: "a", title: "Chat A", messages: transcript("first question", "answer A") },
      { id: "b", title: "Chat B", age: 5000, messages: transcript("second", "answer B") },
    ]);
    await page.goto("/");
    await expect(page.locator(".chat-row")).toHaveCount(2);

    // Create a third conversation so the reload has to reproduce something
    // the fixture did not contain — otherwise this passes on the seed alone.
    await page.getByRole("button", { name: "Start a new chat" }).click();
    await expect(page.locator(".chat-row")).toHaveCount(3);
    await page.waitForTimeout(900);  // let the debounced write land

    await page.reload();

    await expect(page.locator(".chat-row")).toHaveCount(3);
    // Newest first, and opened on the newest.
    await expect(page.locator(".chat-row-title").first()).toHaveText("New chat");
    await chatRow(page, "Chat A").click();
    await expect(page.locator(".bubble-user")).toHaveText(["first question"]);
  });

  test("New chat twice does not stack up empty rows", async ({ page, primary: account }) => {
    await signIn(page, account);
    await page.goto("/");

    const newChat = page.getByRole("button", { name: "Start a new chat" });
    await newChat.click();
    await newChat.click();
    await newChat.click();

    // A blank conversation already open IS the new chat.
    await expect(page.locator(".chat-row")).toHaveCount(1);
  });

  test("deleting the active conversation falls through to the next", async ({ page, primary: account }) => {
    await signIn(page, account);
    await seedChats(page, account.email, [
      { id: "a", title: "Chat A", messages: transcript("first question", "answer A") },
      { id: "b", title: "Chat B", age: 5000, messages: transcript("second question", "answer B") },
    ]);
    await page.goto("/");
    await expect(page.locator(".bubble-user")).toHaveText(["first question"]);

    await page.getByRole("button", { name: "Delete Chat A" }).click();

    await expect(page.locator(".chat-row")).toHaveCount(1);
    // Falls through rather than blanking out.
    await expect(page.locator(".bubble-user")).toHaveText(["second question"]);
    await expect(page.locator(".topbar-title")).toHaveText("Chat B");

    // The debounce has to flush before the reload, or this asserts nothing
    // about persistence.
    await page.waitForTimeout(900);
    await page.reload();
    await expect(page.locator(".chat-row")).toHaveCount(1);
    await expect(page.locator(".chat-row-title")).toHaveText("Chat B");
  });

  test("deleting the last conversation leaves a usable empty state", async ({ page, primary: account }) => {
    await signIn(page, account);
    await seedChats(page, account.email, [
      { id: "a", title: "Only Chat", messages: transcript("q", "a") },
    ]);
    await page.goto("/");

    await page.getByRole("button", { name: "Delete Only Chat" }).click();

    await expect(page.locator(".chat-row")).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "How can I help?" })).toBeVisible();
    await expect(page.locator(".sidebar-empty")).toBeVisible();
    // The composer must still work — not a dead screen.
    await expect(page.locator(".composer textarea")).toBeEnabled();
  });

  test("one account cannot see another's conversations", async ({
    page,
    primary: account,
    secondary: other,
  }) => {
    // ⚠️  THE ONE TEST THAT IS ABOUT DISCLOSURE, NOT CONVENIENCE.
    // Transcripts are kept in localStorage, which outlives the tab — unlike
    // the token, which does not. Partitioning by account is the whole reason
    // that is acceptable on a shared reading-room workstation. If this test
    // ever fails, the storage strategy has to change, not the test.
    await signIn(page, account);
    await seedChats(page, account.email, [
      { id: "a", title: "Confidential Chat", messages: transcript("private question", "private answer") },
    ]);
    await page.goto("/");
    await expect(chatRow(page, "Confidential Chat")).toBeVisible();

    await page.getByRole("button", { name: "Sign out" }).click();
    await expect(page).toHaveURL(/\/login/);

    await page.getByLabel("Email").fill(other.email);
    await page.locator("#password").fill(other.password);
    await page.getByRole("button", { name: "Sign in" }).click();
    await expect(page.getByRole("heading", { name: "How can I help?" })).toBeVisible({
      timeout: 30_000,
    });

    await expect(page.locator(".chat-row")).toHaveCount(0);
    await expect(page.getByText("Confidential Chat")).toHaveCount(0);
    await expect(page.getByText("private question")).toHaveCount(0);

    // And the first account's data is still intact under its own key —
    // isolation, not deletion.
    const theirs = await readStoredChats(page, account.email);
    expect(theirs).toHaveLength(1);
  });

  test("a conversation is titled from the first message, then left alone", async ({ page, primary: account }) => {
    await signIn(page, account);
    await seedChats(page, account.email, []);
    await page.goto("/");

    // Intercept so this test costs nothing and stays deterministic — it is
    // about the title, not the answer.
    await page.route("**/api/v1/chat", (route) =>
      route.fulfill({
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
        body:
          `event: token\ndata: ${JSON.stringify({ token: "Short answer." })}\n\n` +
          `event: done\ndata: ${JSON.stringify({ model: "e2e" })}\n\n`,
      })
    );

    await page.locator(".composer textarea").fill("What are the radiographic findings of pneumothorax?");
    await page.locator(".composer-send").click();
    await expect(page.locator(".stream-caret")).toHaveCount(0, { timeout: 30_000 });

    const title = await page.locator(".chat-row-title").first().textContent();
    expect(title).toContain("What are the radiographic");
    // Truncated on a word boundary, not mid-syllable.
    expect(title!.length).toBeLessThanOrEqual(44);

    // A second message must not rewrite the label.
    await page.locator(".composer textarea").fill("And in a supine patient?");
    await page.locator(".composer-send").click();
    await expect(page.locator(".stream-caret")).toHaveCount(0, { timeout: 30_000 });

    await expect(page.locator(".chat-row-title").first()).toHaveText(title!);
  });

  test("switching conversation mid-answer does not lose the answer", async ({ page, primary: account }) => {
    // ⚠️  REGRESSION TEST FOR A REAL BUG.
    // Streamed tokens were written to "the active conversation" rather than
    // to the one the send started in. Opening another chat while an answer
    // was still arriving sent every remaining token looking for a message
    // that lived somewhere else: the tokens were dropped, the `done` event
    // missed, and the half-written answer kept its caret forever.
    await signIn(page, account);
    await seedChats(page, account.email, [
      { id: "a", title: "Chat A" },
      { id: "b", title: "Chat B", age: 5000, messages: transcript("unrelated", "unrelated answer") },
    ]);

    // Hold the response open long enough that the switch is unambiguous.
    await page.route("**/api/v1/chat", async (route) => {
      await new Promise((r) => setTimeout(r, 2500));
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
        body:
          ["Pneumo", "thorax ", "is ", "air ", "in ", "the ", "pleural ", "space."]
            .map((t) => `event: token\ndata: ${JSON.stringify({ token: t })}\n\n`)
            .join("") + `event: done\ndata: ${JSON.stringify({ model: "e2e" })}\n\n`,
      });
    });

    await page.goto("/");
    await chatRow(page, "Chat A").click();
    await page.locator(".composer textarea").fill("What is a pneumothorax?");
    await page.locator(".composer-send").click();

    // Leave while it is still in flight.
    await page.waitForTimeout(400);
    await chatRow(page, "Chat B").click();
    await expect(page.locator(".bubble-user")).toHaveText(["unrelated"]);
    await page.waitForTimeout(3500);

    // Come back: complete answer, no stuck caret.
    await chatRow(page, "Chat A").click();
    await expect(page.locator(".bubble-assistant").last()).toContainText("pleural space.");
    await expect(page.locator(".stream-caret")).toHaveCount(0);

    // And the other conversation was not written into.
    const stored = await readStoredChats(page, account.email);
    const b = stored.find((c) => c.id === "b");
    expect(b?.messages).toHaveLength(2);
  });

  test("a transcript left mid-stream reloads without a stuck caret", async ({ page, primary: account }) => {
    await signIn(page, account);
    await seedChats(page, account.email, [
      {
        id: "a",
        title: "Interrupted",
        messages: [
          { id: "u", role: "user", content: "a question" },
          // What a tab closed mid-answer leaves behind.
          { id: "x", role: "assistant", content: "half an ans", isStreaming: true },
        ],
      },
    ]);
    await page.goto("/");

    await expect(page.locator(".bubble-assistant")).toContainText("half an ans");
    await expect(page.locator(".stream-caret")).toHaveCount(0);
  });

  test("corrupt stored history degrades to an empty list, not a blank app", async ({ page, primary: account }) => {
    await signIn(page, account);
    await page.addInitScript((who) => {
      localStorage.setItem(
        `radassist.chats.v1.${(who as string).toLowerCase()}`,
        "{ this is not json"
      );
    }, account.email);

    await page.goto("/");

    // The app must come up. Losing history is acceptable; a white screen
    // where the chat used to be is not.
    await expect(page.getByRole("heading", { name: "How can I help?" })).toBeVisible();
    await expect(page.locator(".composer textarea")).toBeEnabled();
  });
});

test.describe("Sidebar", () => {
  test("collapse state survives a reload", async ({ page, primary: account }) => {
    await signIn(page, account);
    await page.goto("/");

    const sidebar = page.locator(".sidebar");
    await expect(sidebar).toHaveAttribute("data-collapsed", "false");

    await page.getByRole("button", { name: "Collapse sidebar" }).click();
    await expect(sidebar).toHaveAttribute("data-collapsed", "true");
    // The way back must exist, or collapsing is a trap.
    await expect(page.getByRole("button", { name: "Open sidebar" })).toBeVisible();

    await page.reload();
    await expect(sidebar).toHaveAttribute("data-collapsed", "true");

    await page.getByRole("button", { name: "Open sidebar" }).click();
    await expect(sidebar).toHaveAttribute("data-collapsed", "false");
  });
});
