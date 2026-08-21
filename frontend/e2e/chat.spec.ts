/**
 * The RAG round trip, against the real pipeline.
 *
 * ⚠️  THESE TWO TESTS SPEND LLM TOKENS. The rest of the suite does not.
 * Keep it that way. A suite with a per-run cost gets skipped, and a skipped
 * suite is worse than no suite because it looks like coverage.
 *
 * ⚠️  NOTHING HERE ASSERTS ON THE WORDING OF AN ANSWER.
 * Generation is not deterministic. Asserting that a reply contains "pleural"
 * produces a test that fails on a perfectly good answer, gets marked flaky,
 * and is then ignored. What these assert is *structural*: tokens arrived,
 * sources came back, a citation resolves to the passage it points at, the
 * error path is legible. Answer quality is the eval harness's job
 * (`backend/eval/run_eval.py`) — that is what it is for, and it measures
 * retrieval properly instead of guessing from one sample.
 */

import { test, expect } from "./support/fixtures";
import { API_URL, ask, signIn, waitForAnswer } from "./support/session";

test.describe("Chat (hits the real LLM)", () => {
  test("a question returns a grounded answer with traceable sources", async ({ page, primary: account }) => {
    await signIn(page, account);
    await page.goto("/");

    await ask(page, "What are the radiographic findings of a pneumothorax?");

    // Something must be shown while the pipeline works — silence reads as a
    // broken button.
    await expect(page.getByText(/Searching the knowledge base/i)).toBeVisible({
      timeout: 20_000,
    });

    await waitForAnswer(page);

    const answer = page.locator(".bubble-assistant").last();
    const text = (await answer.textContent()) ?? "";
    expect(text.trim().length).toBeGreaterThan(80);
    // A failure surfaced as an answer is the failure mode this project keeps
    // running into, so check we are not looking at one.
    await expect(answer).not.toHaveClass(/bubble-assistant--error/);

    // ── Sources ──
    // The grounding prompt requires citations. If none came back, retrieval
    // returned nothing and the model answered from its own weights — which
    // is the single most important thing for this product not to do.
    const sourcesToggle = page.getByRole("button", { name: /\d+ sources?/ });
    await expect(sourcesToggle).toBeVisible();

    await sourcesToggle.click();
    const cards = page.locator(".source-card");
    expect(await cards.count()).toBeGreaterThan(0);
    await expect(cards.first().locator(".source-title")).not.toBeEmpty();

    // ── A citation you can actually follow ──
    // Rendering [3] as text makes the grounding requirement decorative. The
    // chip has to reach the passage it names.
    const chip = page.locator(".citation-chip").first();
    if ((await chip.count()) > 0) {
      const n = (await chip.textContent())?.trim();
      await chip.click();
      const target = page.locator(`[id$="-${n}"].source-card`);
      await expect(target).toBeVisible();
      await expect(target).toBeInViewport();
    }

    // The transcript survives a reload — a real answer, not a seeded one.
    await page.reload();
    await expect(page.locator(".bubble-assistant").last()).toContainText(
      text.slice(0, 40).trim()
    );
  });

  test("a backend failure is reported in the conversation, not swallowed", async ({ page, primary: account }) => {
    await signIn(page, account);

    // Simulate the provider being down, which is the realistic failure —
    // a 500 from the chat route rather than a network partition.
    await page.route(`${API_URL}/api/v1/chat`, (route) =>
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: '{"detail":"LLM provider unavailable"}',
      })
    );

    await page.goto("/");
    await ask(page, "What is a pneumothorax?");

    const answer = page.locator(".bubble-assistant").last();
    await expect(answer).toHaveClass(/bubble-assistant--error/, { timeout: 30_000 });
    await expect(answer).not.toBeEmpty();
    // No caret left spinning on a request that will never finish.
    await expect(page.locator(".stream-caret")).toHaveCount(0);

    // And the user is not stuck — the composer takes another attempt.
    // (Send is correctly disabled while the box is empty, so type first;
    // asserting on the empty state would be asserting the wrong thing.)
    await expect(page.locator(".composer textarea")).toBeEnabled();
    await page.locator(".composer textarea").fill("try again");
    await expect(page.locator(".composer-send")).toBeEnabled();
  });

  test("the mode control switches what gets produced", async ({ page, primary: account }) => {
    // Cheap: asserts the control wiring, does not generate anything.
        await signIn(page, account);
    await page.goto("/");

    const composer = page.locator(".composer textarea");
    await expect(composer).toHaveAttribute("placeholder", /Ask a radiology question/i);

    await page.getByRole("button", { name: "Draft", exact: true }).click();
    await expect(composer).toHaveAttribute("placeholder", /Dictate findings/i);
    // Register is meaningless for a document going into a record.
    await expect(page.locator(".composer-pill")).toBeDisabled();

    await page.getByRole("button", { name: "Compare", exact: true }).click();
    await expect(composer).toHaveAttribute("placeholder", /prior study/i);

    await page.getByRole("button", { name: "Ask", exact: true }).click();
    await expect(page.locator(".composer-pill")).toBeEnabled();
    await expect(page.locator(".composer-pill")).toHaveText("Attending");
    await page.locator(".composer-pill").click();
    await expect(page.locator(".composer-pill")).toHaveText("Resident");
  });

  test("the send button is disabled until there is something to send", async ({ page, primary: account }) => {
    await signIn(page, account);
    await page.goto("/");

    await expect(page.locator(".composer-send")).toBeDisabled();
    await page.locator(".composer textarea").fill("x");
    await expect(page.locator(".composer-send")).toBeEnabled();
    await page.locator(".composer textarea").fill("   ");
    // Whitespace is not a question.
    await expect(page.locator(".composer-send")).toBeDisabled();
  });

  test("an example prompt fills the composer", async ({ page, primary: account }) => {
    await signIn(page, account);
    await page.goto("/");

    await page.locator(".welcome-prompts button").first().click();
    await expect(page.locator(".composer textarea")).not.toBeEmpty();
    await expect(page.locator(".composer-send")).toBeEnabled();
  });
});
