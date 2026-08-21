"use client";

/**
 * Sign in / create account.
 *
 * ⚠️  SIGNUP IS SAFE BECAUSE OF OWNERSHIP, NOT INSTEAD OF IT.
 * It was rejected while every signed-in user could see every report — anyone
 * with the URL could then read uploaded patient material. Reports and images
 * now carry an owner, so a new account lands in an empty workspace.
 *
 * The server can switch registration off entirely with
 * ALLOW_REGISTRATION=false, which a clinical deployment should do; accounts
 * there come from an operator running scripts/create_user.py. This form
 * handles that 403 by saying so rather than failing obscurely.
 *
 * There is no "forgot password". A reset flow is another credential path
 * needing the same protection as the password itself, and for a pilot the
 * recovery procedure is that the operator recreates the account.
 *
 * LAYOUT: one form, two modes. A separate /signup route would duplicate the
 * split-screen layout and double the places a change has to be made.
 */

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { authApi } from "@/lib/api";
import { isSignedIn, setSession } from "@/lib/auth";
import Brand from "@/components/Brand";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [reveal, setReveal] = useState(false);

  const signup = mode === "signup";

  // Already signed in — skip the form.
  useEffect(() => {
    if (isSignedIn()) router.replace("/");
  }, [router]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;

    setBusy(true);
    setError(null);
    try {
      const result = signup
        ? await authApi.register(email.trim(), password, fullName.trim())
        : await authApi.login(email.trim(), password);
      setSession(result.access_token, email.trim().toLowerCase());
      router.replace("/");
    } catch (err) {
      // ⚠️  SHOW THE SERVER'S WORDING, DO NOT REFINE IT.
      // The API deliberately returns one message for unknown-account,
      // wrong-password and disabled-account. A more "helpful" message here —
      // "no account with that email" — would reintroduce exactly the account
      // enumeration the backend goes to some trouble to prevent.
      setError(
        err instanceof Error ? err.message : "Sign-in failed. Please try again."
      );
      setPassword("");
      setReveal(false);
    } finally {
      setBusy(false);
    }
  }

  function switchMode() {
    setMode(signup ? "signin" : "signup");
    setError(null);
    setPassword("");
    setReveal(false);
  }

  return (
    <div className="auth-shell">
      {/* ── Left: brand panel ── */}
      <div className="auth-art">
        <div className="auth-mark">
          <Brand size={26} />
          RadAssistant
        </div>

        <div className="auth-figure" aria-hidden="true">
          {/* A thorax rendered as scan geometry rather than an anatomical
              drawing: rib arcs and lung fields at low opacity inside the
              gantry circle. Vector, so it stays crisp at any panel width and
              costs no image request. */}
          <svg viewBox="0 0 300 300" fill="none">
            <circle cx="150" cy="150" r="118" stroke="rgba(255,255,255,.10)" strokeWidth="1" />
            <circle cx="150" cy="150" r="88" stroke="rgba(255,255,255,.08)" strokeWidth="1" />
            <g stroke="rgba(255,255,255,.55)" strokeWidth="2" strokeLinecap="round">
              <path d="M150 78v150" />
              <path d="M150 92c-22 0-42 4-58 11M150 92c22 0 42 4 58 11" />
              <path d="M150 114c-24 0-45 4-62 12M150 114c24 0 45 4 62 12" />
              <path d="M150 136c-24 0-44 4-60 12M150 136c24 0 44 4 60 12" />
              <path d="M150 158c-21 0-39 4-53 11M150 158c21 0 39 4 53 11" />
              <path d="M150 180c-17 0-32 3-44 9M150 180c17 0 32 3 44 9" />
              <path d="M92 103c-6 30-6 60 0 90M208 103c6 30 6 60 0 90" />
            </g>
            <g stroke="rgba(255,255,255,.28)" strokeWidth="1.5">
              <path d="M112 128c-14 22-16 46-6 68 14-16 20-38 18-62-1-6-6-10-12-6z" />
              <path d="M188 128c14 22 16 46 6 68-14-16-20-38-18-62 1-6 6-10 12-6z" />
            </g>
            <circle
              cx="150"
              cy="150"
              r="118"
              stroke="rgba(255,255,255,.2)"
              strokeWidth="1.5"
              strokeDasharray="4 10"
            />
          </svg>
        </div>

        <div className="auth-copy">
          <h2>Answers with the source attached.</h2>
          <p>
            Ask questions, draft structured reports, compare against priors.
            Every line links back to the literature behind it.
          </p>
        </div>
      </div>

      {/* ── Right: the form ── */}
      <div className="auth-panel">
        <form className="auth-form" onSubmit={submit}>
          <h1>{signup ? "Create account" : "Sign in"}</h1>
          <p className="auth-sub">
            {signup
              ? "Set up your RadAssistant workspace."
              : "Welcome back to RadAssistant."}
          </p>

          {signup && (
            <div className="auth-field">
              <label htmlFor="fullName">Full name</label>
              <input
                id="fullName"
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                autoComplete="name"
              />
            </div>
          )}

          <div className="auth-field">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="username"
              required
              autoFocus
            />
          </div>

          <div className="auth-field">
            <label htmlFor="password">Password</label>
            <div className="auth-input-wrap">
              <input
                id="password"
                className="has-reveal"
                type={reveal ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={signup ? "new-password" : "current-password"}
                required
                minLength={signup ? 12 : undefined}
              />
              <button
                type="button"
                className="auth-reveal"
                onClick={() => setReveal((v) => !v)}
                // The label states the ACTION, not the state — a screen
                // reader announcing "password visible" on a button that
                // hides it is the wrong way round.
                aria-label={reveal ? "Hide password" : "Show password"}
                aria-pressed={reveal}
                title={reveal ? "Hide password" : "Show password"}
              >
                {reveal ? (
                  <svg
                    width="17"
                    height="17"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.7"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M9.9 5.5A9.9 9.9 0 0 1 12 5.3c6.6 0 10.2 6.7 10.2 6.7a18.6 18.6 0 0 1-3 4M6.2 6.4A18.4 18.4 0 0 0 1.8 12S5.4 18.7 12 18.7a9.8 9.8 0 0 0 4.2-.9" />
                    <path d="M10 10a3 3 0 0 0 4.2 4.2M2 2l20 20" />
                  </svg>
                ) : (
                  <svg
                    width="17"
                    height="17"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.7"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M1.8 12S5.4 5.3 12 5.3 22.2 12 22.2 12 18.6 18.7 12 18.7 1.8 12 1.8 12z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                )}
              </button>
            </div>
            {signup && (
              <span className="auth-hint">At least 12 characters.</span>
            )}
          </div>

          {error && <p className="auth-error">{error}</p>}

          <button type="submit" className="auth-submit" disabled={busy}>
            {busy
              ? signup
                ? "Creating account…"
                : "Signing in…"
              : signup
              ? "Create account"
              : "Sign in"}
          </button>

          <p className="auth-alt">
            {signup ? "Already have an account? " : "No account? "}
            <button type="button" onClick={switchMode}>
              {signup ? "Sign in" : "Create one"}
            </button>
          </p>

          {/* ⚠️  THIS LINE EARNS ITS SPACE ON A PUBLIC DEPLOYMENT.
              It was removed when this was a private tool, which was right:
              nobody needs telling what they already know about their own
              instance. Reachable from the internet and labelled "radiology
              decision support", it is the difference between a portfolio
              project and something that reads as a clinical service — and
              the only warning a stranger sees before they upload a file. */}
          <p className="auth-note">
            Demonstration system. For anonymised and synthetic data only —
            do not upload real patient information.
          </p>
        </form>
      </div>
    </div>
  );
}
