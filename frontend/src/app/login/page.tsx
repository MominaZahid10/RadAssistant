"use client";

/**
 * Sign-in (Phase 6, Step 6).
 *
 * ⚠️  SIGNUP IS SAFE BECAUSE OF OWNERSHIP, NOT INSTEAD OF IT.
 * It was rejected while every signed-in user could see every report — anyone
 * with the URL could then read uploaded patient material. Reports and images
 * now carry an owner, so a new account lands in an empty workspace.
 *
 * The server can switch it off entirely with ALLOW_REGISTRATION=false, which
 * a clinical deployment should do; accounts there come from an operator
 * running scripts/create_user.py. This form handles that 403 by saying so
 * rather than failing obscurely.
 *
 * There is no "forgot password". A reset flow is another credential path
 * needing the same protection as the password itself, and for a pilot the
 * recovery procedure is that the operator recreates the account.
 */

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { authApi } from "@/lib/api";
import { isSignedIn, setSession } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // One form, two modes. A separate /signup route would duplicate the
  // layout and double the places a change has to be made.
  const [mode, setMode] = useState<"signin" | "signup">("signin");

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
      const result =
        mode === "signup"
          ? await authApi.register(email.trim(), password, fullName.trim())
          : await authApi.login(email.trim(), password);
      setSession(result.access_token, email.trim().toLowerCase());
      router.replace("/");
    } catch (err) {
      // ⚠️  SHOW THE SERVER'S WORDING, DO NOT REFINE IT.
      // The API deliberately returns one message for unknown-account,
      // wrong-password and disabled-account. Adding a more "helpful" message
      // here — "no account with that email" — would reintroduce exactly the
      // account enumeration the backend goes to some trouble to prevent.
      setError(
        err instanceof Error ? err.message : "Sign-in failed. Please try again."
      );
      setPassword("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-shell">
      <form className="login-card" onSubmit={submit}>
        <div className="login-brand">
          <span className="login-logo">🩻</span>
          <h1>RadAssist AI</h1>
          <p>Radiology reporting and decision support</p>
        </div>

        {mode === "signup" && (
          <label className="login-field">
            <span>Name</span>
            <input
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              autoComplete="name"
            />
          </label>
        )}

        <label className="login-field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="username"
            required
            autoFocus
          />
        </label>

        <label className="login-field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={
              mode === "signup" ? "new-password" : "current-password"
            }
            required
            minLength={mode === "signup" ? 12 : undefined}
          />
          {mode === "signup" && (
            <span className="login-hint">At least 12 characters.</span>
          )}
        </label>

        {error && <p className="login-error">{error}</p>}

        <button type="submit" className="login-submit" disabled={busy}>
          {busy
            ? mode === "signup"
              ? "Creating account…"
              : "Signing in…"
            : mode === "signup"
            ? "Create account"
            : "Sign in"}
        </button>

        <button
          type="button"
          className="login-switch"
          onClick={() => {
            setMode(mode === "signin" ? "signup" : "signin");
            setError(null);
            setPassword("");
          }}
        >
          {mode === "signin"
            ? "Create an account"
            : "Already have an account? Sign in"}
        </button>

        <p className="login-note">
          For anonymised and synthetic data only. Not for real patient
          information.
        </p>
      </form>
    </div>
  );
}
