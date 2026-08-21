/**
 * RadAssist AI — Client-side auth (Phase 6, Step 6)
 *
 * ⚠️  sessionStorage, NOT localStorage.
 *
 * A token in localStorage survives browser restarts. On a shared clinical
 * workstation — which is what a radiology reading room is — that means the
 * next person to sit down is still signed in as the last one, and any report
 * they approve is attributed to somebody who went home an hour ago.
 *
 * sessionStorage is scoped to the tab and cleared when it closes. Signing in
 * again each session is a small cost; a signature attached to the wrong
 * clinician is not.
 *
 * ⚠️  AND WHY NOT A COOKIE.
 * An httpOnly cookie would be safer against XSS, which is the real argument
 * for it. But it needs the API and the app on the same site, or CORS with
 * credentials plus CSRF protection — none of which exists yet, and half of
 * which is easy to get subtly wrong. A bearer token in sessionStorage is the
 * honest choice for this deployment, and the tradeoff is written down rather
 * than assumed away.
 */

const TOKEN_KEY = "radassist.token";
const EMAIL_KEY = "radassist.email";

/** Fires when the token is cleared, so the UI can send the user to /login. */
export const AUTH_EXPIRED_EVENT = "radassist:auth-expired";

export interface LoginResult {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface CurrentUser {
  id: string;
  email: string;
  full_name: string | null;
  is_admin: boolean;
  last_login_at: string | null;
  created_at: string;
}

/** The stored token, or null. Safe to call during SSR. */
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage.getItem(TOKEN_KEY);
  } catch {
    // Private browsing modes can throw on storage access. Treat as signed out
    // rather than crashing the whole app on a storage policy.
    return null;
  }
}

export function getStoredEmail(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage.getItem(EMAIL_KEY);
  } catch {
    return null;
  }
}

export function setSession(token: string, email: string): void {
  try {
    window.sessionStorage.setItem(TOKEN_KEY, token);
    window.sessionStorage.setItem(EMAIL_KEY, email);
  } catch {
    /* see getToken */
  }
}

/**
 * Sign out. Also fired automatically when the API returns 401, so an expired
 * token does not leave the user clicking a dead interface.
 */
export function clearSession(notify = true): void {
  try {
    window.sessionStorage.removeItem(TOKEN_KEY);
    window.sessionStorage.removeItem(EMAIL_KEY);
  } catch {
    /* see getToken */
  }
  if (notify && typeof window !== "undefined") {
    window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
  }
}

export function isSignedIn(): boolean {
  return Boolean(getToken());
}

/**
 * Subscribe to session changes, for `useSyncExternalStore`.
 *
 * ⚠️  WHY THIS EXISTS RATHER THAN `useEffect(() => setEmail(getStoredEmail()))`.
 * sessionStorage cannot be read during render — the server has no such
 * storage, so the markup it produces would disagree with the client's first
 * paint and React would discard the tree. Reading it in an effect and calling
 * setState fixes the mismatch but costs a second render pass on every mount
 * and is exactly the pattern React now flags. `useSyncExternalStore` is the
 * primitive built for this case: it takes a server snapshot (nobody is signed
 * in) and a client snapshot (read the store), with no extra render.
 *
 * Returns an unsubscribe function, per the hook's contract.
 */
export function subscribeToSession(onChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  window.addEventListener(AUTH_EXPIRED_EVENT, onChange);
  // Fired when ANOTHER tab writes to storage — a second tab signing out
  // should not leave this one showing a stale identity.
  window.addEventListener("storage", onChange);
  return () => {
    window.removeEventListener(AUTH_EXPIRED_EVENT, onChange);
    window.removeEventListener("storage", onChange);
  };
}

/** Server snapshot for the hook above: never signed in during SSR. */
export function serverEmailSnapshot(): string | null {
  return null;
}

/** Header to attach to an authenticated request. Empty when signed out. */
export function authHeader(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
