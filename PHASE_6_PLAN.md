# Phase 6 — Explainability, Auth & Hardening

**Plan document:** *Full evidence-citation UI, JWT-based auth, input validation,
error handling, logging/monitoring, basic test coverage.* (Week 11)

---

## What the phase actually still needs

Two of the six stated deliverables are already done, which changes the shape of
the work.

| Deliverable | Status | Evidence |
|---|---|---|
| Full evidence-citation UI | **Done** | Evidence panel, clickable `[N]` chips, figure thumbnails, per-source scores |
| Basic test coverage | **Done** | 455 tests |
| Input validation | **Mostly** | Pydantic on every route, upload size limits, path containment |
| Error handling | **Partial** | Graceful degradation throughout; but exception text is returned to clients |
| Logging / monitoring | **Partial** | Logs exist, but `DEBUG=true` makes SQLAlchemy echo every statement |
| JWT auth | **Not started** | No authentication anywhere |

So Phase 6 is **auth plus hardening**, not explainability.

---

## The four things that are actually wrong right now

Stated plainly, because they should be fixed in this order.

**1. Every endpoint is open.** Anyone who can reach port 8000 can list every
report, read every draft, and download every uploaded image. Those images are
photographs of patient reports. The project document allows only anonymised or
synthetic data during development, which is what makes this survivable so far —
it is not a control, it is a constraint on what you may put in.

**2. Report text is being written into the logs.** `DEBUG=true` sets
`echo=settings.DEBUG` on the SQLAlchemy engine, so every INSERT is logged in
full — including `findings_input` and `ai_draft`. Docker keeps those logs.
A privacy problem hiding as a verbosity setting.

**3. Exception text is returned to callers.**

```python
raise HTTPException(status_code=500, detail=f"Internal error: {e}")
```

That hands file paths, driver messages and occasionally connection strings to
whoever made the request.

**4. No rate limiting.** Every chat message is a paid LLM call. The project's
own risk table lists *"API cost overruns at scale"* with the mitigation
*"per-user usage caps"* — which needs users, hence the ordering below.

---

## Step 0 — Stop logging report content *(do this first)*

**What:** decouple SQL echo from `DEBUG`, and stop printing patient text.

| Action | File |
|---|---|
| MODIFY | `backend/app/config.py` — separate `SQL_ECHO` flag, default false |
| MODIFY | `backend/app/core/database.py` — `echo=settings.SQL_ECHO` |
| MODIFY | `docker-compose.yml` — `SQL_ECHO=${SQL_ECHO:-false}` |

**Why separate from DEBUG:** `DEBUG` also controls the docs endpoint and error
verbosity, both of which you want on in development. Query echo is the one that
writes clinical text to disk, and it should be opt-in on its own.

**Deliverable:** `docker-compose logs backend` no longer contains report text.

**Estimate:** ~20 minutes. Smallest change in the phase, largest immediate
reduction in exposure.

---

## Step 1 — Users and password storage

**What:** a `users` table, a seed command, and login issuing a JWT.

| Action | File |
|---|---|
| NEW | `backend/app/models/user.py` |
| NEW | `backend/alembic/versions/0005_users.py` |
| NEW | `backend/app/core/security.py` — hashing, token issue/verify |
| NEW | `backend/app/schemas/auth.py` |
| NEW | `backend/app/api/v1/endpoints/auth.py` — **login and me only** |
| NEW | `backend/scripts/create_user.py` — operator-run account creation |

> [!IMPORTANT]
> **NO SELF-SERVICE REGISTRATION.** Decided once deployment was confirmed.
>
> A public `/auth/register` on a clinical tool means anyone who finds the URL
> can create an account and read uploaded patient reports. That is arguably
> worse than no authentication at all, because the system now *looks*
> protected — the login screen is doing reassurance rather than access control.
>
> Accounts are created by whoever runs the deployment:
>
>     docker-compose exec backend python scripts/create_user.py \
>         --email radiologist@hospital.org
>
> The pilot has known users by name. A registration form solves a problem this
> deployment does not have, and creates one it would otherwise not have.

**bcrypt via passlib, never a hand-rolled hash.** Cost factor left at the
library default so it rises with hardware.

**The secret is not optional and has no default.** A `JWT_SECRET` with a
fallback value is worse than none: it works in development, ships to
production, and every deployment shares a signing key. The app should refuse to
start without one, loudly, the same way it already refuses to start without an
embedding model.

**Tokens carry an expiry and are checked against it.** Short-lived access token;
no refresh token in this phase — a pilot with one radiologist can log in again.

**Deliverable:** register, log in, receive a token, call `/auth/me` with it.

**Estimate:** ~3 hours.

---

## Step 2 — Protect the routes

**What:** a dependency that rejects unauthenticated requests, applied
deliberately per route.

| Action | File |
|---|---|
| NEW | `backend/app/core/deps.py` — `get_current_user` |
| MODIFY | every router — add the dependency |

**Default deny, with an explicit public list.** Public: `/health`, `/auth/login`,
`/auth/register`. Everything else requires a token. Adding routes to a public
allowlist is a decision someone makes; forgetting to protect a new route is an
accident, and the second failure mode is the one that actually happens.

> [!IMPORTANT]
> **`/images/{id}/file` matters most here.** It serves the actual uploaded
> image. A UUID is not an access control — it leaks through logs, browser
> history, referrer headers and screenshots.

**Deliverable:** every clinical endpoint returns 401 without a valid token.

**Estimate:** ~2 hours.

---

## Step 3 — Ownership and the audit trail

**What:** attach a real user to reports and images, and complete the sign-off
record.

| Action | File |
|---|---|
| NEW | `backend/alembic/versions/0006_ownership.py` — nullable `user_id` |
| MODIFY | `backend/app/models/report.py`, `image.py` |
| MODIFY | `backend/app/api/v1/endpoints/reports.py`, `images.py` — filter by owner |

**`reviewed_by` becomes real.** It is currently free text with a comment saying
*"until Phase 6 adds auth"*. Sign-off recorded against a self-declared name is
not an audit trail; recorded against an authenticated user, it is.

**Nullable, and existing rows stay unowned.** Backfilling every pre-auth report
to whoever registers first would fabricate an attribution — the exact thing an
audit trail exists to prevent. Rows created before authentication existed have
no known author, and should say so.

**Deliverable:** a user sees only their own reports; approvals record who.

**Estimate:** ~2 hours.

---

## Step 4 — Rate limiting and cost control

**What:** per-user limits on the endpoints that cost money.

| Action | File |
|---|---|
| MODIFY | `backend/requirements.txt` — `slowapi` |
| NEW | `backend/app/core/limits.py` |
| MODIFY | `chat.py`, `images.py`, `knowledge.py` |

Limits by cost, not uniformly: chat and image upload invoke a model; listing
reports does not. `fetch-pmc` and `fetch-figures` are also NCBI-facing and
already rate-limited on their side — a per-user cap here stops one user
triggering a ban that affects everyone.

**Deliverable:** the 21st chat message in a minute returns 429 with a
`Retry-After`.

**Estimate:** ~1.5 hours.

---

## Step 5 — Error handling and request tracing

**What:** stop leaking exception text; make an incident traceable.

| Action | File |
|---|---|
| NEW | `backend/app/core/errors.py` — exception handlers |
| NEW | `backend/app/core/logging.py` — request ID middleware |
| MODIFY | `main.py`, and every `detail=f"Internal error: {e}"` |

Clients get a reference; the log gets the detail:

```
500  {"detail": "Internal error", "request_id": "b7f3a2"}
log  b7f3a2  ValueError: ...  <full traceback>
```

Same information reaches the person debugging, none of it reaches the caller.

**Deliverable:** no traceback text in any HTTP response; every response carries
`X-Request-ID`.

**Estimate:** ~2 hours.

---

## Step 6 — Login in the frontend

| Action | File |
|---|---|
| NEW | `frontend/src/app/login/page.tsx` |
| NEW | `frontend/src/lib/auth.ts` |
| MODIFY | `frontend/src/lib/api.ts` — attach the token, handle 401 |
| MODIFY | `frontend/src/app/page.tsx` — redirect when unauthenticated |

**Token in memory plus `sessionStorage`, not `localStorage`.** A token in
`localStorage` survives browser restarts on a shared clinical workstation, which
is exactly where it should not survive.

**Deliverable:** unauthenticated users land on a login screen; a 401 mid-session
returns them there rather than failing silently.

**Estimate:** ~3 hours.

---

## Step 7 — Tests and verification

| Action | File | Covers |
|---|---|---|
| NEW | `backend/tests/test_auth.py` | hashing, expiry, tampered and absent tokens |
| NEW | `backend/tests/test_authz.py` | every route requires auth; the public list is exactly three |
| MODIFY | `backend/tests/*` | existing tests need an authenticated client |
| NEW | `verify_phase6.sh` | end-to-end, in the style of `verify_phase4.sh` |

**The test I care most about:** enumerate every registered route and assert each
one either requires authentication or appears in the public allowlist. A new
endpoint added later fails that test by default, which is the only way this
stays true after the phase ends.

**Estimate:** ~3 hours.

---

## Summary

| Step | What | Est. |
|---|---|---|
| 0 | Stop logging report content | 0.3h |
| 1 | Users, password hashing, JWT | 3h |
| 2 | Protect the routes | 2h |
| 3 | Ownership and audit trail | 2h |
| 4 | Rate limiting | 1.5h |
| 5 | Error handling, request IDs | 2h |
| 6 | Frontend login | 3h |
| 7 | Tests and verification | 3h |
| | **Total** | **~17h** |

### Risks

| Risk | Mitigation |
|---|---|
| Auth breaks the 455 existing tests | An authenticated test client fixture, added in Step 1 and used from then on |
| A route is added later without protection | The route-enumeration test in Step 7 fails by default |
| `JWT_SECRET` ships with a default | No default; the app refuses to start without it |
| Ownership backfill fabricates attribution | Nullable; pre-auth rows stay unowned and say so |
| Scope creep into deployment | Managed Postgres, HTTPS and hosting are Phase 7 |

### What this phase does NOT make the system

Authentication is not clinical safety certification, and none of this changes
the standing constraint: **synthetic or anonymised data only.** Phase 6 makes
the system defensible as a pilot tool. It does not make it lawful to put real
patient data into it, which needs a legal basis, a DPA, encryption at rest, and
a retention policy — none of which are in this plan or the project's scope.

---

## Progress

```
Phase 6  Step 0  Stop logging report content ... ✅ SQL_ECHO, no longer tied to DEBUG
         Step 1  Users, hashing, JWT ........... ✅ security.py, users table, 0005
         Step 2  Protect the routes ............ ✅ router.py default-deny
         Step 3  Ownership and audit trail ..... ✅ 0006, _owned/_visible_to, 404-not-403
         Step 4  Rate limiting ................. ✅ limits.py, per-user and per-IP
         Step 5  Error handling, request IDs ... ✅ errors.py, X-Request-ID
         Step 6  Frontend login ................ ✅ /login, signin + signup
         Step 7  Tests and verification ........ ✅ verify_phase6.sh
```

### One change from the plan as written

The plan assumed accounts would be created by an operator running
`scripts/create_user.py`. That was rejected: an app nobody but its author can
sign in to is not an app. Self-service registration was added instead —
`POST /api/v1/auth/register`, rate limited to 5/hour per IP, and gated by
`ALLOW_REGISTRATION` so a clinical deployment can still close it.

⚠️  **Registration is only defensible because of Step 3.** Ownership is what
makes a new account land in an empty workspace rather than in everyone's
reports. If `tests/test_ownership.py` is ever weakened, `ALLOW_REGISTRATION`
has to go false in the same commit.

### Known gaps carried forward

- No password reset. A reset flow is another credential path needing the same
  care as the password itself; for a pilot, recovery is the operator
  recreating the account.
- No refresh tokens. A 24h access token expires and you sign in again.
- Rate limit state is in-process, so it resets on restart and does not hold
  across replicas. Correct fix at Phase 7 is Redis.
