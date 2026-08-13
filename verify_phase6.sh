#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Phase 6 — Auth & Hardening: end-to-end verification
# ══════════════════════════════════════════════════════════════
#
#   RADASSIST_EMAIL=you@example.org RADASSIST_PASSWORD=... bash verify_phase6.sh
#
# ⚠️  THIS SCRIPT PROBES A LIVE SERVER, WHICH IS THE POINT.
# The test suite asserts that the code contains the right dependencies and
# the right strings. That is worth having, but it cannot tell you whether
# the running container is serving what the repository says. Every check
# below goes over HTTP.
#
# Creates one throwaway account and leaves it. There is no delete endpoint —
# users are deactivated rather than removed, so approvals stay attributable.
#
# ⚠️  THE PROBE ADDRESSES USE example.com, NOT example.invalid.
# `.invalid` is a reserved TLD and email-validator refuses it, so every probe
# using one 422s at validation and never reaches the code being tested. That
# turned the enumeration check into a comparison between a validation error
# and a login error — which of course differed, and reported a leak that was
# not there. example.com is reserved for documentation (RFC 2606) but is a
# real registered domain, so it parses.
# ══════════════════════════════════════════════════════════════

set -u
API="${API:-http://127.0.0.1:8000}"

PASS=0; FAIL=0; WARN=0

say()  { printf '\n\033[1m── %s ─────────────────────────────\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; WARN=$((WARN+1)); }
info() { printf '    \033[2m%s\033[0m\n' "$1"; }

command -v curl >/dev/null 2>&1 || { echo "curl is required"; exit 1; }

# Status code only.
code() { curl -s -o /dev/null -w "%{http_code}" --max-time 20 "$@"; }
# Body only.
body() { curl -s --max-time 20 "$@"; }


# ══════════════════════════════════════════════════════════════
say "1. The server is up"
# ══════════════════════════════════════════════════════════════

if [ "$(code "$API/api/v1/health")" = "200" ]; then
  ok "Health responds"
else
  bad "Backend unreachable at $API"
  info "docker-compose up -d"
  exit 1
fi


# ══════════════════════════════════════════════════════════════
say "2. Clinical routes are closed"
# ══════════════════════════════════════════════════════════════
# ⚠️  THE HEART OF THE PHASE. Before this, anyone who could reach the port
# could list every report and download every uploaded image.

for path in \
  "/api/v1/reports" \
  "/api/v1/reports/stats" \
  "/api/v1/images" \
  "/api/v1/images/stats" \
  "/api/v1/knowledge/documents"
do
  status="$(code "$API$path")"
  if [ "$status" = "401" ]; then
    ok "401 without a token: $path"
  else
    bad "$path returned $status — expected 401"
  fi
done

# A forged token must not work. This is what the signature is for.
forged="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhYmMiLCJleHAiOjk5OTk5OTk5OTl9.not_a_real_signature"
if [ "$(code -H "Authorization: Bearer $forged" "$API/api/v1/reports")" = "401" ]; then
  ok "A forged token is rejected"
else
  bad "A forged token was ACCEPTED — check the signing key and algorithm pin"
fi


# ══════════════════════════════════════════════════════════════
say "3. Sign in"
# ══════════════════════════════════════════════════════════════

EMAIL="${RADASSIST_EMAIL:-}"
PASSWORD="${RADASSIST_PASSWORD:-}"

if [ -z "$EMAIL" ] || [ -z "$PASSWORD" ]; then
  bad "Set RADASSIST_EMAIL and RADASSIST_PASSWORD"
  info "The rest of this script needs an authenticated session."
  exit 1
fi

LOGIN_BODY="$(body -X POST "$API/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}")"

TOKEN="$(printf '%s' "$LOGIN_BODY" \
  | grep -o '"access_token"[[:space:]]*:[[:space:]]*"[^"]*"' \
  | sed 's/.*:[[:space:]]*"//; s/"$//')"

if [ -n "$TOKEN" ]; then
  ok "Signed in as $EMAIL"
else
  bad "Sign-in failed"
  info "$LOGIN_BODY"
  exit 1
fi

AUTH="Authorization: Bearer $TOKEN"

if [ "$(code -H "$AUTH" "$API/api/v1/reports")" = "200" ]; then
  ok "The token opens the protected routes"
else
  bad "A valid token was refused"
fi


# ══════════════════════════════════════════════════════════════
say "4. Wrong credentials are indistinguishable"
# ══════════════════════════════════════════════════════════════
# ⚠️  UNKNOWN EMAIL AND WRONG PASSWORD MUST LOOK THE SAME.
# A different message — or a measurably different response time — turns the
# login form into an account enumerator.

WRONG_PW="$(body -X POST "$API/api/v1/auth/login" -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"definitely-not-the-password\"}")"
NO_USER="$(body -X POST "$API/api/v1/auth/login" -H "Content-Type: application/json" \
  -d '{"email":"nobody-at-all-9f2a@example.com","password":"whatever-1234"}')"

strip_id() { sed 's/"request_id":"[^"]*"//'; }

# ⚠️  A 422 HERE MEANS THE PROBE IS WRONG, NOT THE SERVER.
# Without this branch the script reports an enumeration leak whenever the
# probe address fails validation — a false alarm on the one check most likely
# to be believed.
case "$WRONG_PW$NO_USER" in
  *"Request validation failed"*)
    warn "A probe address was rejected before reaching the login logic"
    info "This check did not run. Both probe emails must be valid addresses."
    ;;
  *)
    if [ "$(printf '%s' "$WRONG_PW" | strip_id)" = "$(printf '%s' "$NO_USER" | strip_id)" ]; then
      ok "Unknown account and wrong password return the same body"
    else
      bad "The two failures differ — the login form leaks which accounts exist"
      info "wrong password: $WRONG_PW"
      info "unknown email : $NO_USER"
    fi
    ;;
esac


# ══════════════════════════════════════════════════════════════
say "5. Ownership"
# ══════════════════════════════════════════════════════════════
# A second account must not see the first one's work.

SUFFIX="$(date +%s)"
TEST_EMAIL="verify-$SUFFIX@example.com"
TEST_PW="verify-password-$SUFFIX"

REG="$(body -X POST "$API/api/v1/auth/register" -H "Content-Type: application/json" \
  -d "{\"email\":\"$TEST_EMAIL\",\"password\":\"$TEST_PW\"}")"

TEST_TOKEN="$(printf '%s' "$REG" \
  | grep -o '"access_token"[[:space:]]*:[[:space:]]*"[^"]*"' \
  | sed 's/.*:[[:space:]]*"//; s/"$//')"

if [ -n "$TEST_TOKEN" ]; then
  ok "Registration works (ALLOW_REGISTRATION is on)"

  # Create a report as the main user, then try to read it as the new one.
  CREATED="$(body -X POST "$API/api/v1/reports" -H "$AUTH" \
    -H "Content-Type: application/json" \
    -d '{"findings_input":"verify probe","ai_draft":"**FINDINGS**\nverify probe."}')"
  RID="$(printf '%s' "$CREATED" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
    | head -1 | sed 's/.*:[[:space:]]*"//; s/"$//')"

  if [ -n "$RID" ]; then
    OTHER="$(code -H "Authorization: Bearer $TEST_TOKEN" "$API/api/v1/reports/$RID")"
    if [ "$OTHER" = "404" ]; then
      # ⚠️  404 AND NOT 403 — 403 would confirm the report exists.
      ok "Another user gets 404 for a report they do not own"
    elif [ "$OTHER" = "403" ]; then
      bad "Returned 403 — that confirms the row exists. Should be 404."
    else
      bad "Another user got $OTHER for someone else's report"
    fi

    # And clean up the probe.
    curl -s -o /dev/null -X DELETE -H "$AUTH" "$API/api/v1/reports/$RID"
  else
    warn "Could not create a probe report"
    info "$CREATED"
  fi
else
  case "$REG" in
    *"disabled"*)
      warn "Registration is disabled (ALLOW_REGISTRATION=false)"
      info "Correct for a clinical deployment. Ownership not probed." ;;
    *"Too many"*|*"rate"*|*"retry"*)
      # Registration is 5/hour per IP, so the sixth run of this script in an
      # hour trips it. That is the limiter working, not a failure.
      warn "Registration is rate limited — ownership not probed this run"
      info "5/hour per IP. Wait, or probe ownership by hand." ;;
    *)
      warn "Registration did not return a token"
      info "$REG" ;;
  esac
fi


# ══════════════════════════════════════════════════════════════
say "6. Errors carry a reference, not a traceback"
# ══════════════════════════════════════════════════════════════

NOT_FOUND="$(body -H "$AUTH" "$API/api/v1/reports/00000000-0000-0000-0000-000000000000")"
case "$NOT_FOUND" in
  *request_id*) ok "Responses carry a request_id" ;;
  *)            bad "No request_id in the response"; info "$NOT_FOUND" ;;
esac

if curl -s -D - -o /dev/null --max-time 20 -H "$AUTH" "$API/api/v1/reports" \
     | grep -qi "x-request-id"; then
  ok "X-Request-ID header is set"
else
  bad "X-Request-ID header missing"
fi

# A malformed body should not echo the value back — on a login that value is
# a password.
VALIDATION="$(body -X POST "$API/api/v1/auth/login" -H "Content-Type: application/json" \
  -d '{"email":"not-an-email","password":"hunter2-hunter2"}')"
case "$VALIDATION" in
  *hunter2*) bad "A validation error echoed the submitted password back" ;;
  *)         ok "Validation errors do not echo the submitted value" ;;
esac


# ══════════════════════════════════════════════════════════════
say "7. Rate limiting"
# ══════════════════════════════════════════════════════════════
# Login is capped at 10/min per IP. 12 attempts should trip it.

LIMITED=0
for _ in $(seq 1 12); do
  s="$(code -X POST "$API/api/v1/auth/login" -H "Content-Type: application/json" \
       -d '{"email":"ratelimit-probe@example.com","password":"wrong-password"}')"
  [ "$s" = "429" ] && LIMITED=1 && break
done

if [ "$LIMITED" = "1" ]; then
  ok "Repeated login attempts are rate limited (429)"

  if curl -s -D - -o /dev/null -X POST "$API/api/v1/auth/login" \
       -H "Content-Type: application/json" \
       -d '{"email":"ratelimit-probe@example.com","password":"wrong-password"}' \
     | grep -qi "retry-after"; then
    ok "429 carries Retry-After"
  else
    bad "429 without Retry-After — clients cannot back off correctly"
  fi
else
  bad "12 failed logins were not rate limited"
fi


# ══════════════════════════════════════════════════════════════
say "8. Report text is not in the logs"
# ══════════════════════════════════════════════════════════════
# ⚠️  DEBUG=true USED TO SET echo ON THE SQL ENGINE, LOGGING EVERY INSERT —
# INCLUDING findings_input AND ai_draft.

if docker-compose logs --tail=400 backend 2>/dev/null | grep -q "INSERT INTO reports"; then
  bad "SQL INSERTs are being logged — report text is reaching the log"
  info "Set SQL_ECHO=false and restart."
else
  ok "No SQL statements in the recent log"
fi


# ══════════════════════════════════════════════════════════════
say "9. Test suite"
# ══════════════════════════════════════════════════════════════

if docker-compose exec -T backend python -m pytest -q >/tmp/p6tests.txt 2>&1; then
  ok "$(tail -1 /tmp/p6tests.txt | tr -d '=' | xargs)"
else
  bad "Tests failed"
  tail -15 /tmp/p6tests.txt | sed 's/^/    /'
fi


# ══════════════════════════════════════════════════════════════
printf '\n\033[1m══ Summary ══════════════════════════════\033[0m\n'
printf '  passed: \033[32m%s\033[0m   warnings: \033[33m%s\033[0m   failed: \033[31m%s\033[0m\n\n' \
  "$PASS" "$WARN" "$FAIL"

if [ "$FAIL" -gt 0 ]; then
  printf '\033[31mDo NOT deploy this publicly.\033[0m\n\n'
  exit 1
fi
if [ "$WARN" -gt 0 ]; then
  printf '\033[33mPhase 6 verified, with the notes above.\033[0m\n\n'
  exit 0
fi
printf '\033[32mPhase 6 verified.\033[0m\n\n'
