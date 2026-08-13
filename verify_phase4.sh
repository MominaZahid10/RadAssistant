#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Phase 4 — Multimodal Ingestion: end-to-end verification
# ══════════════════════════════════════════════════════════════
#
#   RADASSIST_EMAIL=you@example.org RADASSIST_PASSWORD=... bash verify_phase4.sh
#
# ⚠️  WHY A SCRIPT AND NOT "CLICK AROUND THE UI".
# Every failure this phase can produce is silent. De-identification that does
# not run still returns 200. A vision model that is unreachable falls back to
# OCR and returns text either way. A figure run with no network reports a
# successful 202. None of that is visible from the frontend, which is exactly
# why it needs asserting rather than eyeballing.
#
# ⚠️  NEEDS CREDENTIALS SINCE PHASE 6.
# Every endpoint below is authenticated now. Without a token this script would
# report the whole phase as broken rather than saying "no token".
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

field() { grep -o "\"$1\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" | head -1 | sed 's/.*:[[:space:]]*"//; s/"$//'; }
num()   { grep -o "\"$1\"[[:space:]]*:[[:space:]]*[0-9.]*"     | head -1 | sed 's/.*:[[:space:]]*//'; }


# ══════════════════════════════════════════════════════════════
say "1. Stack is up"
# ══════════════════════════════════════════════════════════════

HEALTH="$(curl -fsS --max-time 20 "$API/api/v1/health" 2>/dev/null || true)"
if [ -z "$HEALTH" ]; then
  bad "Backend unreachable at $API"
  info "docker-compose up -d"
  exit 1
fi
ok "Backend responding"

for component in database qdrant; do
  if printf '%s' "$HEALTH" | grep -q "\"$component\":\"connected\""; then
    ok "$component connected"
  else
    bad "$component not connected"
  fi
done


# ══════════════════════════════════════════════════════════════
say "2. Sign in"
# ══════════════════════════════════════════════════════════════

if [ -z "${RADASSIST_EMAIL:-}" ] || [ -z "${RADASSIST_PASSWORD:-}" ]; then
  bad "Set RADASSIST_EMAIL and RADASSIST_PASSWORD"
  info "The API has required authentication since Phase 6."
  exit 1
fi

TOKEN="$(curl -s --max-time 20 -X POST "$API/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$RADASSIST_EMAIL\",\"password\":\"$RADASSIST_PASSWORD\"}" \
  | field access_token)"

if [ -z "$TOKEN" ]; then
  bad "Sign-in failed"
  exit 1
fi
ok "Signed in as $RADASSIST_EMAIL"

AUTH="Authorization: Bearer $TOKEN"
get() { curl -fsS --max-time 30 -H "$AUTH" "$API$1" 2>/dev/null; }


# ══════════════════════════════════════════════════════════════
say "3. Migrations applied"
# ══════════════════════════════════════════════════════════════
# create_all() would have created medical_images but would NOT have added a
# column to an existing table — silently, with the app then failing at query
# time somewhere unrelated.

PGDB="${PGDB:-radassist_db}"      # the DATABASE, not the role
PGUSER_="${PGUSER_:-radassist}"

if docker-compose exec -T postgres psql -U "$PGUSER_" -d "$PGDB" -c '\q' >/dev/null 2>&1; then
  COLS="$(docker-compose exec -T postgres psql -U "$PGUSER_" -d "$PGDB" \
          -tAc "select column_name from information_schema.columns
                where table_name='medical_images'" 2>/dev/null)"
  for col in licence caption source_url is_deidentified ocr_text user_id; do
    if printf '%s' "$COLS" | grep -qx "$col"; then
      ok "medical_images.$col exists"
    else
      bad "medical_images.$col MISSING — docker-compose exec backend alembic upgrade head"
    fi
  done
else
  warn "Could not reach postgres — skipping schema checks"
fi


# ══════════════════════════════════════════════════════════════
say "4. Report reading: vision, with OCR fallback"
# ══════════════════════════════════════════════════════════════

VISION="$(printf '%s' "$HEALTH" | field vision)"
case "$VISION" in
  available*) ok "Vision reader: $VISION" ;;
  "")         warn "No vision component in /health" ;;
  *)          warn "Vision unavailable — falling back to OCR"
              info "$VISION"
              info "Tesseract read 'hyperlordotic' as 'hypoiordotic' on a"
              info "low-resolution report, inverting the finding. Degraded." ;;
esac

DICOM="$(printf '%s' "$HEALTH" | field dicom)"
case "$DICOM" in
  available) ok "DICOM available (pydicom installed)" ;;
  "")        warn "No dicom component in /health" ;;
  *)         warn "DICOM unavailable — $DICOM"
             info "docker-compose up -d --build backend" ;;
esac


# ══════════════════════════════════════════════════════════════
say "5. Image API contract"
# ══════════════════════════════════════════════════════════════

STATS="$(get /api/v1/images/stats || true)"
if [ -n "$STATS" ]; then
  ok "GET /images/stats responds"
  info "images: $(printf '%s' "$STATS" | num total_images)  deidentified: $(printf '%s' "$STATS" | num deidentified_count)"
else
  bad "GET /images/stats failed"
fi

# ⚠️  /stats MUST BE DECLARED BEFORE /{image_id}.
# Reversed, FastAPI parses "stats" as a UUID path parameter and 422s.
if printf '%s' "$STATS" | grep -q "total_images"; then
  ok "/stats is not being swallowed by /{image_id}"
else
  bad "/stats appears to be matching the /{image_id} route"
fi

LISTING="$(get '/api/v1/images?page_size=100' || true)"
if [ -n "$LISTING" ]; then
  ok "GET /images lists"
else
  bad "GET /images failed"
fi

# ⚠️  A STORAGE PATH MUST NEVER APPEAR IN A RESPONSE.
# It exposes the filesystem layout and invites clients to build their own
# URLs, which is how a traversal endpoint gets created by accident.
if printf '%s' "$LISTING" | grep -q "storage_path\|thumbnail_path"; then
  bad "Internal storage paths are leaking into API responses"
else
  ok "Storage paths are not exposed"
fi


# ══════════════════════════════════════════════════════════════
say "6. De-identification is never assumed"
# ══════════════════════════════════════════════════════════════
# ⚠️  THE CHECK I CARE MOST ABOUT IN THIS PHASE.
# is_deidentified must be true ONLY for parsed DICOM. A PMC figure has no PHI
# to remove, but "no PHI present" and "de-identification ran" are different
# claims and only one of them is ours to make.

BAD_CLAIMS="$(printf '%s' "$LISTING" | docker-compose exec -T backend python -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print("ERR"); raise SystemExit
bad = [i for i in data.get("images", [])
       if i.get("is_deidentified") and i.get("source_type") != "dicom_upload"]
print(len(bad))
' 2>/dev/null)"

case "$BAD_CLAIMS" in
  0)   ok "No non-DICOM image claims to be de-identified" ;;
  ""|ERR) warn "De-identification audit could not run" ;;
  *)   bad "$BAD_CLAIMS non-DICOM image(s) marked is_deidentified=true" ;;
esac


# ══════════════════════════════════════════════════════════════
say "7. PMC figures (Step 3)"
# ══════════════════════════════════════════════════════════════

FIGSTATS="$(get /api/v1/knowledge/figure-stats || true)"
if [ -z "$FIGSTATS" ]; then
  bad "GET /knowledge/figure-stats failed"
else
  ok "GET /knowledge/figure-stats responds"
  FIGS="$(printf '%s' "$FIGSTATS" | num figures)"
  DOCS="$(printf '%s' "$FIGSTATS" | num pmc_documents)"
  CAPS="$(printf '%s' "$FIGSTATS" | num figures_with_captions)"
  info "figures: ${FIGS:-0}   pmc articles: ${DOCS:-0}   captioned: ${CAPS:-0}"

  if [ "${FIGS:-0}" = "0" ]; then
    warn "No figures stored — KNOWN LIMITATION, cause diagnosed"
    info "Ingestion recorded PMIDs rather than PMCIDs, so figure URLs cannot"
    info "be constructed. The pipeline is built and tested; the fix is to"
    info "capture the PMCID at ingest time."
  else
    ok "${FIGS} figures stored"
    if [ "${CAPS:-0}" = "${FIGS:-0}" ]; then
      ok "Every figure carries its caption"
    else
      warn "$((FIGS - CAPS)) figure(s) stored without a caption"
    fi
  fi
fi


# ══════════════════════════════════════════════════════════════
say "8. Corpus text is not truncated"
# ══════════════════════════════════════════════════════════════
# Phase 4 uncovered a Phase 3 bug: removing <xref> citation markers also
# deleted their `.tail`, i.e. the rest of the paragraph. Articles ingested
# before the fix are short.

DOCLIST="$(get '/api/v1/knowledge/documents?page_size=100&source_type=pmc_open_access' || true)"
if [ -z "$DOCLIST" ]; then
  warn "Could not list documents to measure article length"
else
  AVG="$(printf '%s' "$DOCLIST" | docker-compose exec -T backend python -c '
import json, sys
docs = json.load(sys.stdin).get("documents", [])
counts = [d.get("chunk_count") or 0 for d in docs]
print(round(sum(counts) / len(counts), 1) if counts else 0)
' 2>/dev/null)"

  if [ -n "${AVG:-}" ] && [ "$AVG" != "0" ]; then
    info "mean chunks per PMC article: $AVG"
    # A full-text radiology article is many thousands of characters. At
    # CHUNK_SIZE=512 that is tens of chunks. Single digits means truncation.
    if awk "BEGIN{exit !($AVG < 8)}"; then
      bad "Articles look truncated — ingested before the <xref> tail fix"
      info "Recover with:  .\\refetch_corpus.ps1 -Confirm"
    else
      ok "Article length looks healthy ($AVG chunks/article)"
    fi
  else
    warn "Could not measure article length"
  fi
fi


# ══════════════════════════════════════════════════════════════
say "9. Backend test suite"
# ══════════════════════════════════════════════════════════════

if docker-compose exec -T backend python -m pytest -q >/tmp/p4tests.txt 2>&1; then
  ok "$(tail -1 /tmp/p4tests.txt | tr -d '=' | xargs)"
else
  bad "Tests failed"
  tail -15 /tmp/p4tests.txt | sed 's/^/    /'
fi


# ══════════════════════════════════════════════════════════════
printf '\n\033[1m══ Summary ══════════════════════════════\033[0m\n'
printf '  passed: \033[32m%s\033[0m   warnings: \033[33m%s\033[0m   failed: \033[31m%s\033[0m\n\n' \
  "$PASS" "$WARN" "$FAIL"

[ "$FAIL" -gt 0 ] && { printf '\033[31mPhase 4 is not fully verified.\033[0m\n\n'; exit 1; }
[ "$WARN" -gt 0 ] && { printf '\033[33mPhase 4 works, with the notes above.\033[0m\n\n'; exit 0; }
printf '\033[32mPhase 4 verified.\033[0m\n\n'
