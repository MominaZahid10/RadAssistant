#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Phase 3.5 — A/B: vector-only vs cross-encoder reranking
# ══════════════════════════════════════════════════════════════
#
#   bash measure_rerank.sh
#
# Runs the evaluation twice against the SAME code path — reranking off, then
# on — and prints the delta. ~3 minutes, no LLM calls, costs nothing.
# ══════════════════════════════════════════════════════════════

set -u
API="${API:-http://localhost:8000}"
CACHE="backend/.hf_cache/hub"

say()  { printf '\n\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '\033[32m%s\033[0m\n' "$1"; }
warn() { printf '\033[33m%s\033[0m\n' "$1"; }
bad()  { printf '\033[31m%s\033[0m\n' "$1"; }

# ── Find a working Python ────────────────────────────────────
# Git Bash on Windows often can't see the `python` that PowerShell uses —
# it may be `py`, `python3`, or a versioned .exe on a different PATH.
find_python() {
  for c in python3 python py "/c/Python313/python.exe" "/c/Python312/python.exe"; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import httpx, yaml" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}

PY=$(find_python) || {
  bad "No Python with httpx + pyyaml found on this shell's PATH."
  echo
  echo "  Git Bash often can't see the interpreter PowerShell uses."
  echo "  Either install the deps for whichever Python IS visible:"
  echo "      python -m pip install httpx pyyaml"
  echo "  or just run the two halves from PowerShell instead:"
  echo
  echo "      \$env:RERANK_ENABLED='false'; docker-compose up -d backend; Start-Sleep 20"
  echo "      cd backend; python eval/run_eval.py --save vector_only; cd .."
  echo "      \$env:RERANK_ENABLED='true';  docker-compose up -d backend; Start-Sleep 20"
  echo "      cd backend; python eval/run_eval.py --vs vector_only --save reranked"
  echo
  exit 1
}
ok "Using Python: $PY"

# ── Model present? ───────────────────────────────────────────
say "Checking for the reranker model"
if ls "$CACHE" 2>/dev/null | grep -qi "ms-marco-MiniLM"; then
  ok "  found in $CACHE"
else
  bad "  cross-encoder/ms-marco-MiniLM-L-6-v2 is NOT cached."
  echo "  Both runs would use vector ordering and the delta would be zero."
  echo "  Fetch it once on a working connection:"
  echo "      HF_HUB_OFFLINE=0 docker-compose up -d --force-recreate backend"
  echo "      cd backend && $PY eval/run_eval.py     # triggers the download"
  exit 1
fi

# ── Wait for the backend, don't guess ────────────────────────
# A fixed `sleep` was wrong: recreating the container after an env change
# takes longer than the restart it replaced, so the health probe fired while
# the app was still booting and reported an empty string.
wait_for_backend() {
  local want="$1" tries=0
  while [ $tries -lt 40 ]; do
    state=$(curl -s --max-time 5 "$API/api/v1/health" 2>/dev/null | grep -o '"reranker":"[^"]*"')
    if [ -n "$state" ]; then
      case "$state" in
        *"$want"*) echo "  $state"; return 0 ;;
      esac
    fi
    tries=$((tries+1)); sleep 2
  done
  warn "  timed out waiting for reranker state to contain '$want'"
  [ -n "${state:-}" ] && echo "  last seen: $state"
  return 1
}

# ── 1/2 Baseline ─────────────────────────────────────────────
say "1/2  Vector-only baseline (RERANK_ENABLED=false)"
RERANK_ENABLED=false docker-compose up -d backend >/dev/null 2>&1
wait_for_backend "disabled" && ok "  confirmed off" || warn "  toggle may not have applied"

(cd backend && "$PY" eval/run_eval.py --save vector_only) || exit 1

# ── 2/2 Reranked ─────────────────────────────────────────────
say "2/2  With cross-encoder reranking (RERANK_ENABLED=true)"
RERANK_ENABLED=true docker-compose up -d backend >/dev/null 2>&1
wait_for_backend "not yet loaded" >/dev/null || true

# Warm the lazy load so question 1 isn't penalised by model init.
printf '  warming reranker...'
curl -s --max-time 120 -X POST "$API/api/v1/knowledge/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"pneumothorax","limit":3}' >/dev/null
printf ' done\n'

state=$(curl -s "$API/api/v1/health" | grep -o '"reranker":"[^"]*"')
echo "  $state"
case "$state" in
  *"loaded:"*)     ok "  confirmed active" ;;
  *unavailable*)   bad "  model failed to load — this run equals the baseline"; exit 1 ;;
  *)               warn "  unexpected state" ;;
esac

(cd backend && "$PY" eval/run_eval.py --vs vector_only --save reranked)

say "Done"
echo "  backend/eval/results/{vector_only,reranked}.json"
echo
echo "  Reranking should raise recall and raise latency."
echo "  If latency is over ~2s, lower RERANK_CANDIDATES in config.py"
echo "  and re-run — the harness makes that a measured trade."
