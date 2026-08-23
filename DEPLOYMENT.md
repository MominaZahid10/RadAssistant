# Deployment

RadAssistant ships as a **Docker Compose stack**: API, Postgres and Qdrant on
one machine, with no dependency on any external service beyond the LLM
provider. That is the deployment that matters — it is what lets the system run
inside a hospital network, where patient data legally cannot leave the
perimeter.

There are two shapes, and the repository supports both:

| | Frontend | API + DBs | Use when |
|---|---|---|---|
| **Split** (default) | Vercel | your server | public demo — free, fast, clean URL |
| **Self-hosted** | your server | your server | on-premise, air-gapped, no external CDN |

The difference is one line in `.env` and one compose flag. Nothing in the
application changes.

Reference deployment below: a Hetzner CX22 (2 vCPU / 4 GB, ~€4/mo) on Ubuntu
24.04, HTTPS via `sslip.io`, registration closed, one published demo account.

Everything runs **on the server** unless it says otherwise.

---

## 0. Before you touch the server

**Set a hard spend cap on your model provider's dashboard.** Two minutes, and
it is the only control a bug in this repository cannot bypass. The in-process
daily ceilings in `app/core/limits.py` are defence in depth; a crash-looping
container resets them on every restart.

Decide two accounts you will create later: one for you, and one **demo**
account whose credentials go in the README.

---

## 1. Provision

Hetzner Cloud → new project → new server:

| Setting | Value |
|---|---|
| Image | Ubuntu 24.04 |
| Type | CX22 (2 vCPU / 4 GB / 40 GB) |
| SSH key | add yours — do not enable password auth |

**Why not the 2 GB CX11.** The API loads MiniLM for embeddings and a
cross-encoder for reranking, alongside Postgres and Qdrant — about 3 GB
resident. On 2 GB the OOM killer takes the backend during model load, and
because a model load failure is *caught rather than fatal*, the app comes up
looking healthy and then retrieves nothing.

Hetzner Cloud Firewall, inbound:

| Port | Source | Why |
|---|---|---|
| 22 | your IP if static, else anywhere | SSH |
| 80 | anywhere | Let's Encrypt HTTP-01 challenge |
| 443 | anywhere | the API |

Nothing else — in particular **not** 5432, 6333, 8000 or 3000.

> Use the *Hetzner Cloud* firewall, not only `ufw`. Docker writes its own
> iptables rules and publishes ports **around** `ufw`, so a host firewall that
> looks correct will not block a published container port. The cloud firewall
> sits outside the machine and has no such hole.

---

## 2. Harden

```bash
ssh root@YOUR_IP

adduser --gecos "" deploy
usermod -aG sudo deploy
rsync --archive --chown=deploy:deploy ~/.ssh /home/deploy/

sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart ssh
```

**Open a second terminal and confirm `ssh deploy@YOUR_IP` works before closing
this one.** If the key copy failed, the session you are in is the only way back
into the machine.

Swap, because 4 GB is tight during the first model load:

```bash
fallocate -l 2G /swapfile && chmod 600 /swapfile
mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

---

## 3. Docker

```bash
curl -fsSL https://get.docker.com | sh
usermod -aG docker deploy
```

Log out and back in as `deploy`, then `docker compose version`.

---

## 4. Clone and configure

```bash
git clone https://github.com/MominaZahid10/RadAssistant.git
cd RadAssistant
git checkout frontend-redesign     # until merged to main
```

Two environment files, **not interchangeable**:

| File | Read by | Contains |
|---|---|---|
| `.env` (repo root) | docker compose | DB credentials, hostnames, which Caddyfile |
| `backend/.env` | the application | API keys, JWT secret, CORS origins |

Compose never passes the root `.env` into the container. An API key placed
there does nothing, silently.

```bash
cp env.prod.example .env
cp backend/.env.example backend/.env
```

### Root `.env`

Your API hostname is the server IP with dots as dashes — `203.0.113.5`
becomes `203-0-113-5.sslip.io`, no DNS record needed.

```
SITE_ADDRESS=203-0-113-5.sslip.io
APP_URL=https://placeholder.vercel.app    # fill in after step 6
CADDYFILE=./Caddyfile
POSTGRES_PASSWORD=<openssl rand -base64 24>
HF_HUB_OFFLINE=0                          # ⚠️  0 for first boot only
```

`HF_HUB_OFFLINE=0` matters because `backend/.hf_cache` is empty on a fresh
server. With offline mode on the models cannot download — and the app starts
anyway, because a load failure is caught. No error; just an assistant that
retrieves nothing.

### `backend/.env`

```
JWT_SECRET=<openssl rand -hex 32>
GROQ_API_KEY=<your key>
OPENAI_API_KEY=<your key>        # fallback: catches Groq rate limits and retired models
LLM_PROVIDER=groq
ALLOW_REGISTRATION=false
NCBI_EMAIL=<a real address>      # placeholders are detected and rejected
CORS_ORIGINS=https://placeholder.vercel.app   # updated in step 6
```

Groq stays primary — free, and it streams fast, which is most of what makes
the chat feel responsive. OpenAI is configured so a Groq rate limit or a
retired model falls through instead of 404ing the demo.

---

## 5. Start the API

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

Five to fifteen minutes. The `frontend` container does **not** start — it sits
behind the `selfhosted` profile, because Vercel is building it instead.

```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f backend
```

`start.sh` runs alembic migrations before uvicorn, so early lines are schema
work. Then turn offline mode back on:

```bash
ls backend/.hf_cache/hub          # confirm models cached
# set HF_HUB_OFFLINE=1 in .env
docker compose -f docker-compose.prod.yml up -d backend
```

Check TLS:

```bash
docker compose -f docker-compose.prod.yml logs caddy | grep -i certificate
curl https://203-0-113-5.sslip.io/api/v1/health
```

If issuance failed: port 80 blocked, or a Let's Encrypt rate limit. For the
rate limit, **swap `sslip.io` for `nip.io`** in `SITE_ADDRESS` — separate
registered domains, separate allowances, and sslip.io's own docs recommend it.

---

## 6. Deploy the frontend to Vercel

From [vercel.com](https://vercel.com) → **Add New Project** → import the
GitHub repo.

| Setting | Value |
|---|---|
| Root Directory | `frontend` ← **this one matters** |
| Framework | Next.js (auto-detected) |
| Branch | `frontend-redesign`, or `main` once merged |

Environment variable:

```
NEXT_PUBLIC_API_URL = https://203-0-113-5.sslip.io
```

**Origin only** — no `/api`, no trailing slash. `lib/api.ts` appends
`/api/v1/...` itself, so a path here yields `/api/api/v1/chat` and every
request 404s.

⚠️ **`NEXT_PUBLIC_*` is compiled into the browser bundle at build time.**
Changing it later requires a **redeploy**, not a restart — the value is
already inside the JavaScript the browser downloads.

Deploy. Vercel gives you a URL. Now close the loop on the server:

```bash
# .env
APP_URL=https://radassistant.vercel.app

# backend/.env
CORS_ORIGINS=https://radassistant.vercel.app

docker compose -f docker-compose.prod.yml up -d backend caddy
```

Without that `CORS_ORIGINS` value the browser blocks every API call and the
app looks broken with a console error most people never open.

> **Preview deployments will fail CORS.** Vercel gives every branch and pull
> request its own hostname, and `CORS_ORIGINS` is an exact-match list. Only
> the production URL works. If you want previews to work too, the backend
> needs `allow_origin_regex` — ask and I'll add it.

### Or: self-hosted instead

Skip Vercel entirely, serve everything from the box:

```bash
# .env
CADDYFILE=./Caddyfile.selfhosted
PUBLIC_API_URL=https://203-0-113-5.sslip.io

docker compose -f docker-compose.prod.yml --profile selfhosted up -d --build
```

This is the configuration to demonstrate for an on-premise audience: one
command, one machine, nothing leaving the network.

---

## 7. Accounts and knowledge base

Registration is off, so accounts come from the script:

```bash
docker compose -f docker-compose.prod.yml exec backend python scripts/create_user.py --help
```

Create yours and the demo account, 12+ character passwords. Then seed the
corpus — these routes need a token:

```bash
docker compose -f docker-compose.prod.yml exec -T backend python - <<'PY'
import json, urllib.request
base = "http://localhost:8000"

req = urllib.request.Request(f"{base}/api/v1/auth/login",
    data=json.dumps({"email": "YOU@example.com", "password": "YOUR_PASSWORD"}).encode(),
    headers={"Content-Type": "application/json"})
token = json.load(urllib.request.urlopen(req))["access_token"]

for path in ("/api/v1/knowledge/seed", "/api/v1/knowledge/fetch-pmc"):
    r = urllib.request.Request(base + path, data=b"",
        headers={"Authorization": f"Bearer {token}"})
    print(path, urllib.request.urlopen(r).status)
PY
```

`fetch-pmc` pulls hundreds of articles at NCBI's rate limit — hours, not
minutes. Run it inside `tmux` so an SSH disconnect does not kill it.

---

## 8. Verify

Open the Vercel URL, sign in as the demo account, ask *"What are the
radiographic findings of a pneumothorax?"* Check three things:

- **tokens stream one at a time.** If the answer appears all at once after a
  pause, something is buffering — confirm `flush_interval -1` is in the
  `Caddyfile` under `handle /api/*`.
- **a sources panel appears.** No sources means retrieval returned nothing:
  the corpus did not seed, or the embedding model did not load.
- **a citation chip scrolls to its passage.**

You can point the E2E suite at the deployment from your laptop:

```powershell
$env:E2E_BASE_URL="https://radassistant.vercel.app"
$env:E2E_API_URL="https://203-0-113-5.sslip.io"
npm run test:e2e
```

The sign-up test will fail — registration is off, correctly. This creates real
accounts and spends real tokens, so run it once to validate the deploy, not
routinely.

---

## 9. Publish

README:

```
**Live demo:** https://radassistant.vercel.app
Sign in: demo@example.com / <password>
Demonstration system — anonymised and synthetic data only.

Deploys as a self-contained Docker Compose stack (API, Postgres, Qdrant).
No patient data leaves the network it runs on.
```

Then merge — **after** the deploy is confirmed working, so `main` always means
"this ran in production":

```bash
git checkout main && git merge frontend-redesign && git push
git tag -a v1.0-pilot -m "First public deployment" && git push --tags
```

---

## Updating

```bash
cd RadAssistant && git pull
docker compose -f docker-compose.prod.yml up -d --build
```

Vercel redeploys the frontend automatically on push. Rebuild it manually
whenever `NEXT_PUBLIC_API_URL` changes — it is compiled in, so a restart is
not enough.

## Backups

Nothing here backs anything up.

```bash
docker compose -f docker-compose.prod.yml exec -T postgres \
  pg_dump -U radassist radassist_db | gzip > ~/backup-$(date +%F).sql.gz
```

Qdrant is rebuildable by re-seeding — though "rebuildable" means hours of NCBI
fetching, not minutes. Postgres holds the accounts and signed reports, which
are not rebuildable at all.

## Known gaps

- **Rate-limit counters are per-process and reset on restart.** Fine for a
  60-second window, meaningful for a daily one. The provider-side spend cap is
  the real bound.
- **`vision_service.py` is Groq-only.** Unlike `llm_service.py` it imports
  `AsyncGroq` directly with no fallback, so a retired vision model stops image
  reading with no automatic recovery.
- **Vercel preview deployments fail CORS** — exact-origin matching only.
- **No monitoring.** You find out it is down by visiting it. An uptime pinger
  against `/api/v1/health` is the cheapest fix.
