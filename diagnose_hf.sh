#!/usr/bin/env bash
# Why can't the backend container reach huggingface.co?
#
# NOTE: python:3.12-slim has no curl/wget, so everything below goes through
# Python's socket layer — which is also exactly what sentence-transformers
# uses, making this a faithful test rather than an approximation.

docker-compose exec -T backend python3 - <<'PY'
import socket, ssl, time

TARGETS = [
    ("api.groq.com",   443, "LLM provider — this worked earlier"),
    ("huggingface.co", 443, "model download — currently failing"),
    ("pypi.org",       443, "general internet control"),
]

print("=" * 62)
for host, port, why in TARGETS:
    print(f"\n{host}:{port}   ({why})")

    # 1. DNS
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        addrs = sorted({i[4][0] for i in infos})
        v4 = [a for a in addrs if ":" not in a]
        v6 = [a for a in addrs if ":" in a]
        print(f"   DNS  ok   IPv4={v4[:3]}  IPv6={v6[:2] or 'none'}")
    except Exception as e:
        print(f"   DNS  FAIL  {type(e).__name__}: {e}")
        continue

    # 2. Raw TCP to each IPv4 address — isolates routing from TLS.
    reached = False
    for addr in v4[:3]:
        t0 = time.time()
        try:
            s = socket.create_connection((addr, port), timeout=8)
            s.close()
            print(f"   TCP  ok   {addr}  ({time.time()-t0:.2f}s)")
            reached = True
            break
        except OSError as e:
            print(f"   TCP  FAIL {addr}  errno={e.errno} {e.strerror or e}")

    if not reached:
        continue

    # 3. TLS handshake + HTTP
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ss:
                ss.send(f"HEAD / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode())
                line = ss.recv(64).decode(errors="replace").split("\r\n")[0]
                print(f"   TLS  ok   {line}")
    except Exception as e:
        print(f"   TLS  FAIL  {type(e).__name__}: {e}")

print("\n" + "=" * 62)
print("""
READING THIS:
  groq ok, hf TCP FAIL errno=101  -> those specific IPs are unroutable
                                     (ISP/firewall blocking, or a stale
                                      Docker network after the WSL restart)
  everything FAIL                 -> container lost all networking;
                                     restart Docker Desktop
  all ok                          -> transient; just retry the download
""")
PY
