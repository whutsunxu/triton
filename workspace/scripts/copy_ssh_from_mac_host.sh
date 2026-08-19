#!/usr/bin/env bash
# Run on the Mac HOST (not inside the container).
# Copies ~/.ssh into the container filesystem (no bind mount).
set -euo pipefail

SRC="${1:-$HOME/.ssh}"
CONTAINER="${2:-}"

if [[ -z "$CONTAINER" ]]; then
  echo "Usage: $0 [host_ssh_dir] <container_name_or_id>" >&2
  echo "Example: $0 /Users/sunxu/.ssh 3e660b62e3d6" >&2
  echo "Find container: docker ps --format '{{.Names}}\t{{.ID}}'" >&2
  exit 1
fi

if [[ ! -d "$SRC" ]]; then
  echo "Host SSH dir not found: $SRC" >&2
  exit 1
fi

docker exec "$CONTAINER" mkdir -p /root/.ssh
docker cp "$SRC/." "$CONTAINER:/root/.ssh/"
docker exec "$CONTAINER" bash -lc '
  chmod 700 /root/.ssh
  find /root/.ssh -type f \( -name "id_*" -o -name "id_*_*" \) ! -name "*.pub" -exec chmod 600 {} + 2>/dev/null || true
  chmod 644 /root/.ssh/*.pub 2>/dev/null || true
  chmod 644 /root/.ssh/known_hosts 2>/dev/null || true
  chmod 600 /root/.ssh/config 2>/dev/null || true
'
echo "Copied $SRC -> $CONTAINER:/root/.ssh"
docker exec "$CONTAINER" bash -lc 'ls -la /root/.ssh; ssh -o BatchMode=yes -T git@github.com || true'
