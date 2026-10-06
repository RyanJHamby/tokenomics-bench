#!/usr/bin/env bash
# Ship results off the pod every INTERVAL seconds, so a terminated pod loses at most a minute.
#
#   SYNC_DEST=/path        copy tarballs there (rsync mount, reverse-tunnel dir, a volume)
#   SYNC_RELEASE=<tag>     upload to a GitHub release with `gh` (needs a token ON the pod:
#                          use a fine-grained token limited to this repo's contents, revoke it after)
#   ONCE=1                 one pass then exit (tests)
#
# The pod cannot make signed commits; pull the tarballs to the laptop and commit there.
set -uo pipefail
ROOT=${1:-results}
INTERVAL=${INTERVAL:-60}; ONCE=${ONCE:-0}
MARK="$ROOT/.sync_marker"; N=0
mkdir -p "$ROOT/_sync"
[ -f "$MARK" ] || { : > "$MARK"; touch -t 197001010000 "$MARK"; }

ship() { # tarball
  if [ -n "${SYNC_DEST:-}" ]; then
    mkdir -p "$SYNC_DEST" && cp "$1" "$SYNC_DEST/"
  elif [ -n "${SYNC_RELEASE:-}" ] && command -v gh >/dev/null 2>&1; then
    gh release view "$SYNC_RELEASE" >/dev/null 2>&1 \
      || gh release create "$SYNC_RELEASE" --title "$SYNC_RELEASE" --notes "raw run data (auto-synced)" --prerelease >/dev/null
    gh release upload "$SYNC_RELEASE" "$1" --clobber
  else
    echo "[sync] no destination configured (SYNC_DEST or SYNC_RELEASE)"; return 1
  fi
}

while :; do
  NEWMARK=$(mktemp)   # taken BEFORE listing: files written during the tar are caught next pass
  files=$(find "$ROOT" -type f -newer "$MARK" ! -path "$ROOT/_sync/*" ! -name '.sync_marker' ! -name '*.tmp')
  if [ -n "$files" ]; then
    N=$((N+1))
    part="$ROOT/_sync/part$(printf %04d "$N")-$(date +%Y%m%d-%H%M%S).tar.gz"
    if echo "$files" | tar -czf "$part" -T - 2>/dev/null && ship "$part"; then
      touch -r "$NEWMARK" "$MARK"; echo "[sync] shipped $(basename "$part") ($(echo "$files" | wc -l | tr -d ' ') files)"
    else
      echo "[sync] FAILED to ship $(basename "$part"); will retry next pass"
    fi
  fi
  rm -f "$NEWMARK"
  [ "$ONCE" = 1 ] && break
  sleep "$INTERVAL"
done
