#!/bin/bash
# Copy the bond model's git-ignored state from this machine to another Mac.
#
#   scheduling/push_to_mini.sh dmcooper@Dans-Mac-Mini.local          # dry run
#   scheduling/push_to_mini.sh dmcooper@Dans-Mac-Mini.local --go     # copy
#
# Moves only what git cannot: output/ (the snapshot corpus, the built
# universe, the calibration files) and data/cache/ (N-PORT, SEC, Treasury,
# FRED; ~700 MB, saves a 30-90 min re-ingest). Never .env -- secrets are
# re-entered on the target with scripts/set_fred_key.sh -- and never .venv,
# whose shebangs pin this machine's paths.
#
# Needs passwordless SSH to the target (ssh-copy-id once) and the repo already
# cloned there at the same path relative to that account's home (the mini's
# account is dmcooper, not danmcooper; REMOTE_REPO overrides). Safe to run
# early and again at cutover: rsync only sends what changed. The final pass
# goes AFTER the last run here, with this machine's scheduled task disabled.
set -euo pipefail

host="${1:?usage: $0 <user@host> [--go]}"
go="${2:-}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
# Relative paths on the remote side resolve against its home directory.
remote_repo="${REMOTE_REPO:-${repo#"$HOME"/}}"

if pgrep -f analyze_bonds.py >/dev/null; then
    echo "analyze_bonds.py is running here; wait for it to finish" >&2
    exit 2
fi
ssh -o BatchMode=yes "$host" "test -d '$remote_repo/.git'" || {
    echo "no clone at $host:$remote_repo (or no passwordless SSH)" >&2
    exit 2
}

flags=(-a --human-readable --stats --exclude '*.tmp.*' --exclude '.DS_Store')
[ "$go" = "--go" ] || flags+=(--dry-run)

for dir in output data/cache; do
    echo "== $dir"
    ssh -o BatchMode=yes "$host" "mkdir -p '$remote_repo/$dir'"
    rsync "${flags[@]}" "$repo/$dir/" "$host:$remote_repo/$dir/"
done

[ "$go" = "--go" ] || echo "(dry run; pass --go to copy)"
