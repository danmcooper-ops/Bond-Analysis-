#!/usr/bin/env bash
# Put a FRED API key into .env, then prove it works.
#
# The key is read from a silent terminal prompt, never passed as an argument.
# An argument would land in your shell history, in `ps` output while the script
# runs, and in the transcript of whatever tool invoked it.
#
#   ./scripts/set_fred_key.sh
#
# Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || cp .env.example .env

read -rsp 'FRED API key: ' key
echo

# 32 lowercase hex. The check exists because a 31-character paste is the
# realistic failure: it silently falls back to the keyless endpoint, and you
# find out weeks later when the historical backfill is thin.
if ! printf '%s' "$key" | grep -Eq '^[0-9a-f]{32}$'; then
  echo "Not a FRED key: expected 32 lowercase hex characters, got ${#key}." >&2
  echo "Nothing was written." >&2
  exit 1
fi

# Write via a temp file with tight permissions rather than editing in place, so
# the key is never briefly world-readable.
umask 077
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
if grep -q '^FRED_API_KEY=' .env; then
  # awk, not sed: a sed replacement interpolates the key into the script text,
  # where any & or | in it would be a substitution metacharacter.
  awk -v k="$key" 'BEGIN{FS=OFS="="} /^FRED_API_KEY=/{print "FRED_API_KEY=" k; next} {print}' .env > "$tmp"
else
  { cat .env; printf 'FRED_API_KEY=%s\n' "$key"; } > "$tmp"
fi
cat "$tmp" > .env
chmod 600 .env
unset key

echo "Written to .env (mode 600, git-ignored)."
echo

set -a; . ./.env; set +a
exec "${PYTHON:-../.venv/bin/python}" - <<'PY'
from data.fred_client import FredClient
c = FredClient()
if not c.api_key:
    raise SystemExit('[fail] client still sees no key')
oas = c.fetch_bucket_oas('2026-08-05')
if not oas:
    raise SystemExit('[fail] key present but FRED returned nothing — '
                     'it may be unactivated or revoked')
print('Keyed request succeeded. Bucket OAS on 2026-08-05:')
for bucket, bp in sorted(oas.items()):
    print(f'  {bucket:5s} {bp:7.1f} bp')
PY
