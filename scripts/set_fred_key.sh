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

# Deliberately NOT `set -a; . ./.env`. Sourcing .env as shell breaks on any
# unquoted value containing a space -- SEC_USER_AGENT=Dan Cooper <addr> tries
# to run `Cooper` as a command. data.http.load_dotenv parses the file properly,
# and using it here means this check exercises the same path the pipeline does.
exec "${PYTHON:-../.venv/bin/python}" - <<'PY'
from data.http import load_dotenv, user_agent
load_dotenv()
print('SEC User-Agent:', user_agent())

from datetime import date, timedelta

from data.fred_client import FREDClient
c = FREDClient()
if not c.api_key:
    raise SystemExit('[fail] client still sees no key')

# A date, not a string: _as_of_value compares observation keys against this
# directly, so a string raises TypeError rather than returning nothing.
# force=True so this reads FRED rather than a cache written before the key
# existed -- otherwise a dead key still "passes".
as_of = date.today() - timedelta(days=3)
c.fetch_series('BAMLC0A4CBBB', force=True)
oas = c.fetch_bucket_oas(as_of)
if not oas:
    raise SystemExit('[fail] key present but FRED returned nothing — '
                     'it may be unactivated or revoked')
print(f'Keyed request succeeded. Bucket OAS near {as_of}:')
for bucket, val in sorted(oas.items()):
    print(f'  {bucket:5s} {val * 1e4:7.1f} bp')
PY
