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

# This repo's OWN interpreter. This used to default to ../.venv/bin/python --
# the stock-analysis-model venv -- back when this repo lived inside that
# checkout. Reaching outside the repo for an interpreter meant a fresh clone
# had no working script; keep this path local. Override with PYTHON=... .
PYTHON="${PYTHON:-./.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "No interpreter at $PYTHON." >&2
  echo "Create one:  python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt" >&2
  echo "Or point at an existing one:  PYTHON=/path/to/python $0" >&2
  exit 1
fi

[ -f .env ] || cp .env.example .env

# --edit skips the prompt entirely and opens .env in an editor. The prompt
# needs a terminal, and plenty of ways of running this script do not have one.
if [ "${1:-}" = "--edit" ]; then
  ed=${VISUAL:-${EDITOR:-}}
  if [ -n "$ed" ]; then "$ed" .env
  elif command -v open >/dev/null 2>&1; then open -e .env
  else echo "Set \$EDITOR, or open .env by hand." >&2; exit 1
  fi
  echo "Fill in the FRED_API_KEY= line, save, then re-run without --edit to verify."
  exit 0
fi

# Prefer /dev/tty over stdin: stdin may be a pipe or closed depending on how the
# script was launched, and `read` would take EOF as an empty answer and report a
# zero-length key instead of the real problem. Probe it by actually opening it
# -- `[ -r /dev/tty ]` passes on the permission bits even where the device is
# not connected, which is exactly the case that needs catching.
# The subshell matters: a redirection failure is reported by the shell itself,
# so `: < /dev/tty 2>/dev/null` still prints "Device not configured". Wrapping
# it puts the failing redirection inside a shell whose stderr is discarded.
if ( exec < /dev/tty ) 2>/dev/null; then
  key_src=/dev/tty               # a real terminal: prompt, hide the input
else
  key_src=                       # stdin: a terminal, or a pipe like `pbpaste |`
fi

if [ -n "$key_src" ] || [ -t 0 ]; then
  echo 'Paste your FRED API key and press Enter.'
  echo 'Nothing will appear as you type -- the input is hidden on purpose.'
  printf 'Key: '
fi
if [ -n "$key_src" ]; then read -rs key < "$key_src"; else read -rs key || key=; fi
[ -n "$key_src" ] || [ -t 0 ] && echo

# Normalise before validating. A paste routinely carries a leading or trailing
# space, and FRED's dashboard has been known to show the key uppercased.
key=$(printf '%s' "$key" | tr -d '[:space:]' | tr 'A-F' 'a-f')

# 32 hex. The check exists because a 31-character paste is the realistic
# failure: it silently falls back to the keyless endpoint, and you find out
# weeks later when the historical backfill comes back thin.
if [ -z "$key" ]; then
  echo "No key received -- nothing was written." >&2
  echo "If you typed one, the terminal may not be passing input through;" >&2
  echo "try ./scripts/set_fred_key.sh --edit instead." >&2
  exit 1
fi
if ! printf '%s' "$key" | grep -Eq '^[0-9a-f]{32}$'; then
  echo "Not a FRED key: expected 32 hex characters, got ${#key}." >&2
  echo "Nothing was written." >&2
  exit 1
fi

# Verify BEFORE writing. Writing first and checking after leaves a rejected key
# sitting in .env, which is worse than no key: the pipeline stops falling back
# to the keyless endpoint and just fails.
#
# The key reaches Python through the environment rather than a file or an
# argument -- an argument shows up in `ps` for the lifetime of the process.
echo 'Checking the key against FRED...'
if ! FRED_CANDIDATE_KEY="$key" "$PYTHON" - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from data.fred_client import FREDClient

c = FREDClient(api_key=os.environ['FRED_CANDIDATE_KEY'])
# _fetch_keyed, NOT fetch_series. fetch_series falls back to the keyless
# endpoint when the keyed call fails -- right for the pipeline, useless here:
# a revoked key would fall through, return real data, and this check would
# congratulate you on a key that does not work.
obs = c._fetch_keyed('BAMLC0A4CBBB')
if not obs:
    sys.exit('[fail] FRED rejected the key. The format was valid, so it is\n'
             '       wrong, revoked, or not yet activated -- new keys can take\n'
             '       a few minutes. Nothing was written; the pipeline still\n'
             '       runs keyless on a ~3-year history window.')
print(f'  accepted: {len(obs):,} observations, earliest {min(obs)}')
PY
then
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
unset key FRED_CANDIDATE_KEY

echo "Written to .env (mode 600, git-ignored)."
echo

# Deliberately NOT `set -a; . ./.env`. Sourcing .env as shell breaks on any
# unquoted value containing a space -- SEC_USER_AGENT=Dan Cooper <addr> tries
# to run `Cooper` as a command. data.http.load_dotenv parses the file properly,
# and using it here means this check exercises the same path the pipeline does.
exec "$PYTHON" - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from datetime import date, timedelta

from data.http import load_dotenv, user_agent
load_dotenv()
print('SEC User-Agent:', user_agent())

from data.fred_client import FREDClient
c = FREDClient()
if not c.api_key:
    sys.exit('[fail] written, but the client still does not see it')

# A date, not a string: _as_of_value compares observation keys against this
# directly, so a string raises TypeError rather than returning nothing.
as_of = date.today() - timedelta(days=3)
oas = c.fetch_bucket_oas(as_of)
print(f'Bucket OAS near {as_of}:')
for bucket, val in sorted(oas.items()):
    print(f'  {bucket:5s} {val * 1e4:7.1f} bp')
print('\nReady. The FRED-gated step is scripts/calibrate_credit.py.')
PY
