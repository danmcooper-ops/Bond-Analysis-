#!/bin/bash
# Prepare a Mac to run the daily bond analysis. Run ON that Mac. Idempotent.
#
#   scheduling/setup_mini.sh            # check + build what it safely can
#
# Builds the venv and checks everything else the scheduled task needs. It
# never handles secrets and never creates the scheduled task: those steps are
# printed at the end as a checklist, because an unattended run that hits an
# unapproved command or a missing key fails silently (2026-09-09, 09-14).
set -uo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
# A non-login SSH shell does not have Homebrew on PATH.
export PATH="/opt/homebrew/bin:$PATH"
py="$repo/.venv/bin/python"
fail=0
ok()   { echo "  ok    $*"; }
bad()  { echo "  TODO  $*"; fail=1; }

echo "== tools"
for tool in git gh python3.14 rsync; do
    command -v "$tool" >/dev/null && ok "$tool" || bad "install $tool (brew install ${tool/python3.14/python@3.14})"
done

echo "== venv"
if command -v python3.14 >/dev/null; then
    # Rebuilt, never copied: a moved venv keeps the old machine's paths in its
    # script shebangs, so pip breaks while python still appears to work.
    [ -x "$py" ] || python3.14 -m venv "$repo/.venv"
    "$repo/.venv/bin/pip" install --quiet --upgrade pip \
        && "$repo/.venv/bin/pip" install --quiet -r "$repo/requirements.txt" \
        && ok "venv $("$py" --version 2>&1)" || bad "venv install failed"
fi

echo "== github"
if gh auth status >/dev/null 2>&1; then
    ok "gh logged in"
    git -C "$repo" config --get-regexp '^credential\..*helper' >/dev/null \
        || git config --global --get-regexp '^credential\..*helper' >/dev/null \
        && ok "git credential helper" || bad "gh auth setup-git"
    git -C "$repo" ls-remote --exit-code origin pages-live >/dev/null 2>&1 \
        && ok "origin reachable" || bad "cannot reach origin pages-live"
else
    bad "gh auth login   (HTTPS, account danmcooper-ops), then: gh auth setup-git"
fi

echo "== .env (names only)"
env_file="$repo/.env"
if [ -f "$env_file" ]; then
    for key in FRED_API_KEY SEC_USER_AGENT; do
        # Length, not presence: the key has been wiped to an empty value before.
        len=$(grep -E "^$key=" "$env_file" | head -1 | cut -d= -f2- | tr -d '"'"'" | wc -c)
        [ "$len" -gt 8 ] && ok "$key set" || bad "$key missing or empty"
    done
    [ "$(stat -f %Lp "$env_file")" = "600" ] && ok ".env mode 600" || bad "chmod 600 .env"
else
    bad "no .env: run scripts/set_fred_key.sh, then add SEC_USER_AGENT=<name email>"
fi

echo "== data"
ls "$repo"/output/universe_*.parquet >/dev/null 2>&1 \
    && ok "universe $(basename "$(ls "$repo"/output/universe_*.parquet | tail -1)")" \
    || bad "no output/universe_*.parquet: run push_to_mini.sh from the old machine"
[ -d "$repo/data/cache/nport" ] && ok "data/cache/nport" \
    || bad "no data/cache/nport: run push_to_mini.sh (or re-ingest, 30-90 min)"
if [ -x "$py" ]; then
    (cd "$repo" && "$py" scripts/daily_checks.py gaps | sed 's/^/        /')
fi

echo "== tests"
if [ -x "$py" ]; then
    (cd "$repo" && "$py" -m pytest -q -x 2>&1 | tail -1 | sed 's/^/        /')
fi

echo "== machine"
[ "$(pmset -g | awk '$1=="sleep"{print $2}')" = "0" ] && ok "system sleep off" \
    || bad "System Settings > Energy: prevent automatic sleeping"
[ "$(pmset -g | awk '$1=="autorestart"{print $2}')" = "1" ] && ok "restart after power failure" \
    || bad "System Settings > Energy: start up after a power failure"
[ -d /Applications/Claude.app ] && ok "Claude.app installed" || bad "install the Claude desktop app"

cat <<EOF

Then, by hand (in this order):
  1. Old machine: pause the daily-bond-analysis task in the Claude app, then
     remove ~/.claude/scheduled-tasks/daily-bond-analysis there.
  2. Old machine: scheduling/push_to_mini.sh <this host> --go
  3. Here: Claude app > new scheduled task "daily-bond-analysis", weekdays
     07:00, working folder $repo, body = scheduling/daily-bond-analysis.SKILL.md.
     Add Claude to Login Items.
  4. Here: "Run now" once while watching, and choose Always allow on every
     step's command exactly as asked. Confirm a NEW deploy-pages.yml run.
EOF
exit $fail
