---
name: daily-bond-analysis
description: Run the full bond analysis pipeline and publish the updated report to GitHub Pages
---

You are running the daily bond analysis routine. Execute the steps in order. Stop and report an error if a step marked BLOCKING fails.

## Execution mode
- **Run fully autonomously.** Do not pause for confirmation — this is an unattended scheduled run. Make reasonable choices for ambiguity and note them in the run summary. Only the write actions described below (writing output artifacts, force-pushing the `pages-live` branch) are permitted; take no other outward-facing or destructive actions.
- Reference `/Users/danmcooper/Projects/bond-analysis-model/CLAUDE.md` for project context if needed.

## Paths
- **Repo:** `/Users/danmcooper/Projects/bond-analysis-model` (its own git repo; remote `origin` = github.com/danmcooper-ops/Bond-Analysis-)
- **Python:** `/Users/danmcooper/Projects/bond-analysis-model/.venv/bin/python` (the repo's own venv — do NOT use the stock model's workspace venv)
- **Equity fundamentals:** read automatically from the sibling stock model's snapshots via `EQUITY_SNAPSHOT_DIR` pinned in the repo's `.env`. If the run log says "every issuer will score without credit gates", the `.env` line is missing — report it prominently and continue (the run is still valid, just thinner).

## IMPORTANT: Command format
Send every Python invocation as a **single-line semicolon-separated Bash command** (not multi-line) so permission matching works. Use the exact format shown per step — **character for character: no output redirection (`>`), pipes, `tee`, `head`/`tail`, or extra flags.** Commands are matched against stored approvals; a changed command waits for a human to click, and on an unattended run nobody does (2026-09-14 stalled at Step 1 because the command had `> …/scratchpad/…` appended). If output is long, read it from the tool result as-is.

## Run date
Determine the run date ONCE, at the start, and use that literal value (e.g. `2026-09-14`) in every later step — never re-run `date`, so a run that crosses midnight stays on its start date:
```
date +%F
```
Below, `RUN_DATE` means that literal value.

## Step 0 — Guards (competing run BLOCKING; gaps report-only)
```
pgrep -fl "analyze_bonds.py" || echo "no competing run"
```
If a competing `analyze_bonds.py` process exists, stop and report it instead of starting a second pipeline (two concurrent runs writing the same results parquet has happened before and double-published).

```
PYTHON="/Users/danmcooper/Projects/bond-analysis-model/.venv/bin/python"; cd "/Users/danmcooper/Projects/bond-analysis-model"; "$PYTHON" scripts/daily_checks.py gaps
```
Copy every `ALERT` line into the run summary under a **Missed runs** heading. Continue regardless.

## Step 1 — Run the analysis (BLOCKING)
```
PYTHON="/Users/danmcooper/Projects/bond-analysis-model/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi 2>/dev/null) && export SSL_CERT_FILE; cd "/Users/danmcooper/Projects/bond-analysis-model"; "$PYTHON" scripts/analyze_bonds.py --as-of RUN_DATE
```
- The universe defaults to `all` and the corporate quarter to the newest built universe.
- Expected runtime 5–15 minutes over ~9,000 instruments.
- At 07:00 local, today's Treasury curve is not yet published; the client falls back to the most recent business day's curve and logs "No curve for X; using Y". That is normal — note the curve date used in the run summary.
- If the log says "No FRED_API_KEY", the run still works on the keyless fallback; mention it in the summary so the user knows to re-enter the key via `scripts/set_fred_key.sh` (NEVER write a key into `.env` yourself, and never echo key material).
- Success: exit 0, and both `output/results_RUN_DATE.parquet` and `output/run_meta_RUN_DATE.json` exist (run_meta is written last; its presence means the run completed).
- If the script exits non-zero, STOP. Do not render or publish — an older report must never go out in its place.

## Step 2 — Render the HTML report (BLOCKING)
```
PYTHON="/Users/danmcooper/Projects/bond-analysis-model/.venv/bin/python"; cd "/Users/danmcooper/Projects/bond-analysis-model"; "$PYTHON" scripts/report_html.py --run-date RUN_DATE
```
Success: exit 0 and `output/bond_analysis_RUN_DATE.html` exists (~2–3 MB). The script refuses to fall back to an older snapshot. If the instrument count is far from ~9,000 (say below 5,000), report it as a data problem and still publish — the report itself surfaces coverage.

## Step 3 — Publish to GitHub Pages (separate failure surface)
```
PYTHON="/Users/danmcooper/Projects/bond-analysis-model/.venv/bin/python"; cd "/Users/danmcooper/Projects/bond-analysis-model"; "$PYTHON" scripts/publish.py --run-date RUN_DATE
```
- Exit 0: pushed. Exit 3: the published report was already identical — nothing pushed, skip the deploy verification and say so. Any other non-zero: publish failed; report it clearly (today's parquet and HTML still exist locally and the publish can be retried without re-running the analysis).

After an exit-0 publish, verify the deploy actually ran (the push trigger has silently not fired before):
```
cd "/Users/danmcooper/Projects/bond-analysis-model"; gh run list --workflow=deploy-pages.yml --limit 1
```
Confirm the newest run is NEW (started within the last few minutes) and succeeded. If no new run appeared, trigger one with `cd "/Users/danmcooper/Projects/bond-analysis-model"; gh workflow run deploy-pages.yml --ref pages-live` and check again.

## Step 4 — Rating health (report-only)
```
PYTHON="/Users/danmcooper/Projects/bond-analysis-model/.venv/bin/python"; cd "/Users/danmcooper/Projects/bond-analysis-model"; "$PYTHON" scripts/daily_checks.py ratings --run-date RUN_DATE
```
Include the rating-mix table in the summary. Surface every `ALERT` line prominently — zero corporate BUY/LEAN BUY, HY BUYs above IG, >90% capped, or a BUY share moving >3pp all indicate a data or calibration problem rather than a market move.

## Step 5 — N-PORT freshness (report-only)
```
PYTHON="/Users/danmcooper/Projects/bond-analysis-model/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi 2>/dev/null) && export SSL_CERT_FILE; cd "/Users/danmcooper/Projects/bond-analysis-model"; "$PYTHON" scripts/daily_checks.py nport
```
If it reports a published-but-not-ingested quarter or an old data vintage, surface it prominently. Do NOT ingest — that is a manual 30–90 minute job for the user.

## Success criteria
- `output/results_RUN_DATE.parquet` and `run_meta_RUN_DATE.json` exist (the accumulating snapshot corpus — the long-term point of the daily run)
- `output/bond_analysis_RUN_DATE.html` was rendered
- The Pages deploy ran with a NEW run id and succeeded, or its failure (or "unchanged") is clearly reported
- The run summary includes: curve date used, instrument count, rating mix per asset class, all ALERT lines from Steps 0/4/5, any FRED-key or equity-snapshot warnings

## Notes
- The live report is at https://danmcooper-ops.github.io/Bond-Analysis-/
- Marks are monthly N-PORT data at ~60-day lag; the daily run's value is the fresh curve overlay and the accumulating snapshot corpus, not new marks
- Do not commit anything to `main`; the only git write is the `pages-live` force-push inside `publish.py`
