#!/usr/bin/env python3
"""Publish the report to the `pages-live` branch.

    python scripts/publish.py                     # newest report
    python scripts/publish.py --report path.html
    python scripts/publish.py --dry-run

WHY A SEPARATE BRANCH WITH ONE AMENDED COMMIT

The report is ~2.3 MB and regenerates every run. Committed normally that is
roughly 800 MB of history a year, none of which anyone will ever read, and the
sibling equity model hit exactly that wall — it now force-pushes a single
amended commit for the same reason. `pages-live` is an orphan branch holding
only docs/, and every publish amends its one commit and force-pushes.

Force-pushing is safe HERE precisely because the branch is disposable: it holds
no history worth keeping, it is never merged, and it is regenerated from
output/ on demand. It would not be safe anywhere else in this repo, so this
script refuses to operate on any other branch.

Work happens in a git worktree so the main working tree is never touched — no
stashing, no risk of publishing a half-finished edit.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.logging_setup import get_logger

log = get_logger('publish')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')
BRANCH = 'pages-live'
WORKTREE = os.path.join(REPO_ROOT, '.pages-live')


def git(*args, cwd=REPO_ROOT, check=True, quiet=False):
    result = subprocess.run(['git', *args], cwd=cwd, capture_output=True,
                            text=True)
    if check and result.returncode != 0:
        raise SystemExit(f'[fatal] git {" ".join(args)}\n{result.stderr.strip()}')
    if not quiet and result.stdout.strip():
        log.debug(result.stdout.strip())
    return result


def newest_report():
    candidates = sorted(glob.glob(os.path.join(OUTPUT_DIR,
                                               'bond_analysis_*.html')))
    if not candidates:
        raise SystemExit('[fatal] no report in output/ — run '
                         'scripts/report_html.py first')
    return candidates[-1]


def ensure_worktree():
    """A worktree on `pages-live`, creating the orphan branch if needed."""
    if os.path.isdir(WORKTREE):
        # A directory left behind by an interrupted run is reusable only if it
        # is genuinely a registered worktree on the publish branch. An orphan
        # checkout that was interrupted leaves a plain directory that `git
        # worktree remove` will not touch, so fall through to deleting it.
        head = git('rev-parse', '--abbrev-ref', 'HEAD', cwd=WORKTREE,
                   check=False, quiet=True)
        if head.returncode == 0 and head.stdout.strip() == BRANCH:
            return
        log.warning('Stale worktree at %s; recreating', WORKTREE)
        git('worktree', 'remove', '--force', WORKTREE, check=False, quiet=True)
        if os.path.isdir(WORKTREE):
            shutil.rmtree(WORKTREE, ignore_errors=True)
        git('worktree', 'prune', check=False, quiet=True)

    exists = git('rev-parse', '--verify', BRANCH, check=False,
                 quiet=True).returncode == 0
    if exists:
        git('worktree', 'add', WORKTREE, BRANCH)
    else:
        log.info('Creating the %s orphan branch', BRANCH)
        git('worktree', 'add', '--detach', WORKTREE)
        git('checkout', '--orphan', BRANCH, cwd=WORKTREE)
        # An orphan checkout inherits the index; clear it so the branch starts
        # with docs/ alone rather than a copy of the whole repo.
        git('rm', '-rf', '--cached', '.', cwd=WORKTREE, check=False, quiet=True)
        for entry in os.listdir(WORKTREE):
            if entry == '.git':
                continue
            path = os.path.join(WORKTREE, entry)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)


def stage_workflow():
    """Copy the Pages workflow onto the publish branch.

    GitHub reads workflow files from the branch that was PUSHED, so a
    `on: push: branches: [pages-live]` trigger in a workflow that lives only on
    main never fires — the first publish pushed successfully and nothing
    happened. The orphan branch needs its own copy.
    """
    source = os.path.join(REPO_ROOT, '.github', 'workflows', 'deploy-pages.yml')
    if not os.path.exists(source):
        log.warning('No deploy-pages.yml on main; Pages will not rebuild')
        return
    target_dir = os.path.join(WORKTREE, '.github', 'workflows')
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy2(source, os.path.join(target_dir, 'deploy-pages.yml'))


def stage(report_path):
    docs = os.path.join(WORKTREE, 'docs')
    os.makedirs(docs, exist_ok=True)
    for stale in glob.glob(os.path.join(docs, '*')):
        os.remove(stale) if os.path.isfile(stale) else None

    shutil.copy2(report_path, os.path.join(docs, 'index.html'))
    # Jekyll would otherwise ignore files it does not recognise.
    open(os.path.join(docs, '.nojekyll'), 'w').close()
    size = os.path.getsize(os.path.join(docs, 'index.html')) / 1e6
    log.info('Staged %s as docs/index.html (%.1f MB)',
             os.path.basename(report_path), size)
    return size


def commit_and_push(stamp, dry_run=False):
    git('add', '-A', 'docs', '.github', cwd=WORKTREE)
    status = git('status', '--porcelain', cwd=WORKTREE, quiet=True)
    if not status.stdout.strip():
        head = git('rev-parse', '--verify', 'HEAD', cwd=WORKTREE, check=False,
                   quiet=True)
        if head.returncode == 0:
            log.info('Report is unchanged; nothing to publish')
            return False

    has_commit = git('rev-parse', '--verify', 'HEAD', cwd=WORKTREE,
                     check=False, quiet=True).returncode == 0
    message = f'Published report {stamp}'
    args = ['commit', '-q', '-m', message]
    if has_commit:
        # Amend, so the branch never accumulates more than one report.
        args.insert(1, '--amend')
    if dry_run:
        log.info('[dry run] would %s and force-push %s',
                 'amend' if has_commit else 'commit', BRANCH)
        return True

    git(*args, cwd=WORKTREE)
    remote = git('remote', check=False, quiet=True).stdout.split()
    if not remote:
        log.warning('No git remote configured — committed locally only. '
                    'Add one, then re-run to publish.')
        return True
    git('push', '--force', remote[0], f'{BRANCH}:{BRANCH}', cwd=WORKTREE)
    log.info('Force-pushed %s to %s', BRANCH, remote[0])
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', default=None)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--keep-worktree', action='store_true')
    args = ap.parse_args()

    report = args.report or newest_report()
    stamp = os.path.basename(report)[14:24]

    ensure_worktree()
    try:
        stage_workflow()
        size = stage(report)
        if size > 95:
            raise SystemExit(f'[fatal] {size:.0f} MB exceeds the GitHub Pages '
                             f'100 MB per-file limit')
        published = commit_and_push(stamp, dry_run=args.dry_run)
    finally:
        if not args.keep_worktree:
            git('worktree', 'remove', '--force', WORKTREE, check=False)

    if published and not args.dry_run:
        print(f'\n  Published {stamp}. Pages will rebuild in a minute or two.')
        print('  Check the run: gh run list --workflow=deploy-pages.yml\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
