# CLAUDE.md — allex

Consolidated ALLEX repo. Add shared code here — don't spin up a new repo. Subsystem-specific notes live in each subdir's own `CLAUDE.md`.

Per-developer prefs (e.g. response language) go in a personal `CLAUDE.md` at the workspace level (dir above this repo, git-ignored) — keep this file team-shared.

## Conventions
- **No AI attribution** on commits/PRs (no `Co-Authored-By`, no generated-by footer).
- **No remote git ops without explicit, per-action approval** — push (incl. force-push), history-rewrite, merge/close PRs. Per-push, not per-task ("update the PR"/"relaunch" don't authorize a push); commit locally, push only when told. Same for AWS admin creds.
- **Read a PR's live title/description before editing it** — always fetch the current version (`gh pr view <n> --json title,body`) and edit from that, never from memory, so you don't clobber or drop content that's already there. Keep the description in sync with the pushed diff.
- **Branch names**: `dev/{username}/{branch_name}` — `dev/` prefix + your GitHub login + a short kebab-case description (e.g. `dev/chrisryu0/fk-validation`). Branches predating this convention are grandfathered; use it for all new branches.
- **Don't modify vendored/upstream code** (Isaac Lab, `allex_groot/Isaac-GR00T`) — wrap/override in our
  own code. GR00T is internalized under `learning/model/groot` (a faithful, minimally-diverged copy);
  `allex_groot/Isaac-GR00T` is kept as the pristine **upstream reference** (its `UPSTREAM.md` pins the
  source SHA) for diffing/re-syncing that copy — it is NOT installed, cloned, or imported at runtime.
- **Fail loudly, never silently** — no error-swallowing try/except or fallbacks; `assert` invariants and let errors propagate; catch only at system boundaries.
- **Imports at the top of the file**, never inside functions — explicit deps + fail-fast if one is missing.
- **`README.md` = user manual** — don't edit without an explicit request.

## Training
- **Run/deploy via the maintained scripts, not ad-hoc commands.** Use `learning/scripts/` (`train.sh`, `train_groot.sh`, `aws/train_aws.sh`) — don't hand-roll one-off `aws`/`ssm`/`accelerate`/`scp` invocations for a real run. If a script can't do what you need, extend it and commit; don't work around it with a throwaway command (and never stage code through S3 — `train_aws.sh` scp's over SSM).
- **Smoke-test val/ckpt/sim-eval changes** before any long run — poll at short intervals to confirm one full cycle works end-to-end.
- **wandb run names `[name]-[datetime]-[SHA]`** (UTC + short git SHA) so runs trace to exact code; scratch/sweep → shared dev project, not one-off projects.

## Testing
- **Test the production code path, not a re-implementation** — exercise the real functions; don't paraphrase logic into the test.

## Storage & docs
- **S3 ⇄ CloudFront**: the bucket `s3://wirobotics-internal/` is fronted by the CloudFront
  distribution `https://d1iitptfxhu64e.cloudfront.net/` — the object key maps 1:1. So
  `s3://wirobotics-internal/<key>` ⇄ `https://d1iitptfxhu64e.cloudfront.net/<key>`
  (e.g. `s3://wirobotics-internal/hansol/docs/foo.html` ⇄ `…cloudfront.net/hansol/docs/foo.html`).
  CloudFront URLs are **Slack-authed** (browser only); from code/CLI read the `s3://` path instead.
- Datasets live under `s3://wirobotics-internal/<user>/datasets/allex/...` — **raw captures** under
  `…/raw/...`, **converted LeRobot datasets** under `…/lerobot/...` (LeRobot datasets do NOT go under
  `raw/`). Shared docs under `…/<user>/docs/...`.
