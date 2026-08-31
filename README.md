# allex

Single repository for the ALLEX stack — robot learning (RL / VLA / sim), on-robot deployment, and the shared contracts they depend on. Private repository.

## Objective

Consolidate the ALLEX components — which grew as separate repos that each copied the same contracts (robot model, state/action layouts, topic maps, data converters) and then drifted — into one repository where those contracts are **single-sourced** (referenced, not copied) and a change to a contract plus all of its consumers lands in a single atomic commit. See the [restructure report](https://d1iitptfxhu64e.cloudfront.net/chrisryu/docs/allex_restructure_report.html) (Slack-authed) for the full design and rationale.

## Setup

After cloning, activate the repo's git hooks (one-time, per clone):

```sh
git config core.hooksPath .githooks
```

Two local guards (the first needs `uv` or `ruff` on your PATH):
- **`pre-commit`** — runs `ruff check learning` (import order + lint) on staged `learning/` changes. Mirrors CI.
- **`pre-push`** — blocks accidental direct pushes to `main` (open a PR instead).

**If you're going to train or test** (locally or on AWS GPU nodes), continue the setup — editable
install, AWS profile/key, and W&B — in **[`learning/README.md`](learning/README.md)**.

## Storage (S3 ⇄ CloudFront)

Shared data and docs live in `s3://wirobotics-internal/`, fronted by the CloudFront distribution
`https://d1iitptfxhu64e.cloudfront.net/` with a 1:1 key mapping:

```
s3://wirobotics-internal/<key>   ⇄   https://d1iitptfxhu64e.cloudfront.net/<key>
```

e.g. `s3://wirobotics-internal/hansol/docs/phase1_bolt_drilling_data_map.html` is the same object as
`https://d1iitptfxhu64e.cloudfront.net/hansol/docs/phase1_bolt_drilling_data_map.html`. CloudFront
URLs are **Slack-authenticated** (open in a browser); from code or CLI use the `s3://` path. Training
datasets: `s3://wirobotics-internal/chrisryu/datasets/allex/raw/...`.
