#!/usr/bin/env bash
# CLIENT-SIDE (runs on your laptop) — run the test suite on a remote node and print the result.
# scp's the scoped committed code over the SSM SSH proxy (no S3), runs run_ci_tests.sh there
# (ruff + the full test suite), waits, and prints the output. The training counterpart is train_aws.sh.
#
# Usage (AWS_PROFILE, INSTANCE_ID, SSH_KEY are required — no defaults):
#   AWS_PROFILE=<profile> INSTANCE_ID=i-... SSH_KEY=~/.ssh/<key> bash learning/scripts/aws/test_aws.sh
# Env (optional): AWS_REGION (us-east-1), REMOTE_DIR, PYTEST_ARGS (e.g. "-k split").
# NOTE: the first run on a fresh node builds the venv (torch etc., a few minutes).
set -euo pipefail

: "${INSTANCE_ID:?set INSTANCE_ID=i-... (the target node)}"
export AWS_PROFILE="${AWS_PROFILE:?set AWS_PROFILE=<your aws profile> (no default; be explicit about the account)}"
: "${SSH_KEY:?set SSH_KEY=/path/to/your/ssh/private/key (for scp over the SSM proxy)}"
# AWS_PROFILE must belong to the gpu-launchers group:
WHO=$(aws sts get-caller-identity --query Arn --output text)
aws iam list-groups-for-user --user-name "${WHO##*/}" --query 'Groups[].GroupName' --output text | grep -qw gpu-launchers \
  || { echo "REFUSING: $WHO is not in the gpu-launchers group." >&2; exit 1; }
export AWS_REGION="${AWS_REGION:-us-east-1}"
REMOTE_DIR="${REMOTE_DIR:-/opt/dlami/nvme/allex}"
PYTEST_ARGS="${PYTEST_ARGS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"

git -C "$REPO_ROOT" archive --format=tar.gz -o /tmp/allex_code.tgz HEAD learning pyproject.toml
PXY='aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p'
echo "scp $SHA -> ubuntu@$INSTANCE_ID (over SSM)"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o "ProxyCommand=$PXY" \
    /tmp/allex_code.tgz "ubuntu@$INSTANCE_ID:/tmp/allex_code.tgz"

REMOTE="set -e; rm -rf $REMOTE_DIR; mkdir -p $REMOTE_DIR; tar xzf /tmp/allex_code.tgz -C $REMOTE_DIR; cd $REMOTE_DIR; bash learning/scripts/run_ci_tests.sh $PYTEST_ARGS"
printf '{"commands":["%s"]}\n' "$REMOTE" > /tmp/allex_test_ssm.json
CID="$(aws ssm send-command --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript --parameters file:///tmp/allex_test_ssm.json \
  --query 'Command.CommandId' --output text)"
echo "running tests on $INSTANCE_ID (SSM $CID) ..."
aws ssm wait command-executed --command-id "$CID" --instance-id "$INSTANCE_ID" 2>/dev/null || true
echo "--- status ---";  aws ssm get-command-invocation --command-id "$CID" --instance-id "$INSTANCE_ID" --query Status --output text
echo "--- stdout ---";  aws ssm get-command-invocation --command-id "$CID" --instance-id "$INSTANCE_ID" --query StandardOutputContent --output text
echo "--- stderr ---";  aws ssm get-command-invocation --command-id "$CID" --instance-id "$INSTANCE_ID" --query StandardErrorContent --output text
