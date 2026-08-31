#!/usr/bin/env bash
# CLIENT-SIDE — launch a single-GPU node (default g6e = L40S) and print INSTANCE_ID. MUST run under
# a gpu-launchers profile (group membership is checked), not an admin one. The node is tagged
# ManagedBy=gpu-launcher (the gpu-launchers SSM policy keys on it) and named <profile>-<NODE_NAME>.
#
# The node config below is the account's standard gpu-launcher config, declared here (version-
# controlled) and overridable via env. These are shared infra references (private repo), not secrets.
#
# Usage:  AWS_PROFILE=<gpu-launchers profile> bash learning/scripts/aws/launch_aws.sh
set -euo pipefail

: "${AWS_PROFILE:?set AWS_PROFILE=<your gpu-launchers profile>}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6e.2xlarge}"                          # 1x L40S
case "$INSTANCE_TYPE" in                                              # GPU label for the node name
  g6e.*) GPU=l40s ;; g6.*) GPU=l4 ;; g5.*) GPU=a10g ;;
  p4d.*|p4de.*) GPU=a100 ;; p5.*|p5e.*|p5en.*) GPU=h100 ;; p3.*) GPU=v100 ;;
  *) GPU="${INSTANCE_TYPE%%.*}" ;;
esac
NODE_NAME="${NODE_NAME:-$GPU}"                                         # node = <profile>-<gpu>, e.g. chrisryu-gpu-l40s
OWNER="${OWNER:-${AWS_PROFILE%%-*}}"                                    # Owner tag for cost attribution (CLAUDE.md); scoped profile can only tag at RunInstances
KEY_NAME="${KEY_NAME:-project-x}"                                      # ec2 key pair (you hold the private key)
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-0c5cdbadc0aeefe42}"        # VPC default SG: no public inbound, open egress (SSM)
IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE:-project-x-ssm-profile}" # node role: SSM agent + S3
# (no subnet: EC2 auto-picks an AZ in the default VPC, like project-x/launch_gpu_node.sh)

# AWS_PROFILE must belong to the gpu-launchers group:
WHO=$(aws sts get-caller-identity --query Arn --output text)
aws iam list-groups-for-user --user-name "${WHO##*/}" --query 'Groups[].GroupName' --output text | grep -qw gpu-launchers \
  || { echo "REFUSING: $WHO is not in the gpu-launchers group." >&2; exit 1; }

# latest Deep Learning Base DLAMI (Ubuntu 22.04):
DLAMI='Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*'
AMI_ID="${AMI_ID:-$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=$DLAMI" \
  --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)}"

IID="$(aws ec2 run-instances --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SECURITY_GROUP_ID" \
  --iam-instance-profile Name="$IAM_INSTANCE_PROFILE" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$AWS_PROFILE-$NODE_NAME},{Key=ManagedBy,Value=gpu-launcher},{Key=Owner,Value=$OWNER}]" \
  --query 'Instances[0].InstanceId' --output text)"
echo "launched $IID ($INSTANCE_TYPE, name=$AWS_PROFILE-$NODE_NAME) — waiting for status checks + SSM ..."
aws ec2 wait instance-status-ok --instance-ids "$IID"
echo "READY: INSTANCE_ID=$IID bash learning/scripts/aws/train_aws.sh"
