# CLAUDE.md — learning/

GR00T / ALLEX training and experiments. Guidance for anyone (including Claude) working here.

## Launch AWS GPU nodes through the gpu-launchers kit

Launch **all** GPU nodes for training/eval via the **gpu-launchers** kit
(`../sandbox/chrisryu0/gpu-launchers/scripts/`) — not ad-hoc `aws ec2 run-instances`:

- `launch_gpu_node.sh` (A100/H100/H200, capacity-retry) · `launch_l40s_node.sh` (g5/A10G) · `launch_v100_node.sh`
- Reach and manage them over SSM: `ssm.sh {connect|stop|start} <name>` (no SSH keys / open ports)

Why it matters:
- launches under the **scoped least-privilege** GPU profile, and makes the node **SSM-reachable**;
- tags every node `ManagedBy=gpu-launcher` (**required** — the scoped policy gates stop/terminate and
  SSM on that tag) and `Owner=$AWS_USER` (cost attribution + the running-compute snapshot);
- ad-hoc launches skip the scoping and tagging, and show up in cost reports only as unowned/"shared".

Cost is reported by the **aws-usage-reporter** kit (daily/weekly Slack + S3 dashboard); per-person spend
is attributed automatically via the AWS-managed `aws:createdBy` tag. **Stop or terminate GPU nodes as
soon as they're idle** — they are by far the dominant spend (A100 ~$41/hr, H100 ~$55/hr).
