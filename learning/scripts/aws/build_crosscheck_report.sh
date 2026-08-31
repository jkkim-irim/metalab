#!/usr/bin/env bash
# Build a PAIRED reference-vs-policy cross-check report: row i = Reference i (kinematic replay) next to
# the policy rollout tracking that SAME reference (deterministic env i -> ref i in both runs).
# Usage: build_crosscheck_report.sh <REF_S3_DIR> <POLICY_S3_DIR> [REPORT_S3]
set -euo pipefail
REF_DIR="${1:?s3 dir of reference-replay env*.mp4}"
POL_DIR="${2:?s3 dir of policy-eval env*.mp4}"
REPORT_S3="${3:-s3://wirobotics-internal/chrisryu/sim_rl/reports/simrl_tracking_crosscheck.html}"
CF="https://d1iitptfxhu64e.cloudfront.net"
BUCKET="wirobotics-internal"
[ "${REF_DIR%/}" = "$REF_DIR" ] && REF_DIR="$REF_DIR/"
[ "${POL_DIR%/}" = "$POL_DIR" ] && POL_DIR="$POL_DIR/"
REF_KEY="${REF_DIR#s3://$BUCKET/}"; POL_KEY="${POL_DIR#s3://$BUCKET/}"; REPORT_KEY="${REPORT_S3#s3://$BUCKET/}"

mapfile -t REFS < <(aws s3 ls "$REF_DIR" | grep -oE 'env[0-9A-Za-z_]+\.mp4' | sort -u)
mapfile -t POLS < <(aws s3 ls "$POL_DIR" | grep -oE 'env[0-9A-Za-z_]+\.mp4' | sort -u)
[ "${#REFS[@]}" -gt 0 ] && [ "${#POLS[@]}" -gt 0 ] || { echo "ERROR: missing clips (refs=${#REFS[@]} policy=${#POLS[@]})" >&2; exit 1; }
N=$(( ${#REFS[@]} < ${#POLS[@]} ? ${#REFS[@]} : ${#POLS[@]} ))

OUT="$(dirname "$0")/simrl_tracking_crosscheck.html"
{
  cat <<HEAD
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SimRL — Reference vs Policy cross-check</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0d1117; color:#e6edf3; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:28px 32px 8px; border-bottom:1px solid #21262d; }
  h1 { margin:0 0 6px; font-size:22px; }
  .sub { color:#9da7b3; max-width:78ch; }
  .meta { color:#6e7681; font-size:12.5px; margin-top:10px; }
  .meta code { background:#161b22; padding:1px 6px; border-radius:5px; color:#adbac7; }
  main { padding:22px 32px 48px; }
  .colhead { display:grid; grid-template-columns:44px 1fr 1fr; gap:14px; font-size:13px; color:#9da7b3;
             text-transform:uppercase; letter-spacing:.06em; margin:0 0 8px; }
  .row { display:grid; grid-template-columns:44px 1fr 1fr; gap:14px; align-items:center; margin-bottom:16px; }
  .idx { color:#6e7681; font-size:15px; text-align:center; }
  figure { margin:0; background:#161b22; border:1px solid #21262d; border-radius:10px; overflow:hidden; }
  video { width:100%; display:block; background:#000; aspect-ratio:1/1; object-fit:contain; }
  .spd { background:#161b22; color:#adbac7; border:1px solid #21262d; border-radius:6px; padding:3px 10px;
         margin-right:6px; cursor:pointer; font:12.5px inherit; }
  .spd.on { background:#1f6feb; color:#fff; border-color:#1f6feb; }
  .fstep { display:block; padding:4px 10px; border-top:1px solid #21262d; font-size:12px; color:#6e7681; }
  .fstep button { background:#0d1117; color:#adbac7; border:1px solid #21262d; border-radius:5px;
                  padding:1px 9px; margin-left:6px; cursor:pointer; font:12px inherit; }
</style>
<script>
function setRate(r){
  document.querySelectorAll('video').forEach(v=>{v.playbackRate=r; v.defaultPlaybackRate=r;});
  document.querySelectorAll('.spd').forEach(b=>b.classList.toggle('on', parseFloat(b.dataset.r)===r));
}
document.addEventListener('DOMContentLoaded', ()=>setRate(0.2));  // default: 5x slower than recorded
function stepF(b, s){  // pause + step one recorded frame (12 fps file)
  const v=b.closest('figure').querySelector('video'); v.pause(); v.currentTime=Math.max(0, v.currentTime + s/12);
}
</script>
</head><body>
<header>
  <h1>Reference vs Policy — WBT tracking cross-check</h1>
  <div class="sub">Row i pairs <b>Reference i</b> (the stored reference motion, kinematic replay — no dynamics)
  with the <b>1000-iter WBT policy</b> rolled out while tracking that same reference (deterministic
  env&nbsp;i&nbsp;→&nbsp;ref&nbsp;i; RSI to the same initial state; per-frame tracking-error overlay).</div>
  <div class="meta">reference <code>${REF_KEY}</code> · policy <code>${POL_KEY}</code> ·
    playback <button class="spd on" data-r="0.2" onclick="setRate(0.2)">0.2× (slow)</button><button class="spd" data-r="0.5" onclick="setRate(0.5)">0.5×</button><button class="spd" data-r="1" onclick="setRate(1)">1×</button></div>
</header><main>
<div class="colhead"><div></div><div>Reference (target motion)</div><div>Policy @1000 iters (overlay: track score + hammer error)</div></div>
HEAD
  for ((i=0; i<N; i++)); do
    echo "<div class=\"row\"><div class=\"idx\">$((i+1))</div>"
    echo "<figure><video controls muted loop autoplay playsinline preload=\"auto\" src=\"$CF/$REF_KEY${REFS[$i]}\"></video><span class=\"fstep\">frame<button onclick=\"stepF(this,-1)\">&#x276E;</button><button onclick=\"stepF(this,1)\">&#x276F;</button></span></figure>"
    echo "<figure><video controls muted loop autoplay playsinline preload=\"auto\" src=\"$CF/$POL_KEY${POLS[$i]}\"></video><span class=\"fstep\">frame<button onclick=\"stepF(this,-1)\">&#x276E;</button><button onclick=\"stepF(this,1)\">&#x276F;</button></span></figure>"
    echo "</div>"
  done
  echo "</main></body></html>"
} > "$OUT"

aws s3 cp "$OUT" "$REPORT_S3" --content-type "text/html" >/dev/null
echo "CROSSCHECK_OK rows=$N url=$CF/$REPORT_KEY"
