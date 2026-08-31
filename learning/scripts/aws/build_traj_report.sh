#!/usr/bin/env bash
# Build an HTML video-report from per-env MP4s already on S3 (served via CloudFront) and upload the page
# back to S3. Usage:
#   build_traj_report.sh <VIDEO_S3_DIR> [REPORT_S3] [TITLE] [SUBTITLE]
# VIDEO_S3_DIR: s3://wirobotics-internal/.../<dir>/  (holds env*.mp4)
set -euo pipefail
VIDEO_S3_DIR="${1:?s3://wirobotics-internal/.../video_dir/}"
REPORT_S3="${2:-s3://wirobotics-internal/chrisryu/sim_rl/reports/simrl_tracking_trajectories.html}"
TITLE="${3:-SimRL — WBT Reference Trajectories}"
SUBTITLE="${4:-Reference hammer-lift trajectories the whole-body-tracking policy learns to reproduce.}"
CF="https://d1iitptfxhu64e.cloudfront.net"
BUCKET="wirobotics-internal"
[ "${VIDEO_S3_DIR%/}" = "$VIDEO_S3_DIR" ] && VIDEO_S3_DIR="$VIDEO_S3_DIR/"

VDIR_KEY="${VIDEO_S3_DIR#s3://$BUCKET/}"
REPORT_KEY="${REPORT_S3#s3://$BUCKET/}"
mapfile -t MP4S < <(aws s3 ls "$VIDEO_S3_DIR" | grep -oE 'env[0-9A-Za-z_]+\.mp4' | sort -u)
[ "${#MP4S[@]}" -gt 0 ] || { echo "ERROR: no env*.mp4 under $VIDEO_S3_DIR" >&2; exit 1; }

OUT="$(dirname "$0")/simrl_tracking_trajectories.html"
{
  cat <<HEAD
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${TITLE}</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0d1117; color:#e6edf3; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:28px 32px 8px; border-bottom:1px solid #21262d; }
  h1 { margin:0 0 6px; font-size:22px; }
  .sub { color:#9da7b3; max-width:70ch; }
  .meta { color:#6e7681; font-size:12.5px; margin-top:10px; }
  .meta code { background:#161b22; padding:1px 6px; border-radius:5px; color:#adbac7; }
  main { padding:22px 32px 48px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:18px; }
  figure { margin:0; background:#161b22; border:1px solid #21262d; border-radius:10px; overflow:hidden; }
  video { width:100%; display:block; background:#000; aspect-ratio:1/1; object-fit:contain; }
  figcaption { padding:8px 12px; font-size:13px; color:#adbac7; border-top:1px solid #21262d; }
  .n { color:#6e7681; }
  details { background:#161b22; border:1px solid #21262d; border-radius:10px; padding:14px 18px; margin:0 0 22px; }
  summary { cursor:pointer; font-weight:600; color:#e6edf3; }
  details h3 { font-size:14px; margin:16px 0 6px; color:#adbac7; }
  details p, details li { color:#9da7b3; font-size:13.5px; }
  details table { border-collapse:collapse; font-size:12.5px; margin:6px 0; display:block; overflow-x:auto; }
  details th, details td { border:1px solid #21262d; padding:4px 10px; text-align:left; color:#adbac7; }
  details th { background:#0d1117; color:#9da7b3; }
  details code { background:#0d1117; padding:1px 5px; border-radius:4px; color:#adbac7; }
  .spd { background:#161b22; color:#adbac7; border:1px solid #21262d; border-radius:6px; padding:3px 10px;
         margin-right:6px; cursor:pointer; font:12.5px inherit; }
  .spd.on { background:#1f6feb; color:#fff; border-color:#1f6feb; }
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
  <h1>${TITLE}</h1>
  <div class="sub">${SUBTITLE}</div>
  <div class="meta">${#MP4S[@]} clips · source <code>${VDIR_KEY}</code> · generated on S3, served via CloudFront ·
    playback <button class="spd on" data-r="0.2" onclick="setRate(0.2)">0.2× (slow)</button><button class="spd" data-r="0.5" onclick="setRate(0.5)">0.5×</button><button class="spd" data-r="1" onclick="setRate(1)">1×</button></div>
</header><main>
<details open>
<summary>What goes into a reference trajectory</summary>
<h3>Provenance — how these motions were produced</h3>
<p>The source is the <b>full-curriculum dexblind hammer-lift RL policy</b> (blind/proprioceptive, LSTM actor;
checkpoint <code>hammer_lift_2026-06-30-12-50/model_1600</code>, eval SR&nbsp;<b>0.84</b> under the canonical gate), rolled
out over the sim service in the <b>WBT-collect</b> env variant (play regime: observation noise off, no external
force disturbance; per-episode reset randomization: hammer x/y/yaw, hammer mass, hammer/robot/table friction,
robot root height, joint offsets). Only episodes that claimed <code>task_success</code> under the <b>canonical
full-grasp gate</b> — all&nbsp;<b>5 fingertips</b> in contact + hammer lifted to the goal (pos&nbsp;&lt;&nbsp;0.2&nbsp;m,
rot&nbsp;&lt;&nbsp;0.5&nbsp;rad) + palm&nbsp;&lt;&nbsp;0.084&nbsp;m, all held <b>10 consecutive steps</b> — were kept
(1,160 episodes → 1,000 successes), then a quality pass: the first 2 frames are trimmed (stale post-reset
observations) and trajectories whose hammer rises &lt;8&nbsp;cm are dropped → <b>826 references</b>. Every reference
therefore demonstrates a sustained full 5-finger grasp by construction (826/826 verified).</p>
<h3>Per-file contents (<code>ref_NNNN.npz</code>)</h3>
<table>
<tr><th>array</th><th>shape</th><th>what it is</th></tr>
<tr><td><code>state</code></td><td>[T, 59] @ 50 Hz</td><td>the tracked state per control frame — layout below; all quantities expressed relative to the robot's <code>Chest_Origin_Link</code> frame (quaternions xyzw)</td></tr>
<tr><td><code>contact</code></td><td>[T, 5]</td><td>fingertip↔hammer contact flags (Index, Middle, Ring, Little, Thumb), sliced from the recorded privileged obs — the grasp-pattern target</td></tr>
<tr><td><code>setup</code></td><td>[84]</td><td>the episode's exact initial env state for Reference State Initialization: hammer pose (7) + velocity (6), all 33 robot joints pos+vel (66), robot root z (1), hammer mass (1), hammer/robot/table friction (3)</td></tr>
<tr><td><code>action</code></td><td>[T, 18]</td><td>the source policy's recorded actions (provenance + replay tooling; not a training target)</td></tr>
</table>
<h3><code>state</code> layout (59 dims)</h3>
<table>
<tr><th>dims</th><th>field</th></tr>
<tr><td>0:3 / 3:7</td><td>hammer position / orientation (xyzw)</td></tr>
<tr><td>7:10 / 10:13</td><td>hammer linear / angular velocity</td></tr>
<tr><td>13:16 / 16:20</td><td>palm position / orientation</td></tr>
<tr><td>20:23 / 23:26</td><td>palm linear / angular velocity</td></tr>
<tr><td>26:41</td><td>5 fingertip positions (3 each, finger-index order)</td></tr>
<tr><td>41:59</td><td>18 right-arm + hand joint positions</td></tr>
</table>
<h3>Reward breakdown — how the tracker is scored against these</h3>
<p>The tracking reward is a <b>flat, object-dominant sum</b>:
<code>r = 1.5·k_obj_pos + 1.0·k_obj_ori|grip + 1.0·k_c + 0.5·k_hold + 0.5·k_kp + 0.5·k_joint</code> — kernels 0..1, max 5.
(A hierarchical variant that multiplied the object terms by grip quality <code>k_c</code> was measured <b>worse</b>
on eval success across two independent runs — pre-grasp it zeroes the object gradient — and was reverted; the
mime-without-grasping failure it targeted is instead closed by the success gate's full 5-finger contact
condition.) Each kernel logs its own <code>Episode_Reward/wbt_*</code> curve in training; evals report the
trajectory <b>mean</b> per kernel (fidelity)
and the <b>final-frame</b> hammer error with <i>track&nbsp;success&nbsp;=&nbsp;final&nbsp;hammer&nbsp;error&nbsp;&lt;&nbsp;5&nbsp;cm</i> (the lift outcome).</p>
<table>
<tr><th>kernel</th><th>weight</th><th>σ</th><th>error it measures (vs the reference at the current frame)</th></tr>
<tr><td><code>wbt_obj_pos</code></td><td>1.5</td><td>0.08 m</td><td>hammer position — the dominant term; the hammer must actually follow the lift</td></tr>
<tr><td><code>wbt_obj_ori</code></td><td>1.0</td><td>0.2 rad</td><td>hammer orientation — scored only while the reference grips (constant 1.0 pre-grasp); sharpened after the in-grip tilt autopsy</td></tr>
<tr><td><code>wbt_contact</code> (k_c)</td><td>1.0</td><td>—</td><td><b>per-finger</b> match fraction of the reference contact pattern (<code>contact[T,5]</code> above): a 3/5-finger hold scores 0.6, so partial grips get gradient toward full ones</td></tr>
<tr><td><code>wbt_hold</code></td><td>0.5</td><td>—</td><td>sustained grasp: <code>min(consecutive 5/5-contact steps, 10)/10</code> inside the reference's own ≥10-frame hold window (neutral outside) — one flicker forfeits the ramp</td></tr>
<tr><td><code>wbt_keypoint</code></td><td>0.5</td><td>0.1 m</td><td>palm + 5 fingertip task-space positions (mean over the 6 keypoints)</td></tr>
<tr><td><code>wbt_joint</code></td><td>0.5</td><td>0.5 rad</td><td>18 arm+hand joint angles (RMS)</td></tr>
</table>
<p>Every revision of this reward was forced by a measured failure — mime-without-touching (→ object-dominant,
task-scaled σ), 80% static-tail episodes (→ end at reference end), free orientation credit (→ grip windowing),
starved partial grips (→ per-finger match), and a hierarchical grip-gate tried against desk-parked policies then
reverted on measured eval-SR regression. The full revision-by-revision justification table lives in
<code>learning/rl/tracking/README.md</code>.</p>
<h3>How to read the tracking curves (drawn on the videos)</h3>
<p>The six colored curves are the <b>reward terms</b> — each drawn as its earned fraction (0..1 of its
weight) at every frame: the raw kernels, with <code>obj_ori</code> scored only inside the reference's grip window
(constant 1.0 outside it) — the curves show what is being <i>earned</i>, mirroring the
<code>Episode_Reward/wbt_*</code>
training panels. <b>Every term compares against the reference's recorded value at the same frame t</b> — never a
fixed goal; perfect tracking = pinned to the reference timeline in all terms at once (these reference replays
read ≈1.0 everywhere by construction). Two readings that trip people up:</p>
<ul>
<li><b><code>contact</code> starts at 1.0 before anything touches</b> — it is a contact-pattern <i>match</i>:
early in the episode the reference hand isn't touching either, so no-touch vs no-touch = perfect match. It
drops exactly where the reference closes its grip and the policy doesn't follow — that drop marks the
reference's grip onset.</li>
<li><b><code>obj_pos</code>/<code>obj_ori</code> decay even if the hammer is never touched</b> — the live
hammer is static, but the <i>reference target</i> rises through the lift, so the error grows as the target
climbs away. The measurement frame (robot chest) is fixed; nothing is measured relative to the palm.</li>
</ul>
<p>The orange <code>hold</code> curve is the sustained-grasp ramp (consecutive full-contact steps ÷ 10,
scored inside the reference's own hold window): a clean climb to 1.0 is a stable grip; a sawtooth is
contact flicker — the dominant hold-breaker — made visible.</p>
<p>Curves are drawn up to the current frame (current value = right edge) — pause/scrub to read instant +
history. In training logs, <code>Episode_Reward/wbt_*</code> is a time-normalized rate (episodic sum ÷ 5 s
max-episode-length), so with ~1 s episodes each term's ceiling ≈ weight × 0.23 — for true kernel means (0..1)
read the eval <code>WBT_BREAKDOWN</code>. Evals report per-kernel <b>means</b> (tracking fidelity) and
<b>final-frame</b> hammer error with <i>track success = final error &lt; 5 cm</i> (lift outcome).</p>
<p><b>Reference State Initialization (RSI):</b> every episode starts in the reference's exact recorded
initial state (hammer pose+vel, joints, root height, mass, frictions), so the motion is reachable.
<b>Phase-RSI</b> extends this: a fraction of training episodes start at a uniformly sampled phase t of the
reference — including mid/post-grasp frames with the hammer already in the hand — so the policy experiences
the grasped state directly instead of having to discover grasp closure by exploration (evals and these
replays always start at t=0).</p>
<h3>How the videos are rendered</h3>
<p><b>Kinematic replay</b> — the env is RSI'd to the reference's recorded <code>setup</code>, then each frame the recorded
hammer pose + joint positions are written into the sim and forward kinematics recomputed; <b>no physics stepping</b>, so
this is the exact stored motion with zero drift. Recorded every control frame (50 Hz) into a 12 fps file; the page plays
it at <b>0.2× by default</b> (≈ 21× slower than real time — humanly interpretable; switch with the speed buttons above).</p>
<h3>Dataset stats &amp; consumers</h3>
<p>826 references (174 filtered &lt;8 cm; 2-frame head trim) · length 53–245 frames (1.1–4.9 s) · hammer lift ≥8 cm
(mean 11.0) · every reference holds a full 5-finger grasp for ≥10 consecutive frames (826/826). The WBT tracking
policy consumes these as: goal obs = <code>state</code>+<code>contact</code>+phase at the current
frame; the reward's Gaussian kernels compare the live sim against the same quantities; <code>setup</code> drives RSI at reset.
The grid below shows references 1–${#MP4S[@]} of the set.</p>
</details>
<div class="grid">
HEAD
  i=0
  for m in "${MP4S[@]}"; do
    i=$((i+1))
    echo "<figure><video controls muted loop autoplay playsinline preload=\"auto\" src=\"$CF/$VDIR_KEY$m\"></video><figcaption>Reference $i<span class=\"fstep\" style=\"float:right\">frame<button onclick=\"stepF(this,-1)\">&#x276E;</button><button onclick=\"stepF(this,1)\">&#x276F;</button></span></figcaption></figure>"
  done
  cat <<'FOOT'
</div></main></body></html>
FOOT
} > "$OUT"

aws s3 cp "$OUT" "$REPORT_S3" --content-type "text/html" >/dev/null
echo "REPORT_OK clips=${#MP4S[@]} url=$CF/$REPORT_KEY"
