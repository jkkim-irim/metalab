#!/usr/bin/env bash
# Reproducible build of the canonical 132-D bolt-drilling LeRobot dataset.
#
# Output: v3_132d_20260510_12_382ep  (382 episodes / 416,936 frames, LeRobot v3.0, state 132-D
# = q|dq|tau). This reproduces hansol's combined baseline dataset (his private
# v3_132d_20260513_382ep at /home/hansol/pippo/... which was deleted/lost), used by the
# reference run 5m3dzdwe.
#
# Source episodes: capture dates 20260510 (nice 253 + middle 12) + 20260512 (nice 113 + middle 4)
# = 382 eps, OLD 2-camera setup. Raw per-episode arrays (state/action/velocity/torque) + JPEG
# frames. The raw tarballs originate from the fellowship dataset bucket; the build node has no
# GetObject there, so they were bridged into wirobotics-internal first (see SRC below).
#
# Run on a build node (g6e) with the LeRobot env + this directory's converter.
exec > /work/canonical_build.log 2>&1
set -x
date -u

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC=s3://wirobotics-internal/chrisryu/_dlcmp_tmp/canonical_tars
W=/opt/dlami/nvme/build_canonical
rm -rf "$W"; mkdir -p "$W/tars" "$W/extracted" "$W/raw" "$W/lerobot"

# 1) download bridged tarballs (abort unless all 4 present)
for k in nice_20260510 middle_20260510 nice_20260512 middle_20260512; do
  aws s3 cp "$SRC/$k.tar.gz" "$W/tars/$k.tar.gz" --only-show-errors
done
ndl=$(ls "$W"/tars/*.tar.gz 2>/dev/null | wc -l)
echo DOWNLOADED_TARBALLS=$ndl
[ "$ndl" -eq 4 ] || { echo ABORT_DOWNLOAD_INCOMPLETE; exit 1; }

# 2) extract
for k in nice_20260510 middle_20260510 nice_20260512 middle_20260512; do
  mkdir -p "$W/extracted/$k"
  tar -I pigz -xf "$W/tars/$k.tar.gz" -C "$W/extracted/$k"
done

# 3) combine into one sequentially-numbered raw dir (preserve nice->middle, date order)
i=0
for k in nice_20260510 middle_20260510 nice_20260512 middle_20260512; do
  for ep in $(find "$W/extracted/$k" -maxdepth 2 -type d -name 'episode_*' | sort); do
    ln -s "$ep" "$W/raw/$(printf 'episode_%06d' "$i")"; i=$((i+1))
  done
done
echo COMBINE_DONE total_episodes=$i
[ "$i" -gt 0 ] || { echo ABORT_NO_EPISODES; exit 1; }

# 4) convert -> 132-D LeRobot v3.0 (state = pos|vel|torque via the two include flags)
OUT="$W/lerobot/v3_132d_20260510_12_382ep"
python3 "$HERE/convert_raw_to_v3_state_variant.py" -i "$W/raw" -o "$OUT" \
  --fps 30 --task 'pick up the drill and tighten the bolts' --include-velocity --include-torque
echo CONVERT_STATUS=$?

# 5) verify episode/frame counts (expect 382 / ~416,939)
EP=$(python3 -c "import json;print(json.load(open('$OUT/meta/info.json'))['total_episodes'])")
FR=$(python3 -c "import json;print(json.load(open('$OUT/meta/info.json'))['total_frames'])")
echo VERIFY total_episodes=$EP total_frames=$FR expected_ep=382

# NOTE: the converter writes state/action stats but NOT image stats. Before training, patch
# meta/stats.json with ImageNet mean [0.485,0.456,0.406] / std [0.229,0.224,0.225] (shape (3,1,1))
# for each observation.images.* key, else lerobot make_dataset raises KeyError under VISUAL norm.

# 6) upload ONLY if episode count is exactly 382 (guard against a bad partial upload)
DST=s3://wirobotics-internal/chrisryu/datasets/allex/lerobot/bolt_drilling/v3_132d_20260510_12_382ep
if [ "$EP" = "382" ]; then
  aws s3 sync "$OUT" "$DST/" --only-show-errors
  echo UPLOAD_STATUS=$? DST="$DST"
else
  echo SKIP_UPLOAD got_episodes=$EP expected=382 -- inspect before uploading
fi
date -u
echo ALL_DONE
