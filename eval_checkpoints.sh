#!/bin/bash
# Sweep audio_chkpnt*.pth for one or more RWAVS scenes and report MAG/ENV.
#
# Usage:
#   bash eval_checkpoints.sh 6
#   bash eval_checkpoints.sh 6 10 12
#   SCENES="1 2 3 4 5" bash eval_checkpoints.sh
#   HIP_VISIBLE_DEVICES=1 bash eval_checkpoints.sh 6
#   AV_RESOLUTION="64 180" bash eval_checkpoints.sh 6   # must match training resolution
#
# RWAVS category mapping (AV-NeRF eval.py):
#   Office: 1-5 | House: 6-8 | Apt.: 9-11 | Out.: 12-13

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${DATA_DIR:-/workspace/data/release}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/SAVAF-AV/output}"
GPU="${HIP_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${GPU}"
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
read -r AV_RES_H AV_RES_W <<< "${AV_RESOLUTION:-64 180}"

if [[ $# -gt 0 ]]; then
    read -ra SCENE_LIST <<< "$*"
elif [[ -n "${SCENES:-}" ]]; then
    read -ra SCENE_LIST <<< "${SCENES}"
else
    echo "Usage: bash eval_checkpoints.sh <scene> [scene2 ...]"
    echo "   or: SCENES=\"6 10 12\" bash eval_checkpoints.sh"
    exit 1
fi

RESULTS_TSV="$(mktemp)"
trap 'rm -f "${RESULTS_TSV}"' EXIT

parse_eval_line() {
    # stdout is silenced when train_av.py is passed --quiet, so the caller
    # must leave stdout on. env is a Python float; mag is often np.float32.
    printf '%s' "$1" | python3 -c '
import re, sys
text = sys.stdin.read()
matches = re.findall(
    r"'\''env'\'': (?:np\.float\d+\()?([^,)]+)\)?, '\''mag'\'': (?:np\.float\d+\()?([^,)]+)\)?",
    text,
)
if not matches:
    sys.exit(1)
env, mag = matches[-1]
print(f"{float(env):.6f},{float(mag):.6f}")
'
}

scene_category() {
    local scene="$1"
    if [[ "${scene}" -le 5 ]]; then echo "Office"
    elif [[ "${scene}" -le 8 ]]; then echo "House"
    elif [[ "${scene}" -le 11 ]]; then echo "Apt."
    else echo "Out."
    fi
}

echo "==> Evaluating audio checkpoints"
echo "    Scenes : ${SCENE_LIST[*]}"
echo "    GPU    : ${HIP_VISIBLE_DEVICES}"
echo "    AV res : ${AV_RES_H}x${AV_RES_W}"
echo "    Output : ${OUTPUT_DIR}"
echo ""

for scene in "${SCENE_LIST[@]}"; do
    scene_dir="${OUTPUT_DIR}/${scene}"
    if [[ ! -d "${scene_dir}" ]]; then
        echo "[skip] scene ${scene}: directory not found (${scene_dir})"
        continue
    fi

    shopt -s nullglob
    ckpts=( "${scene_dir}"/audio_chkpnt*.pth )
    shopt -u nullglob

    if [[ ${#ckpts[@]} -eq 0 ]]; then
        echo "[skip] scene ${scene}: no audio_chkpnt*.pth in ${scene_dir}"
        continue
    fi

    mapfile -t ckpts_sorted < <(printf '%s\n' "${ckpts[@]}" | sort -V)
    category="$(scene_category "${scene}")"

    echo "---- Scene ${scene} (${category}) ----"
    printf "  %-10s  %8s  %8s\n" "checkpoint" "ENV ↓" "MAG ↓"

    for ckpt in "${ckpts_sorted[@]}"; do
        ckpt_name="$(basename "${ckpt}")"
        iter="${ckpt_name#audio_chkpnt}"
        iter="${iter%.pth}"
        port=$((6099 + scene))

        echo "  [eval] scene=${scene} ckpt=${ckpt_name} ..."
        log="$(python train_av.py \
            -s "${DATA_DIR}/${scene}" \
            -m "${scene_dir}" \
            --eval \
            --start_checkpoint "${scene_dir}" \
            --checkpoint_path "${scene_dir}" \
            --av_checkpoint_path "${ckpt}" \
            --av-resolution "${AV_RES_H}" "${AV_RES_W}" \
            --eval_aud \
            --port "${port}" \
            2>&1)" || {
            echo "  [fail] ${ckpt_name}: train_av.py exited with error"
            tail -n 8 <<< "${log}" | sed 's/^/    /'
            continue
        }

        if ! metrics="$(parse_eval_line "${log}")"; then
            echo "  [fail] ${ckpt_name}: could not parse eval metrics from log"
            echo "  [debug] last log lines:"
            tail -n 5 <<< "${log}" | sed 's/^/    /'
            continue
        fi

        env="${metrics%%,*}"
        mag="${metrics##*,}"
        printf "  %-10s  %8.3f  %8.3f\n" "${ckpt_name}" "${env}" "${mag}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${scene}" "${category}" "${iter}" "${env}" "${mag}" "${ckpt}" >> "${RESULTS_TSV}"
    done
    echo ""
done

if [[ ! -s "${RESULTS_TSV}" ]]; then
    echo "No successful evaluations."
    exit 1
fi

python3 - "${RESULTS_TSV}" << 'PY'
import sys
from collections import defaultdict

path = sys.argv[1]
rows = []
with open(path) as f:
    for line in f:
        scene, category, iteration, env, mag, ckpt = line.rstrip("\n").split("\t")
        rows.append({
            "scene": int(scene),
            "category": category,
            "iteration": int(iteration),
            "env": float(env),
            "mag": float(mag),
            "ckpt": ckpt,
        })

print("==> Best checkpoint per scene (min ENV, then MAG — same as AV-NeRF eval.py --best)")
print(f"{'Scene':>5}  {'Cat.':<7}  {'Best ckpt':<16}  {'ENV ↓':>8}  {'MAG ↓':>8}")
best_by_scene = {}
for scene in sorted({r["scene"] for r in rows}):
    scene_rows = [r for r in rows if r["scene"] == scene]
    best = min(scene_rows, key=lambda r: (r["env"], r["mag"]))
    best_by_scene[scene] = best
    print(f"{scene:5d}  {best['category']:<7}  audio_chkpnt{best['iteration']:<6d}  "
          f"{best['env']:8.3f}  {best['mag']:8.3f}")

# Category averages using best-per-scene
cats = ["Office", "House", "Apt.", "Out."]
cat_rows = defaultdict(list)
for best in best_by_scene.values():
    cat_rows[best["category"]].append(best)

# Published SAVAF category numbers, stored as (MAG, ENV).
baseline = {
    "Office": (0.73, 0.12),
    "House": (1.85, 0.15),
    "Apt.": (2.04, 0.17),
    "Out.": (0.72, 0.11),
    "Overall": (1.35, 0.14),
}

print()
print("==> RWAVS Scene Categories (best checkpoint per evaluated scene)")
print(f"{'Category':<10}  {'Scenes':<12}  {'ENV ↓':>8}  {'MAG ↓':>8}    ref ENV / MAG")
for cat in cats:
    items = cat_rows.get(cat, [])
    if not items:
        print(f"{cat:<10}  {'—':<12}  {'—':>8}  {'—':>8}    "
              f"{baseline[cat][1]:.2f} / {baseline[cat][0]:.2f}")
        continue
    scenes = ",".join(str(r["scene"]) for r in sorted(items, key=lambda r: r["scene"]))
    env = sum(r["env"] for r in items) / len(items)
    mag = sum(r["mag"] for r in items) / len(items)
    print(f"{cat:<10}  {scenes:<12}  {env:8.3f}  {mag:8.3f}    "
          f"{baseline[cat][1]:.2f} / {baseline[cat][0]:.2f}")

all_best = list(best_by_scene.values())
if all_best:
    env = sum(r["env"] for r in all_best) / len(all_best)
    mag = sum(r["mag"] for r in all_best) / len(all_best)
    print(f"{'Overall':<10}  {f'n={len(all_best)}':<12}  {env:8.3f}  {mag:8.3f}    "
          f"{baseline['Overall'][1]:.2f} / {baseline['Overall'][0]:.2f}")
PY
