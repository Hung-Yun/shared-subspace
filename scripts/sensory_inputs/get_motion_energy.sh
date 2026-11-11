#!/usr/bin/env bash

# Fig. S11–S12 input: sum of dense optical-flow magnitudes over the whole frame.
# Input: videos/<session>/18486638/stitched_18486638.mp4 (30 fps).
# Output: energy/<session>_motion_energy.csv (one value per frame, no header).
# Existing outputs are retained. The processing record explains the session skip.

set -uo pipefail
shopt -s nullglob

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
video_root="${VIDEO_OUTPUT_ROOT:-${script_dir}/videos}"
energy_dir="${ENERGY_DIR:-${script_dir}/energy}"
camera_serial="18486638"
excluded_session="YFB_0044_pursuit"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${HOME}/.venvs/motion-energy}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

dry_run=0
case "${1:-}" in
    --dry-run) dry_run=1 ;;
    -h|--help)
        printf 'Usage: bash %s [--dry-run]\n' "$(basename "$0")"
        printf 'Extract full-frame Farneback motion energy; retain existing CSVs.\n'
        printf 'Set VIDEO_OUTPUT_ROOT for videos and ENERGY_DIR for output CSVs.\n'
        printf 'Defaults: %s and %s\n' "$video_root" "$energy_dir"
        exit 0 ;;
    '') ;;
    *) printf 'ERROR: Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
if [[ "$#" -gt 1 ]]; then
    printf 'ERROR: Expected at most one option.\n' >&2
    exit 2
fi
if [[ ! -d "$video_root" ]]; then
    printf 'ERROR: Video directory does not exist: %s\n' "$video_root" >&2
    exit 1
fi
if [[ "$dry_run" -eq 0 ]]; then
    mkdir -p -- "$energy_dir" || exit 1
fi

processed=0
existing=0
skipped=0
failed=0

for folder in "$video_root"/*; do
    [[ -d "$folder" ]] || continue
    name=$(basename "$folder")
    file="$folder/$camera_serial/stitched_${camera_serial}.mp4"
    output="$energy_dir/${name}_motion_energy.csv"

    if [[ "$name" == "$excluded_session" ]]; then
        printf 'SKIP %s: retained exclusion from the original batch.\n' "$name"
        ((skipped += 1))
    elif [[ -f "$output" ]]; then
        printf 'EXISTS %s\n' "$output"
        ((existing += 1))
    elif [[ ! -f "$file" ]]; then
        printf 'SKIP %s: expected video not found at %s\n' "$name" "$file"
        ((skipped += 1))
    elif [[ "$dry_run" -eq 1 ]]; then
        printf 'DRY RUN: %s -> %s\n' "$file" "$output"
        ((processed += 1))
    elif uv run --project "$script_dir" python "$script_dir/motion-energy/video_motion_analysis.py" \
        "$file" --method farneback --save-data "$output" --no-plot; then
        printf 'SAVED %s\n' "$output"
        ((processed += 1))
    else
        printf 'FAILED %s\n' "$name" >&2
        ((failed += 1))
    fi
done

printf 'Motion energy: %d %s; %d existing; %d skipped; %d failed.\n' \
    "$processed" "$([[ "$dry_run" -eq 1 ]] && printf 'planned' || printf 'saved')" \
    "$existing" "$skipped" "$failed"
if [[ "$failed" -ne 0 ]]; then
    exit 1
fi
