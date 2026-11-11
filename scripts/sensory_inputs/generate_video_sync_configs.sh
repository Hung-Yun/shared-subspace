#!/usr/bin/env bash

# Generate one video-sync YAML config per eligible row in sessions.csv.
# Run this on the server that contains the stitched NSP and camera directories.

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

sessions_csv="${SESSIONS_CSV:-${script_dir}/sessions.csv}"
source_root="${EMU_SOURCE_ROOT:-${HOME}/wrangell/stitched/EMU-18112}"
camera_root="${CAMERA_SOURCE_ROOT:-${HOME}/wrangell/datalake/emu}"
video_output_root="${VIDEO_OUTPUT_ROOT:-${HOME}/wrangell/hungyunlu/videos}"
config_output_dir="${CONFIG_OUTPUT_DIR:-${script_dir}/configs}"

dry_run=0
force=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--dry-run] [--force]

Generate one full-body camera (18486638 / B1) config per session, excluding
patients YEX and YEW.

Options:
  --dry-run  Validate folder matches and show the configs that would be made.
  --force    Replace config files that already exist.
  -h, --help Show this help message.

Optional environment variables:
  SESSIONS_CSV        Sessions CSV path (default: ${sessions_csv})
  EMU_SOURCE_ROOT     Stitched NSP root (default: ${source_root})
  CAMERA_SOURCE_ROOT  Camera data root (default: ${camera_root})
  VIDEO_OUTPUT_ROOT   Synced-video output root (default: ${video_output_root})
  CONFIG_OUTPUT_DIR   Generated-config directory (default: ${config_output_dir})
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            ;;
        --force)
            force=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ! -f "$sessions_csv" ]]; then
    printf 'ERROR: Sessions CSV does not exist: %s\n' "$sessions_csv" >&2
    exit 1
fi

if [[ ! -d "$source_root" ]]; then
    printf 'ERROR: Stitched NSP root does not exist: %s\n' "$source_root" >&2
    exit 1
fi

if [[ ! -d "$camera_root" ]]; then
    printf 'ERROR: Camera root does not exist: %s\n' "$camera_root" >&2
    exit 1
fi

if [[ "$dry_run" -eq 0 ]] && ! mkdir -p -- "$config_output_dir"; then
    printf 'ERROR: Could not create config directory: %s\n' "$config_output_dir" >&2
    exit 1
fi

eligible=0
generated=0
skipped_patient=0
skipped_existing=0
failures=0

while IFS=',' read -r patient emu task _date _time _ranking; do
    # Accommodate CSV files with Windows line endings.
    patient=${patient//$'\r'/}
    emu=${emu//$'\r'/}
    task=${task//$'\r'/}

    [[ -z "$patient" ]] && continue
    [[ "$patient" == "Patient" ]] && continue

    if [[ "$patient" == "YEX" || "$patient" == "YEW" ]]; then
        ((skipped_patient += 1))
        continue
    fi

    ((eligible += 1))

    if [[ ! "$emu" =~ ^[0-9]+$ ]]; then
        printf 'ERROR: %s has a non-numeric EMU value: %s\n' "$patient" "$emu" >&2
        ((failures += 1))
        continue
    fi

    if [[ -z "$task" ]]; then
        printf 'ERROR: %s EMU %s has an empty task name.\n' "$patient" "$emu" >&2
        ((failures += 1))
        continue
    fi

    printf -v emu_padded '%04d' "$((10#$emu))"
    task_lower=$(printf '%s' "$task" | tr '[:upper:]' '[:lower:]')

    source_pattern="${source_root}/${patient}/EMU-${emu_padded}_*"
    source_dirs=()
    while IFS= read -r candidate; do
        [[ -d "$candidate" ]] && source_dirs+=("$candidate")
    done < <(compgen -G "$source_pattern" || true)

    session_name="${patient}_${emu_padded}_${task_lower}"

    if [[ "${#source_dirs[@]}" -ne 1 ]]; then
        printf 'ERROR: %s: expected one NSP folder matching %s; found %d\n' \
            "$session_name" "$source_pattern" "${#source_dirs[@]}" >&2
        ((failures += 1))
        continue
    fi

    camera_dir="${camera_root}/${patient}Datafile/VIDEO"
    if [[ ! -d "$camera_dir" ]]; then
        printf 'ERROR: %s: camera directory does not exist: %s\n' \
            "$session_name" "$camera_dir" >&2
        ((failures += 1))
        continue
    fi

    config_path="${config_output_dir}/${session_name}.yaml"
    output_dir="${video_output_root}/${session_name}"

    if [[ "$dry_run" -eq 1 ]]; then
        printf 'DRY RUN: %s\n' "$config_path"
        printf '         nsp_dir: %s\n' "${source_dirs[0]}"
        printf '         cam_recording_dir: %s\n' "$camera_dir"
        printf '         output_dir: %s\n' "$output_dir"
        ((generated += 1))
        continue
    fi

    if [[ -e "$config_path" && "$force" -eq 0 ]]; then
        printf 'SKIP: Config already exists (use --force to replace): %s\n' \
            "$config_path"
        ((skipped_existing += 1))
        continue
    fi

    temporary_path=$(mktemp "${config_path}.tmp.XXXXXX") || {
        printf 'ERROR: Could not create a temporary file for %s\n' "$config_path" >&2
        ((failures += 1))
        continue
    }

    {
        printf '# Generated from sessions.csv for %s\n' "$session_name"
        printf 'cam_serial: ["18486638"]\n\n'
        printf 'nsp_dir: "%s"\n' "${source_dirs[0]}"
        printf 'cam_recording_dir: "%s"\n\n' "$camera_dir"
        printf 'output_dir: "%s"\n\n' "$output_dir"
        printf 'channel_name: "RoomMic2"\n'
    } > "$temporary_path"

    if mv -- "$temporary_path" "$config_path"; then
        printf 'GENERATED: %s\n' "$config_path"
        ((generated += 1))
    else
        printf 'ERROR: Could not write config: %s\n' "$config_path" >&2
        ((failures += 1))
    fi
done < "$sessions_csv"

printf '\nSummary: %d eligible; %d %s; %d excluded (YEX/YEW); %d existing; %d failed.\n' \
    "$eligible" "$generated" \
    "$([[ "$dry_run" -eq 1 ]] && printf 'validated' || printf 'generated')" \
    "$skipped_patient" "$skipped_existing" "$failures"

if [[ "$failures" -ne 0 ]]; then
    exit 1
fi
