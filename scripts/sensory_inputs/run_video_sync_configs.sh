#!/usr/bin/env bash

# Run video-sync once for every generated YAML config.
# Activate the videosync Conda environment before starting this script.

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_dir="${CONFIG_DIR:-${script_dir}/configs}"
python_bin="${PYTHON_BIN:-python}"

if [[ -d "${script_dir}/video-sync/pyvideosync" ]]; then
    default_project_dir="${script_dir}/video-sync"
else
    default_project_dir="${script_dir}"
fi
project_dir="${VIDEO_SYNC_DIR:-${default_project_dir}}"

dry_run=0
stop_on_error=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--dry-run] [--stop-on-error]

Run video-sync for every .yaml or .yml file in:
  ${config_dir}

Options:
  --dry-run        Show commands without running video-sync.
  --stop-on-error  Stop immediately if a config fails. By default, the next
                   config is attempted and all failures are summarized.
  -h, --help       Show this help message.

Optional environment variables:
  CONFIG_DIR  Config directory to process (default: ${config_dir})
  PYTHON_BIN  Python executable to use (default: ${python_bin})
  VIDEO_SYNC_DIR  video-sync repository (default: ${project_dir})

Activate the videosync Conda environment before running this script:
  conda activate videosync
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            ;;
        --stop-on-error)
            stop_on_error=1
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

if [[ ! -d "$config_dir" ]]; then
    printf 'ERROR: Config directory does not exist: %s\n' "$config_dir" >&2
    printf 'Run ./generate_video_sync_configs.sh first.\n' >&2
    exit 1
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'ERROR: Python executable not found: %s\n' "$python_bin" >&2
    printf 'Activate the videosync Conda environment first.\n' >&2
    exit 1
fi

if [[ ! -f "${project_dir}/pyvideosync/main.py" ]]; then
    printf 'ERROR: video-sync repository not found at: %s\n' "$project_dir" >&2
    printf 'Set VIDEO_SYNC_DIR to the repository path.\n' >&2
    exit 1
fi

shopt -s nullglob
config_files=("$config_dir"/*.yaml "$config_dir"/*.yml)
shopt -u nullglob

if [[ "${#config_files[@]}" -eq 0 ]]; then
    printf 'ERROR: No .yaml or .yml files found in: %s\n' "$config_dir" >&2
    exit 1
fi

printf 'Python: %s\n' "$(command -v "$python_bin")"
printf 'Repository: %s\n' "$project_dir"
printf 'Configs: %s\n' "$config_dir"
printf 'Sessions: %d\n\n' "${#config_files[@]}"

succeeded=0
failed=0
failure_names=()
batch_start=$SECONDS

for index in "${!config_files[@]}"; do
    config_path=${config_files[$index]}
    config_name=$(basename "$config_path")
    display_index=$((index + 1))

    printf '[%d/%d] %s\n' "$display_index" "${#config_files[@]}" "$config_name"

    if [[ "$dry_run" -eq 1 ]]; then
        printf 'DRY RUN: %q -m pyvideosync.main --config %q\n\n' \
            "$python_bin" "$config_path"
        ((succeeded += 1))
        continue
    fi

    session_start=$SECONDS
    if (
        cd -- "$project_dir" || exit 1
        "$python_bin" -m pyvideosync.main --config "$config_path"
    ); then
        printf 'SUCCESS: %s (%ds)\n\n' \
            "$config_name" "$((SECONDS - session_start))"
        ((succeeded += 1))
    else
        exit_code=$?
        printf 'FAILED: %s (exit %d, %ds)\n\n' \
            "$config_name" "$exit_code" "$((SECONDS - session_start))" >&2
        ((failed += 1))
        failure_names+=("$config_name")

        if [[ "$stop_on_error" -eq 1 ]]; then
            break
        fi
    fi
done

if [[ "$dry_run" -eq 1 ]]; then
    printf 'Dry run complete: %d command(s) validated.\n' "$succeeded"
    exit 0
fi

printf 'Batch complete in %ds: %d succeeded; %d failed.\n' \
    "$((SECONDS - batch_start))" "$succeeded" "$failed"

if [[ "$failed" -ne 0 ]]; then
    printf 'Failed configs:\n' >&2
    for config_name in "${failure_names[@]}"; do
        printf '  - %s\n' "$config_name" >&2
    done
    exit 1
fi
