#!/usr/bin/env bash
# Unattended campaign for the suboptimal-feedback experiment.
#
# Usage:
#   bash scripts/run_subopt_campaign_unattended.sh [RUN_ID [PROB_GRID [NOISE_GRID]]]
#
# Arguments:
#   RUN_ID     Output directory tag (default: subopt_<timestamp>)
#   PROB_GRID  Comma-separated subopt_prob values for the one-swap sweep
#              (default: 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
#   NOISE_GRID Comma-separated subopt_noise_std values for the score-perturb sweep
#              (default: 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
#
# Outputs (under results/<RUN_ID>/ and figures/<RUN_ID>/):
#   results/<RUN_ID>/one_swap/          per-run seed_summary.csv
#   results/<RUN_ID>/score_perturb/     per-run seed_summary.csv
#   figures/<RUN_ID>/subopt_combined.png
#
# On macOS, wrap with caffeinate -dimsu to prevent sleep during long runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found at ${PYTHON_BIN}; create .venv or set PYTHON_BIN to an executable path." >&2
  exit 2
fi

RUN_ID="${1:-subopt_$(date +%Y%m%d_%H%M%S)}"
PROB_GRID="${2:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
NOISE_GRID="${3:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"

if (( $# > 3 )); then
  echo "Expected at most three positional arguments; see docs/REPRODUCTION.md." >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "RUN_ID must start with a letter or digit and contain only letters, digits, '.', '_' or '-'." >&2
  exit 2
fi

BASE_RESULTS="results/${RUN_ID}"
BASE_FIGURES="figures/${RUN_ID}"
# Keep runtime comparable and reproducible across jobs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${BASE_RESULTS}/mplconfig"


# Check again before each job so one campaign cannot mix source versions.
check_campaign_identity() {
  "${PYTHON_BIN}" scripts/campaign_manifest.py --output-root "${BASE_RESULTS}" \
    feedback --prob-grid "${PROB_GRID}" --noise-grid "${NOISE_GRID}"
}
check_campaign_identity

LOCK_DIR="${BASE_RESULTS}/.campaign-lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Campaign lock exists: ${LOCK_DIR}. Check for an active process before removing a stale lock." >&2
  exit 2
fi
trap 'rmdir "${LOCK_DIR}"' EXIT

mkdir -p "${BASE_FIGURES}" "${BASE_RESULTS}/invocations"
INVOCATION_DIR="$(mktemp -d "${BASE_RESULTS}/invocations/$(date +%Y%m%d-%H%M%S).XXXXXX")"
LOG_DIR="${INVOCATION_DIR}/logs"
STATUS_FILE="${INVOCATION_DIR}/status.tsv"
mkdir -p "${LOG_DIR}" "${MPLCONFIGDIR}"
ln -sfn "invocations/$(basename "${INVOCATION_DIR}")/status.tsv" "${BASE_RESULTS}/campaign_status_latest.tsv"


{
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date +"%Y-%m-%dT%H:%M:%S%z")"
  echo "repo_root=${REPO_ROOT}"
  echo "cwd=$(pwd)"
  echo "python_path=${PYTHON_BIN}"
  echo "python_version=$("${PYTHON_BIN}" --version 2>&1)"
  echo "prob_grid=${PROB_GRID}"
  echo "noise_grid=${NOISE_GRID}"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
  echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
  echo "VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS}"
  echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
  echo "MPLCONFIGDIR=${MPLCONFIGDIR}"
} > "${INVOCATION_DIR}/environment_snapshot.txt"

ts_now() { date +"%Y-%m-%dT%H:%M:%S%z"; }

append_status() {
  local job="$1" status="$2" attempt="$3" message="$4"
  printf "%s\t%s\t%s\t%s\t%s\n" "$(ts_now)" "${job}" "${status}" "${attempt}" "${message}" \
    >> "${STATUS_FILE}"
}

printf "timestamp\tjob\tstatus\tattempt\tmessage\n" > "${STATUS_FILE}"
append_status "campaign" "start" "1" "run_id=${RUN_ID}"

run_job() {
  local job_name="$1"
  shift
  check_campaign_identity
  local log_file="${LOG_DIR}/${job_name}.log"
  append_status "${job_name}" "start" "1" "${log_file}"
  {
    echo "[launcher] ts=$(ts_now) job=${job_name}"
    printf '[launcher] command:'
    printf ' %q' "$@"
    printf '\n'
  } > "${log_file}"
  if "$@" >> "${log_file}" 2>&1; then
    append_status "${job_name}" "done" "1" "${log_file}"
  else
    local rc=$?
    append_status "${job_name}" "fail" "1" "exit_code=${rc} log=${log_file}"
    append_status "campaign" "failed" "1" "job=${job_name}"
    echo "Job failed; no automatic retry. See ${log_file}" >&2
    return "${rc}"
  fi
}

run_job \
  "subopt_combined" \
  "${PYTHON_BIN}" scripts/run_subopt_combined.py \
    --config configs/main_linear_t10000_s10.yaml \
    --subopt-prob-grid "${PROB_GRID}" \
    --subopt-noise-std-grid "${NOISE_GRID}" \
    --targets corectron,ons,ogd \
    --ons-lr 2.0 \
    --output-root "${BASE_RESULTS}" \
    --figure-root "${BASE_FIGURES}"

append_status "campaign" "done" "1" "run_id=${RUN_ID}"
echo "[done] subopt campaign completed: ${RUN_ID}"
echo "[figure] ${BASE_FIGURES}/subopt_combined.png"
