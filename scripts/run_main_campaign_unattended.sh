#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found at ${PYTHON_BIN}; create .venv or set PYTHON_BIN to an executable path." >&2
  exit 2
fi

RUN_ID="${1:-comparison_$(date +%Y%m%d_%H%M%S)}"
COEFF_GRID="${2:-0.001,0.01,0.1,1,10,100,1000}"
INCLUDE_POLY="${3:-0}"

if (( $# > 3 )); then
  echo "Expected at most three positional arguments; see docs/REPRODUCTION.md." >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "RUN_ID must start with a letter or digit and contain only letters, digits, '.', '_' or '-'." >&2
  exit 2
fi
if [[ "${INCLUDE_POLY}" != "0" && "${INCLUDE_POLY}" != "1" ]]; then
  echo "INCLUDE_POLY must be 0 or 1 (got: ${INCLUDE_POLY})" >&2
  exit 2
fi

BASE_RESULTS="results/${RUN_ID}"
BASE_FIGURES="figures/${RUN_ID}"
# Keep runtime comparable across jobs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${BASE_RESULTS}/mplconfig"


# Check again before each job so one campaign cannot mix source versions.
check_campaign_identity() {
  "${PYTHON_BIN}" scripts/campaign_manifest.py --output-root "${BASE_RESULTS}" \
    comparison --coeff-grid "${COEFF_GRID}" --include-poly "${INCLUDE_POLY}"
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
  echo "coeff_grid=${COEFF_GRID}"
  echo "include_poly=${INCLUDE_POLY}"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
  echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
  echo "VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS}"
  echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
  echo "MPLCONFIGDIR=${MPLCONFIGDIR}"
} > "${INVOCATION_DIR}/environment_snapshot.txt"

ts_now() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

append_status() {
  local job="$1"
  local status="$2"
  local attempt="$3"
  local message="$4"
  printf "%s\t%s\t%s\t%s\t%s\n" "$(ts_now)" "${job}" "${status}" "${attempt}" "${message}" >> "${STATUS_FILE}"
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
  "linear_t10000_pow10_s10" \
  "${PYTHON_BIN}" scripts/run_reg_strength_sweep.py \
    --config configs/main_linear_t10000_s10.yaml \
    --coeff-grid "${COEFF_GRID}" \
    --targets corectron,ons,ogd \
    --ons-lr 2.0 \
    --kons-lr 2.0 \
    --output-root "${BASE_RESULTS}/linear_t10000_pow10_s10" \
    --figure-root "${BASE_FIGURES}/linear_t10000_pow10_s10"

run_job \
  "kernel_rbf_sigma0p5_t1000_pow10_s10_all5" \
  "${PYTHON_BIN}" scripts/run_reg_strength_sweep.py \
    --config configs/main_kernel_t1000_s10.yaml \
    --coeff-grid "${COEFF_GRID}" \
    --targets corectron_rbf,kons,corectron,ons,ogd \
    --ons-lr 2.0 \
    --kons-lr 2.0 \
    --kernel-type rbf \
    --kernel-sigma 0.5 \
    --output-root "${BASE_RESULTS}/kernel_rbf_sigma0p5_t1000_pow10_s10_all5" \
    --figure-root "${BASE_FIGURES}/kernel_rbf_sigma0p5_t1000_pow10_s10_all5"

run_job \
  "kernel_rbf_sigma1p0_t1000_pow10_s10_all5" \
  "${PYTHON_BIN}" scripts/run_reg_strength_sweep.py \
    --config configs/main_kernel_t1000_s10.yaml \
    --coeff-grid "${COEFF_GRID}" \
    --targets corectron_rbf,kons,corectron,ons,ogd \
    --ons-lr 2.0 \
    --kons-lr 2.0 \
    --kernel-type rbf \
    --kernel-sigma 1.0 \
    --output-root "${BASE_RESULTS}/kernel_rbf_sigma1p0_t1000_pow10_s10_all5" \
    --figure-root "${BASE_FIGURES}/kernel_rbf_sigma1p0_t1000_pow10_s10_all5"

run_job \
  "kernel_rbf_sigma2p0_t1000_pow10_s10_all5" \
  "${PYTHON_BIN}" scripts/run_reg_strength_sweep.py \
    --config configs/main_kernel_t1000_s10.yaml \
    --coeff-grid "${COEFF_GRID}" \
    --targets corectron_rbf,kons,corectron,ons,ogd \
    --ons-lr 2.0 \
    --kons-lr 2.0 \
    --kernel-type rbf \
    --kernel-sigma 2.0 \
    --output-root "${BASE_RESULTS}/kernel_rbf_sigma2p0_t1000_pow10_s10_all5" \
    --figure-root "${BASE_FIGURES}/kernel_rbf_sigma2p0_t1000_pow10_s10_all5"

if [[ "${INCLUDE_POLY}" == "1" ]]; then
  run_job \
    "kernel_poly_deg2_t1000_pow10_s10_all5" \
    "${PYTHON_BIN}" scripts/run_reg_strength_sweep.py \
      --config configs/main_kernel_t1000_s10.yaml \
      --coeff-grid "${COEFF_GRID}" \
      --targets corectron_rbf,kons,corectron,ons,ogd \
      --ons-lr 2.0 \
      --kons-lr 2.0 \
      --kernel-type poly \
      --kernel-degree 2 \
      --kernel-coef0 0.0 \
      --kernel-alpha 1.0 \
      --output-root "${BASE_RESULTS}/kernel_poly_deg2_t1000_pow10_s10_all5" \
      --figure-root "${BASE_FIGURES}/kernel_poly_deg2_t1000_pow10_s10_all5"

  run_job \
    "kernel_poly_deg4_t1000_pow10_s10_all5" \
    "${PYTHON_BIN}" scripts/run_reg_strength_sweep.py \
      --config configs/main_kernel_t1000_s10.yaml \
      --coeff-grid "${COEFF_GRID}" \
      --targets corectron_rbf,kons,corectron,ons,ogd \
      --ons-lr 2.0 \
      --kons-lr 2.0 \
      --kernel-type poly \
      --kernel-degree 4 \
      --kernel-coef0 0.0 \
      --kernel-alpha 1.0 \
      --output-root "${BASE_RESULTS}/kernel_poly_deg4_t1000_pow10_s10_all5" \
      --figure-root "${BASE_FIGURES}/kernel_poly_deg4_t1000_pow10_s10_all5"

  run_job \
    "kernel_poly_deg8_t1000_pow10_s10_all5" \
    "${PYTHON_BIN}" scripts/run_reg_strength_sweep.py \
      --config configs/main_kernel_t1000_s10.yaml \
      --coeff-grid "${COEFF_GRID}" \
      --targets corectron_rbf,kons,corectron,ons,ogd \
      --ons-lr 2.0 \
      --kons-lr 2.0 \
      --kernel-type poly \
      --kernel-degree 8 \
      --kernel-coef0 0.0 \
      --kernel-alpha 1.0 \
      --output-root "${BASE_RESULTS}/kernel_poly_deg8_t1000_pow10_s10_all5" \
      --figure-root "${BASE_FIGURES}/kernel_poly_deg8_t1000_pow10_s10_all5"
else
  append_status "kernel_poly_suite" "skip" "0" "INCLUDE_POLY=0"
fi

if [[ "${INCLUDE_POLY}" == "1" ]]; then
  DASHBOARD_GROUPS="linear,kernel_rbf,kernel_poly"
else
  DASHBOARD_GROUPS="linear,kernel_rbf"
fi

run_job \
  "build_dashboards" \
  "${PYTHON_BIN}" scripts/build_sweep_dashboards.py \
    --run-root "${BASE_RESULTS}" \
    --out-root "${BASE_FIGURES}/dashboards" \
    --groups "${DASHBOARD_GROUPS}"

append_status "campaign" "done" "1" "run_id=${RUN_ID}"
echo "[done] campaign completed: ${RUN_ID}"
