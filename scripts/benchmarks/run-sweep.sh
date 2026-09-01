#!/usr/bin/env bash
# Sweep (M,N,K) ∈ {512,1024,2048,4096,8192}³ for both i8 and bf16 GEMMs.
#
# 1. PERFORMANCE: reuse benchmarks/driver.py with a generated sweep JSON.
# 2. CORRECTNESS: per-shape diff of default tpp-run vs
#                 tpp-run --nano-kernels --registerBlocking=... (same
#                 flags as the perf config).
# 3. PLOT:        TFLOPS vs shape (one line per dtype) via
#                 benchmarks/scripts/plot_benchmarks.py ...

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

BUILD_DIR="${ROOT_DIR}/build"
ITERS=100
OUT_DIR=""
SKIP_CORRECTNESS=0
SKIP_PERF=0
SKIP_PLOT=0
SHAPES_CSV="512,1024,2048,4096,8192"
DTYPES_CSV="i8,bf16"
ABS_TOL="0.02"
REL_TOL="0.02"

usage() {
  cat <<EOF
Usage: $0 [options]

  -b, --build DIR        tpp-mlir build dir (default: ${BUILD_DIR})
  -o, --out DIR          output dir (default: build/sweep-results/<ts>)
  -n, --iters N          perf iterations per shape (default: ${ITERS})
      --shapes A,B,C     dim sweep (default: ${SHAPES_CSV})
      --dtypes i8,bf16   subset of dtypes (default: ${DTYPES_CSV})
      --abs-tol F        fpcmp absolute tolerance (default: ${ABS_TOL})
      --rel-tol F        fpcmp relative tolerance (default: ${REL_TOL})
      --skip-perf        skip performance phase
      --skip-correctness skip correctness phase
      --skip-plot        skip plotting phase
  -h, --help             show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build) BUILD_DIR="$2"; shift 2;;
    -o|--out) OUT_DIR="$2"; shift 2;;
    -n|--iters) ITERS="$2"; shift 2;;
    --shapes) SHAPES_CSV="$2"; shift 2;;
    --dtypes) DTYPES_CSV="$2"; shift 2;;
    --abs-tol) ABS_TOL="$2"; shift 2;;
    --rel-tol) REL_TOL="$2"; shift 2;;
    --skip-perf) SKIP_PERF=1; shift;;
    --skip-correctness) SKIP_CORRECTNESS=1; shift;;
    --skip-plot) SKIP_PLOT=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${BUILD_DIR}/sweep-results/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/ir"
# Normalize to absolute paths so subshells that `cd` keep working.
BUILD_DIR="$(cd -- "${BUILD_DIR}" &>/dev/null && pwd)"
OUT_DIR="$(cd -- "${OUT_DIR}" &>/dev/null && pwd)"

BIN_DIR="${BUILD_DIR}/bin"
LIB_DIR="${BUILD_DIR}/lib"
MLIR_GEN="${BIN_DIR}/mlir-gen"
TPP_RUN="${BIN_DIR}/tpp-run"
FPCMP="${BIN_DIR}/fpcmp"
DRIVER="${ROOT_DIR}/benchmarks/driver.py"
GEN_CFG="${SCRIPT_DIR}/gen-sweep-config.py"
PLOTTER="${ROOT_DIR}/scripts/benchmarks/plot_benchmarks.py"
POSTPROCESS="${ROOT_DIR}/benchmarks/mlir/sfc_ca_gemm/postprocess.py"

for f in "${MLIR_GEN}" "${TPP_RUN}" "${FPCMP}" "${DRIVER}" "${GEN_CFG}" "${PLOTTER}" "${POSTPROCESS}"; do
  if [[ ! -e "${f}" ]]; then
    echo "ERROR: missing required file: ${f}" >&2
    exit 1
  fi
done

export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH:-}"

CFG_JSON="${OUT_DIR}/sweep.json"
echo ">> Generating sweep config -> ${CFG_JSON}"
python3 "${GEN_CFG}" \
  -o "${CFG_JSON}" \
  --iters "${ITERS}" \
  --shapes "${SHAPES_CSV}" \
  --dtypes "${DTYPES_CSV}"

# ---------------------------------------------------------------------------
# Phase 1: Performance via driver.py
# ---------------------------------------------------------------------------
PERF_TXT="${OUT_DIR}/perf.txt"
if [[ "${SKIP_PERF}" -eq 0 ]]; then
  echo ">> Running performance sweep (this can take a while)"
  # driver.py does `sys.path.append("harness")` relative to CWD, so it must
  # be invoked from the benchmarks/ directory.
  ( cd "${ROOT_DIR}/benchmarks" && \
    python3 "${DRIVER}" \
      --build "${BUILD_DIR}" \
      --config "${CFG_JSON}" \
      --ignore-errors \
      2>"${OUT_DIR}/logs/perf.stderr" ) \
    | tee "${PERF_TXT}"
else
  echo ">> Skipping performance phase"
fi

# ---------------------------------------------------------------------------
# Phase 1b: Throughput CSV (sorted by arithmetic complexity. Parse perf.txt
#           into a raw (M,N,K,runtime_s) CSV per dtype, then reuse the shared
#           postprocess.py to emit a sorted CSV with FLOPs and TFLOP/s.
# ---------------------------------------------------------------------------
if [[ -s "${PERF_TXT}" ]]; then
  echo ">> Building throughput CSV(s) from ${PERF_TXT}"
  # Split perf.txt into one raw CSV per dtype prefix (e.g. i8_i8_quant,
  # bf16_bf16). Raw CSVs land under logs/ so only the sorted throughput_*.csv
  # files live in OUT_DIR. Emits the list of raw CSV paths on stdout.
  RAW_CSVS="$(python3 - "${PERF_TXT}" "${OUT_DIR}/logs" <<'PY'
import os
import re
import sys

perf_path, out_dir = sys.argv[1], sys.argv[2]
line_re = re.compile(r"^(?P<name>\S+):\s+(?P<g>[0-9.]+)\s+gflops\s*$")
dims_re = re.compile(r"_(\d+)x(\d+)x(\d+)_")

rows = {}  # prefix -> list of (M, N, K, runtime_s)
for line in open(perf_path):
    m = line_re.match(line.strip())
    if not m:
        continue
    name = m.group("name")
    d = dims_re.search(name)
    if not d:
        continue
    M, N, K = (int(x) for x in d.groups())
    gflops = float(m.group("g"))
    runtime_s = (2.0 * M * N * K) / (gflops * 1.0e9) if gflops > 0.0 else float("nan")
    prefix = name[: d.start()]
    rows.setdefault(prefix, []).append((M, N, K, runtime_s))

paths = []
for prefix, recs in rows.items():
    raw = os.path.join(out_dir, f"throughput_raw_{prefix}.csv")
    with open(raw, "w") as f:
        f.write("M,N,K,runtime_s\n")
        for M, N, K, runtime_s in recs:
            f.write(f"{M},{N},{K},{runtime_s:.10g}\n")
    paths.append(raw)

print("\n".join(paths))
PY
)"

  while IFS= read -r raw_csv; do
    [[ -z "${raw_csv}" ]] && continue
    prefix="$(basename "${raw_csv}")"; prefix="${prefix#throughput_raw_}"; prefix="${prefix%.csv}"
    sorted_csv="${OUT_DIR}/throughput_${prefix}.csv"
    python3 "${POSTPROCESS}" "${raw_csv}" "${sorted_csv}"
  done <<< "${RAW_CSVS}"
fi

# ---------------------------------------------------------------------------
# Phase 2: Correctness per (dtype, M, N, K)
# ---------------------------------------------------------------------------
CORR_CSV="${OUT_DIR}/correctness.csv"
if [[ "${SKIP_CORRECTNESS}" -eq 0 ]]; then
  echo ">> Running correctness sweep"
  echo "dtype,M,N,K,status,detail" > "${CORR_CSV}"

  # Detect AMX support (mirror driver.py's extension gating).
  HAS_AMX_INT8=0; HAS_AMX_BF16=0
  if grep -q amx_int8 /proc/cpuinfo 2>/dev/null; then HAS_AMX_INT8=1; fi
  if grep -q amx_bf16 /proc/cpuinfo 2>/dev/null; then HAS_AMX_BF16=1; fi

  # mlir-gen flag templates (must match gen-sweep-config.py).
  # Use --kernel=args (same as the perf config); tpp-run --splat-to-random
  # --seed=123 makes both baseline and test runs see identical inputs.
  gen_args_i8="--kernel=args --data-type=i8 --tiles=32,32,64 --vnni=4 --quant --scale-type=f8E8M0FNU"
  gen_args_bf16="--kernel=args --data-type=bf16 --tiles=32,32,32 --vnni=2"
  # Data-init flags must be applied identically to baseline and test so both
  # runs see the same inputs; i8 requires --init-type=quant.
  init_args_i8="--init-type=quant"
  init_args_bf16="--init-type=normal"
  vk_args_i8="--def-parallel --nano-kernels --registerBlocking=32,32,64 --gemm-unroll=16,16,16 --bench-replication-gb=5"
  vk_args_bf16="--def-parallel --nano-kernels --registerBlocking=32,32,32 --gemm-unroll=16,16,16 --bench-replication-gb=5"

  IFS=',' read -ra DTYPES <<< "${DTYPES_CSV}"
  IFS=',' read -ra SHAPES <<< "${SHAPES_CSV}"

  pass_count=0; fail_count=0; skip_count=0
  for dt in "${DTYPES[@]}"; do
    case "${dt}" in
      i8)
        if [[ "${HAS_AMX_INT8}" -eq 0 ]]; then
          echo "   [skip] dtype=i8 (amx_int8 not available)"
          continue
        fi
        gen_args="${gen_args_i8}"; vk_args="${vk_args_i8}"; init_args="${init_args_i8}";;
      bf16)
        if [[ "${HAS_AMX_BF16}" -eq 0 ]]; then
          echo "   [skip] dtype=bf16 (amx_bf16 not available)"
          continue
        fi
        gen_args="${gen_args_bf16}"; vk_args="${vk_args_bf16}"; init_args="${init_args_bf16}";;
      *) echo "Unknown dtype: ${dt}" >&2; exit 2;;
    esac

    for M in "${SHAPES[@]}"; do
      for N in "${SHAPES[@]}"; do
        for K in "${SHAPES[@]}"; do
          tag="${dt}_${M}x${N}x${K}"
          ir="${OUT_DIR}/ir/${tag}.mlir"
          ref_out="${OUT_DIR}/logs/${tag}.ref.out"
          test_out="${OUT_DIR}/logs/${tag}.test.out"
          log="${OUT_DIR}/logs/${tag}.corr.log"

          # Run each correctness check in a subshell so a failing command
          # inside it does not abort the outer `set -e` script.
          rc=0
          (
            set +e
            {
              echo "=== ${tag} ==="
              echo "+ mlir-gen ${gen_args} --batch=${M} --layers=${K},${N}"
              "${MLIR_GEN}" ${gen_args} --batch=${M} --layers=${K},${N} > "${ir}" || exit 10
              echo "+ tpp-run (baseline)"
              "${TPP_RUN}" -e entry -entry-point-result=void -print \
                --splat-to-random --seed=123 ${init_args} "${ir}" > "${ref_out}" || exit 20
              echo "+ tpp-run (nano-kernels)"
              "${TPP_RUN}" -e entry -entry-point-result=void -print \
                --splat-to-random --seed=123 ${init_args} ${vk_args} "${ir}" > "${test_out}" || exit 30
              echo "+ fpcmp"
              "${FPCMP}" -a "${ABS_TOL}" -r "${REL_TOL}" -i "${ref_out}" "${test_out}" || exit 40
            } >"${log}" 2>&1
          ) || rc=$?

          if [[ ${rc} -eq 0 ]]; then
            status=PASS; pass_count=$((pass_count+1))
            # tidy: drop bulky output files for passing shapes
            rm -f "${ref_out}" "${test_out}" "${ir}"
          else
            status="FAIL(rc=${rc})"; fail_count=$((fail_count+1))
          fi
          echo "${dt},${M},${N},${K},${status},log=${log}" >> "${CORR_CSV}"
          printf "   %-14s %s\n" "${status}" "${tag}"
        done
      done
    done
  done

  echo ">> Correctness summary: PASS=${pass_count} FAIL=${fail_count} SKIP=${skip_count}"
else
  echo ">> Skipping correctness phase"
fi

# ---------------------------------------------------------------------------
# Phase 3: Plot TFLOPS vs shapes from the sorted throughput CSV(s).
# ---------------------------------------------------------------------------
if [[ "${SKIP_PLOT}" -eq 0 ]]; then
  PNG="${OUT_DIR}/tflops.png"
  shopt -s nullglob
  THROUGHPUT_CSVS=("${OUT_DIR}"/throughput_*.csv)
  shopt -u nullglob
  if [[ "${#THROUGHPUT_CSVS[@]}" -gt 0 ]]; then
    echo ">> Plotting -> ${PNG}"
    python3 "${PLOTTER}" -o "${PNG}" "${THROUGHPUT_CSVS[@]}" \
      || echo "WARN: plotting failed (see stderr above)"
  else
    echo ">> Skipping plot (no throughput CSVs found)"
  fi
fi

echo
echo "Done."
echo "  Output dir : ${OUT_DIR}"
echo "  Perf       : ${PERF_TXT}"
echo "  Throughput : ${OUT_DIR}/throughput_*.csv"
echo "  Correctness: ${CORR_CSV}"
echo "  Plot       : ${OUT_DIR}/tflops.png"
