#!/usr/bin/env bash
# Copyright (C) 2026 Yibo Yang
# SPDX-License-Identifier: GPL-3.0-or-later

set -Eeuo pipefail

# Complete Case-2 launcher.
# Paper cases: CASE_SET=paper (default: S1,S2,S3,S4,S7)
# All scales:  CASE_SET=all bash run_case2_all_scales_220.sh
# Selected:    CASE_SET=S2,S4,S7 bash run_case2_all_scales_220.sh
# Restart:     AUTO_RESTART=1 CASE_SET=S7 bash run_case2_all_scales_220.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Optional environment file. A preconfigured shell works without this file.
CALLER_EXE="${EXE:-}"
CALLER_MPIEXEC="${MPIEXEC:-}"
ENV_FILE_EXPLICIT="${ENV_FILE+x}"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/int64_env.sh}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
elif [[ -n "$ENV_FILE_EXPLICIT" ]]; then
  echo "ERROR: requested ENV_FILE does not exist: $ENV_FILE" >&2
  exit 2
fi
ENV_EXE="${EXE:-}"
ENV_MPIEXEC="${MPIEXEC:-}"

# Prefer a caller override, then this repository's build, then env-file values.
if [[ -n "$CALLER_EXE" ]]; then
  EXE="$CALLER_EXE"
elif [[ -x "$SCRIPT_DIR/build-int64/TopOpt3DGPU4Rank" ]]; then
  EXE="$SCRIPT_DIR/build-int64/TopOpt3DGPU4Rank"
elif [[ -x "$SCRIPT_DIR/build/TopOpt3DGPU4Rank" ]]; then
  EXE="$SCRIPT_DIR/build/TopOpt3DGPU4Rank"
elif [[ -n "$ENV_EXE" ]]; then
  EXE="$ENV_EXE"
else
  EXE="$SCRIPT_DIR/build/TopOpt3DGPU4Rank"
fi

if [[ -n "$CALLER_MPIEXEC" ]]; then
  MPIEXEC="$CALLER_MPIEXEC"
elif [[ -n "$ENV_MPIEXEC" ]]; then
  MPIEXEC="$ENV_MPIEXEC"
elif [[ -n "${PETSC_DIR:-}" && -n "${PETSC_ARCH:-}" && -x "$PETSC_DIR/$PETSC_ARCH/bin/mpiexec" ]]; then
  MPIEXEC="$PETSC_DIR/$PETSC_ARCH/bin/mpiexec"
else
  MPIEXEC="$(command -v mpiexec || true)"
fi

RUN_ROOT="${1:-${RUN_ROOT:-$SCRIPT_DIR/results/case2_220}}"
CASE_SET="${CASE_SET:-paper}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_ITER="${MAX_ITER:-220}"
WRITE_EVERY="${WRITE_EVERY:-10}"
NT="${NT:-1000}"
TF="${TF:-1000}"
AUTO_RESTART="${AUTO_RESTART:-0}"
CHANGE_TOL="${CHANGE_TOL:-1e-2}"
CHANGE_STABLE_ITERS="${CHANGE_STABLE_ITERS:-5}"
TMGTOP_MMA_SOLVER="${TMGTOP_MMA_SOLVER:-scalar_dual}"

# Same physical filter radius as S7 (rmin=6), except the reported S1 lower
# bound of 1.5 cells. S5/S6 complete the established mesh sequence.
RMIN_S1="${RMIN_S1:-1.5}"
RMIN_S2="${RMIN_S2:-1.7578125}"
RMIN_S3="${RMIN_S3:-3.0}"
RMIN_S4="${RMIN_S4:-3.75}"
RMIN_S5="${RMIN_S5:-4.21875}"
RMIN_S6="${RMIN_S6:-4.6875}"
RMIN_S7="${RMIN_S7:-6.0}"

[[ -x "$EXE" ]] || {
  echo "ERROR: executable not found: $EXE" >&2
  echo "Build the repository first or set EXE=/absolute/path/TopOpt3DGPU4Rank." >&2
  exit 2
}
[[ -n "$MPIEXEC" && -x "$MPIEXEC" ]] || {
  echo "ERROR: MPI launcher not found: ${MPIEXEC:-<empty>}" >&2
  echo "Set MPIEXEC or load the PETSc/MPI environment." >&2
  exit 2
}
[[ "$MAX_ITER" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid MAX_ITER=$MAX_ITER" >&2; exit 2; }
[[ "$WRITE_EVERY" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid WRITE_EVERY=$WRITE_EVERY" >&2; exit 2; }
[[ "$NT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid NT=$NT" >&2; exit 2; }
[[ "$AUTO_RESTART" == 0 || "$AUTO_RESTART" == 1 ]] || { echo "ERROR: AUTO_RESTART must be 0 or 1." >&2; exit 2; }

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
(( ${#GPU_ARRAY[@]} > 0 )) || { echo "ERROR: GPU_IDS is empty." >&2; exit 2; }
for gpu_id in "${GPU_ARRAY[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid GPU ID '$gpu_id'." >&2; exit 2; }
done
NRANKS="${#GPU_ARRAY[@]}"

case "$CASE_SET" in
  paper) CASES=(S1 S2 S3 S4 S7) ;;
  all)   CASES=(S1 S2 S3 S4 S5 S6 S7) ;;
  *) IFS=',' read -r -a CASES <<< "${CASE_SET// /}" ;;
esac
(( ${#CASES[@]} > 0 )) || { echo "ERROR: CASE_SET selected no cases." >&2; exit 2; }
for scale in "${CASES[@]}"; do
  [[ "$scale" =~ ^S[1-7]$ ]] || {
    echo "ERROR: unknown scale '$scale'. Use paper, all, or S1,S4,S7." >&2
    exit 2
  }
done

mesh_and_filter() {
  case "$1" in
    S1) printf '70 42 35 %s\n' "$RMIN_S1" ;;
    S2) printf '150 90 75 %s\n' "$RMIN_S2" ;;
    S3) printf '256 154 128 %s\n' "$RMIN_S3" ;;
    S4) printf '320 192 160 %s\n' "$RMIN_S4" ;;
    S5) printf '360 216 180 %s\n' "$RMIN_S5" ;;
    S6) printf '400 240 200 %s\n' "$RMIN_S6" ;;
    S7) printf '512 307 256 %s\n' "$RMIN_S7" ;;
  esac
}

mkdir -p "$RUN_ROOT"
export GPU_ID_LIST="$GPU_IDS" TMGTOP_MMA_SOLVER
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export MPIR_CVAR_ENABLE_GPU=0 MPICH_GPU_SUPPORT_ENABLED=0 MPIR_CVAR_CH4_OFI_ENABLE_HMEM=0
export PETSC_VEC_PINNED_MEMORY_MIN="${PETSC_VEC_PINNED_MEMORY_MIN:-2000000000}"

RANK_WRAPPER="$RUN_ROOT/rank_wrapper.sh"
cat > "$RANK_WRAPPER" <<'WRAPPER'
#!/usr/bin/env bash
set -Eeuo pipefail
local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${MV2_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${PMI_RANK:-0}}}}}"
IFS=',' read -r -a gpu_ids <<< "${GPU_ID_LIST:-0}"
if (( local_rank < 0 || local_rank >= ${#gpu_ids[@]} )); then
  echo "GPU_BIND_ERROR local_rank=$local_rank GPU_ID_LIST=${GPU_ID_LIST:-}" >&2
  exit 96
fi
export CUDA_VISIBLE_DEVICES="${gpu_ids[$local_rank]}"
export MPIR_CVAR_ENABLE_GPU=0 MPICH_GPU_SUPPORT_ENABLED=0 MPIR_CVAR_CH4_OFI_ENABLE_HMEM=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
echo "RANK_BIND local_rank=$local_rank CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host=$(hostname)"
exec "$@"
WRAPPER
chmod +x "$RANK_WRAPPER"

run_case() {
  local scale="$1" nelx nely nelz rmin
  read -r nelx nely nelz rmin <<< "$(mesh_and_filter "$scale")"
  local rtag="${rmin//./p}"
  local case_name="case2_${scale}_${nelx}x${nely}x${nelz}_rmin${rtag}"
  local case_root="$RUN_ROOT/$scale"
  local outdir="$case_root/output"
  local logdir="$case_root/logs"
  local final_vtk="$outdir/${case_name}_iter${MAX_ITER}.vtk"
  mkdir -p "$outdir" "$logdir"

  if [[ -f "$final_vtk" ]]; then
    echo "SKIP scale=$scale reason=iteration_${MAX_ITER}_vtk_exists file=$final_vtk"
    return 0
  fi

  local restart_file='' restart_iter=-1 latest_iter=-1 vtk name n
  local -a restart_args=()
  if [[ "$AUTO_RESTART" == 1 ]]; then
    shopt -s nullglob
    for vtk in "$outdir/${case_name}_iter"*.vtk; do
      name="${vtk##*/}"
      n="${name#${case_name}_iter}"
      n="${n%.vtk}"
      if [[ "$n" =~ ^[0-9]+$ ]]; then
        (( n > latest_iter )) && latest_iter="$n"
        if (( n > restart_iter && n < MAX_ITER )); then restart_iter="$n"; restart_file="$vtk"; fi
      fi
    done
    shopt -u nullglob
  else
    local -a previous=()
    shopt -s nullglob
    previous=("$outdir/${case_name}_iter"*.vtk)
    shopt -u nullglob
    [[ -e "$outdir/${case_name}_density.csv" ]] && previous+=("$outdir/${case_name}_density.csv")
    if (( ${#previous[@]} > 0 )); then
      echo "ERROR: fresh-run directory already contains topology data for $scale:" >&2
      printf '  %s\n' "${previous[@]}" >&2
      echo "Use a new RUN_ROOT or set AUTO_RESTART=1 deliberately." >&2
      return 2
    fi
  fi

  if (( latest_iter >= MAX_ITER )); then echo "SKIP scale=$scale reason=already_complete"; return 0; fi
  if (( restart_iter >= 0 )); then
    restart_args=(-restart_vtk "$restart_file" -restart_iter "$restart_iter" -restart_invert_projection 1)
    echo "AUTO_RESTART scale=$scale file=$restart_file completed_iteration=$restart_iter"
    echo "WARNING: MMA asymptote history is not restored."
  fi

  local mat_type="mpiaijcusparse" vec_type="mpicuda"
  if (( NRANKS == 1 )); then mat_type="aijcusparse"; vec_type="cuda"; fi

  local -a app=(
    "$EXE"
    -backend gpu -mat_type "$mat_type" -vec_type "$vec_type"
    -ksp_type cg -ksp_rtol 1e-5 -ksp_max_it 2000 -ksp_error_if_not_converged
    -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 0
    -pc_gamg_aggressive_coarsening 2 -pc_gamg_aggressive_square_graph false
    -pc_gamg_low_memory_threshold_filter -pc_gamg_process_eq_limit 50 -pc_gamg_coarse_eq_limit 500
    -mg_levels_ksp_type richardson -mg_levels_ksp_max_it 1 -mg_levels_pc_type jacobi
    -mg_coarse_pc_type redundant
    -history_mode host_full -history_host_memory pageable -checkpoint_interval 32
    -use_gpu_aware_mpi 0 -on_error_mpiabort -vec_pinned_memory_min "$PETSC_VEC_PINNED_MEMORY_MIN"
    -benchmark_fixed_design 0
    -max_iter "$MAX_ITER" -change_tol "$CHANGE_TOL"
    -min_iter_before_stop "$MAX_ITER" -change_stable_iters "$CHANGE_STABLE_ITERS"
    -write_vtk 1 -write_every_iter "$WRITE_EVERY" -save_problem_preview 0 -save_final_density 1
    -step_print_every 0
    -Lx 0.100 -Ly 0.060 -Lz 0.050
    -nelx "$nelx" -nely "$nely" -nelz "$nelz" -volfrac 0.15 -rmin "$rmin"
    -beta0 1 -beta_max 16 -beta_update_every 40 -eta 0.5
    -k_low 0.27 -k_high 202.4 -cv_low 9.0e5 -cv_high 2367490 -pk 2.5 -pc 1.5
    -q0 6.0e4 -q_stage1_end 250 -q_stage2_end 500 -q_stage3_end 750
    -q_stage1_scale 1.0 -q_stage2_scale 0.7 -q_stage3_scale 0.7 -q_stage4_scale 0.7
    -Tc 25 -hconv 60 -tf "$TF" -nt "$NT"
    -output_dir "$outdir" -case_name "$case_name"
    "${restart_args[@]}"
  )
  local -a cmd=("$MPIEXEC" -n "$NRANKS" "$RANK_WRAPPER" "${app[@]}")

  printf '%q ' "${cmd[@]}" > "$case_root/last_command.txt"; printf '\n' >> "$case_root/last_command.txt"
  {
    echo "start=$(date -Iseconds)"
    echo "scale=$scale mesh=${nelx}x${nely}x${nelz} elements=$((nelx*nely*nelz))"
    echo "rmin_cells=$rmin nt=$NT tf=$TF max_iter=$MAX_ITER write_every=$WRITE_EVERY"
    echo "mpi_ranks=$NRANKS gpu_ids=$GPU_IDS mma_solver=$TMGTOP_MMA_SOLVER"
    echo "restart_file=${restart_file:-none} restart_iter=$restart_iter"
    nvidia-smi -L 2>/dev/null || true
  } | tee -a "$logdir/launcher.log"

  echo "RUN scale=$scale output=$case_root"
  set +e
  env -u PETSC_OPTIONS "${cmd[@]}" 2>&1 | tee -a "$logdir/optimization.log"
  local rc="${PIPESTATUS[0]}"
  set -e
  echo "end=$(date -Iseconds) exit_status=$rc" | tee -a "$logdir/launcher.log"
  return "$rc"
}

echo "Case-2 production optimization"
echo "  executable: $EXE"
echo "  MPI:        $MPIEXEC"
echo "  cases:      ${CASES[*]}"
echo "  GPU IDs:    $GPU_IDS ($NRANKS ranks)"
echo "  output:     $RUN_ROOT"

for scale in "${CASES[@]}"; do run_case "$scale"; done
echo "ALL_SELECTED_CASES_FINISHED run_root=$RUN_ROOT"
