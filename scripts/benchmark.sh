#!/bin/bash
# =============================================================================
# benchmark.sh — Πλήρες automated benchmark για όλες τις υλοποιήσεις
# =============================================================================

set -euo pipefail

# ── Χρώματα για terminal output ───────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Παράμετροι από γραμμή εντολών ────────────────────────────────────────────
DATASET_SIZE="${1:-small}"
MODE_ARG="${2:-both}"
RUNS="${3:-3}"

# ── Directories ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results"
DATA_DIR="$PROJECT_DIR/data"
BIN_DIR="$PROJECT_DIR"

mkdir -p "$RESULTS_DIR" "$DATA_DIR"

# ── Παράμετροι dataset ────────────────────────────────────────────────────────
case "$DATASET_SIZE" in
    small)   N=1000000;  D=32;  SIZE_LABEL="Small  (1M×32,    ~256MB)"  ;;
    medium)  N=5000000;  D=64;  SIZE_LABEL="Medium (5M×64,   ~2.56GB)"  ;;
    large)   N=10000000; D=128; SIZE_LABEL="Large  (10M×128, ~10.24GB)" ;;
    *)
        echo -e "${RED}[ERROR]${RESET} Unknown dataset size: '$DATASET_SIZE'"
        echo "Use: small | medium | large"
        exit 1 ;;
esac

# ── Modes ─────────────────────────────────────────────────────────────────────
case "$MODE_ARG" in
    both)     MODES=("standard" "minmax") ;;
    standard) MODES=("standard") ;;
    minmax)   MODES=("minmax") ;;
    *)
        echo -e "${RED}[ERROR]${RESET} Unknown mode: '$MODE_ARG'"
        echo "Use: standard | minmax | both"
        exit 1 ;;
esac

# ── Thread/Process configurations ─────────────────────────────────────────────
OPENMP_THREADS=(1 2 4 8)
MPI_PROCS=(1 2 4 8)

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_BIN="$DATA_DIR/${DATASET_SIZE}.bin"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV_OUT="$RESULTS_DIR/benchmark_${DATASET_SIZE}_${TIMESTAMP}.csv"
LOG_OUT="$RESULTS_DIR/benchmark_${DATASET_SIZE}_${TIMESTAMP}.log"

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         Parallel Scaler — Benchmark Suite                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Logging setup ─────────────────────────────────────────────────────────────
exec > >(tee -a "$LOG_OUT") 2>&1
echo "[$(date)] Benchmark started"

# ── Παραγωγή dataset αν δεν υπάρχει ──────────────────────────────────────────
if [ ! -f "$INPUT_BIN" ]; then
    echo -e "\n${YELLOW}[GEN]${RESET} Generating dataset: N=$N, D=$D"
    python3 "$SCRIPT_DIR/generate_data.py" --samples "$N" --features "$D" --output "$INPUT_BIN" --dtype float64
else
    echo -e "${GREEN}[DATA]${RESET} Dataset exists: $(du -h "$INPUT_BIN" | cut -f1)"
fi

# ── Εύρεση διαθέσιμων εκτελέσιμων ────────────────────────────────────────────
SERIAL_BIN="$BIN_DIR/scaler_serial"
SIMD_BIN="$BIN_DIR/scaler_simd"
OPENMP_BIN="$BIN_DIR/scaler_OpenMP"
MPI_BIN="$BIN_DIR/scaler_mpi"
CUDA_BIN="$BIN_DIR/scaler_cuda"

# ── CSV Header ────────────────────────────────────────────────────────────────
# Εδώ αποθηκεύω όλα τα αποτελέσματα σε CSV για να μπορούν να αναλυθούν αργότερα 
# με Python ή για την παραγωγή γραφημάτων και υπολογισμό  του speedup
echo "impl,variant,mode,threads_or_procs,run,time_secs,dataset,N,D" > "$CSV_OUT"

# ── Εκκένωση page cache (Απενεργοποιημένο για τον Server) ──────────────────────
# Εδώ καθαρίζω το page cache του λειτουργικού πριν από κάθε μέτρηση για να μειωθεί
# η επίδραση των προηγούμενων αναγνώσεων του ίδιου file
drop_cache() {
    return 0
}

# ── Κεντρική συνάρτηση εκτέλεσης ─────────────────────────────────────────────
# Εκτελεί πολλαπλές επαναλήψεις της ίδιας υλοποίησης. Οι χρόνοι αποθηκεύονται
# στο CSV και στο τέλος υπολογίζεται ο μέσος χρόνος εκτέλεσης ώστε να μειωθεί 
# η επίδραση τυχαίων διακυμάνσεων του συστήματος.
run_impl() {
    local impl="$1"; shift
    local variant="$1"; shift
    local mode="$1"; shift
    local tp="$1"; shift

    local times=()
    for ((r=1; r<=RUNS; r++)); do
        local output
        if output=$("$@" 2>&1); then
            local t
            t=$(echo "$output" | grep -oP '[\d]+\.[\d]+(?= sec)' | tail -1)
            if [ -z "$t" ]; then
                t=$(echo "$output" | grep -iE 'time' | grep -oP '[\d]+\.[\d]+' | tail -1)
            fi
            if [ -n "$t" ]; then
                echo -e "    Run $r: ${t}s"
                times+=("$t")
                echo "${impl},${variant},${mode},${tp},${r},${t},${DATASET_SIZE},${N},${D}" >> "$CSV_OUT"
            fi
        fi
    done

    if [ ${#times[@]} -eq 0 ]; then
        LAST_AVG_TIME="ERROR"
        return 1
    fi

    local arr_str
    arr_str=$(printf '%s,' "${times[@]}")
    LAST_AVG_TIME=$(python3 -c "vals=[float(x) for x in '${arr_str%,}'.split(',')]; print(f'{sum(vals)/len(vals):.4f}')")
    echo -e "    ${GREEN}→ Average: ${LAST_AVG_TIME}s${RESET}"
}

# Ο χρόνος της σειριακής υλοποίησης χρησιμοποιείται ως baseline για τον υπολογισμό του speedup:
# speedup = T_serial / T_parallel
declare -A SERIAL_TIME

for MODE in "${MODES[@]}"; do
    OUT_BIN="$DATA_DIR/out_${DATASET_SIZE}_${MODE}.bin"

    echo -e "\n${BOLD}  MODE: $MODE | Dataset: $DATASET_SIZE${RESET}"

    # 1. Serial
    if [ -x "$SERIAL_BIN" ]; then
        echo -e "\n[1/5] Serial"
        run_impl "serial" "serial" "$MODE" "1" "$SERIAL_BIN" "$INPUT_BIN" "$OUT_BIN" "$N" "$D" "$MODE"
        SERIAL_TIME["$MODE"]=$LAST_AVG_TIME
    fi

    # 2. SIMD
    if [ -x "$SIMD_BIN" ]; then
        echo -e "\n[2/5] SIMD (AVX2)"
        run_impl "simd" "avx2" "$MODE" "1" "$SIMD_BIN" "$INPUT_BIN" "$OUT_BIN" "$N" "$D" "$MODE"
    fi

    # 3. OpenMP
    # Εκτέλεση για διαφορετικό αριθμό threads ώστε να
    # μελετηθεί η κλιμάκωση (scalability) της OpenMP υλοποίησης.
    if [ -x "$OPENMP_BIN" ]; then
        echo -e "\n[3/5] OpenMP"
        for T in "${OPENMP_THREADS[@]}"; do
            echo -e "  → $T thread(s)"
            run_impl "openmp" "openmp" "$MODE" "$T" "$OPENMP_BIN" "$INPUT_BIN" "$OUT_BIN" "$N" "$D" "$MODE" "$T"
        done
    fi

    # 4. MPI
    # Εκτέλεση για διαφορετικό αριθμό MPI διεργασιών.
    # Με αυτόν τον τρόπο αξιολογείται το κόστος επικοινωνίας
    # και το συνολικό speedup της κατανεμημένης εκτέλεσης.
    if command -v mpirun &>/dev/null && [ -x "$MPI_BIN" ]; then
        echo -e "\n[4/5] MPI"
        for P in "${MPI_PROCS[@]}"; do
            echo -e "  → $P process(es)"
            run_impl "mpi" "mpi" "$MODE" "$P" mpirun --oversubscribe -np "$P" "$MPI_BIN" "$INPUT_BIN" "$OUT_BIN" "$N" "$D" "$MODE"
        done
    fi

    # 5. CUDA
    if [ -x "$CUDA_BIN" ]; then
        echo -e "\n[5/5] CUDA (GPU)"
        run_impl "cuda" "cuda" "$MODE" "1" "$CUDA_BIN" "$INPUT_BIN" "$OUT_BIN" "$N" "$D" "$MODE"
    fi
done

echo -e "\n${BOLD}${GREEN}Benchmark Complete!${RESET}"
