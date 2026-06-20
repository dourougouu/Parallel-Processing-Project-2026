# Parallel Data Preprocessing Scaler

High-performance feature scaling for large numerical datasets — benchmarking five architectural paradigms on enterprise hardware.

> Developed for the **Parallel Processing** course  
> Department of Computer Engineering & Informatics, University of Patras (CEID) — 2026

---

## Overview

This project implements and compares four parallel programming models for accelerating `StandardScaler` and `MinMaxScaler` operations on large datasets. Each implementation targets a different hardware or software layer, from CPU vectorization to GPU massively parallel execution.

---

## Implementations

| #   | Paradigm        | Technology     | Description                     |
| --- | --------------- | -------------- | ------------------------------- |
| 1   | Serial          | C++17          | Baseline for benchmarking       |
| 2   | SIMD            | AVX2 (256-bit) | Vectorized CPU computation      |
| 3   | Multi-threading | OpenMP         | Shared-memory parallelism       |
| 4   | Multi-process   | MPI            | Distributed-memory parallelism  |
| 5   | GPU             | CUDA           | Massively parallel acceleration |

---

## Build Instructions

The project includes a unified `Makefile` targeting the **`krylov100`** lab cluster. All binaries are compiled with high-level optimizations (`-O3`, `-march=native`, `-ffast-math`).

**Compile all implementations:**

```bash
make all
```

This invokes the following compilation chain:

```bash
g++    -O3 -march=native -ffast-math -std=c++17          src/scaler_serial.cpp  -o scaler_serial
g++    -O3 -march=native -ffast-math -std=c++17 -mavx2   src/scaler_simd.cpp    -o scaler_simd
g++    -O3 -march=native -ffast-math -std=c++17 -fopenmp src/scaler_OpenMP.cpp  -o scaler_OpenMP
mpicxx -O3 -march=native -ffast-math -std=c++17          src/scaler_MPI.cpp     -o scaler_mpi
nvcc   -O3               -std=c++17  -arch=sm_70         src/scaler_cuda.cu     -o scaler_cuda
```

**Remove build artifacts:**

```bash
make clean
```

---
## Create the bin files
```bash
python3 scripts/generate_data.py --samples 1000000 --features 32 --output data/small.bin --dtype float64
```
```bash
python3 scripts/generate_data.py --samples 5000000 --features 64 --output data/medium.bin --dtype float64
```
---

## Running Benchmarks

### 1. Validate Output Correctness

Cross-validate execution output against a reference using the Python validation framework:

```bash
python3 scripts/validate.py data/small.bin out.bin 1000000 32 standard
```
```bash
python3 scripts/validate.py data/medium.bin out.bin 5000000 64 standard
```
```bash
python3 scripts/validate.py data/small.bin out.bin 1000000 32 minmax
```
```bash
python3 scripts/validate.py data/medium.bin out.bin 5000000 64 minmax
```

### 2. Run the Full Benchmark Suite

Execute multiple runs across all configurations and collect statistical timing measurements:

```bash
./benchmark.sh
```
or
```bash
bash scripts/benchmark.sh small both 3
```
```bash
bash scripts/benchmark.sh medium  both 3
```

### 3. Generate Plots and Summaries

Produce high-resolution `.png` charts and `.txt` summary tables into the `results/` directory:

```bash
python3 scripts/plot_visuals_results.py results/benchmark_small_20260619_183617.csv  results/
python3 scripts/plot_visuals_results.py results/benchmark_medium_20260619_183733.csv results/
```

---

## Project Structure

```
├── data/         # The small and medium bin files created
├── src/          # C++/CUDA source files (Serial, SIMD, OpenMP, MPI, CUDA)
├── scripts/      # Validation, parsing, and benchmarking utilities
├── results/      # CSV metrics, PNG charts, and evaluation summaries
├── gitignore
├── Makefile      # Build configuration
└── README.md
```

---

## Dependencies

Verified to compile and run on the `krylov100` lab cluster.

**Compilers**

- GCC / G++ v11+ with AVX2 support
- NVCC (CUDA Toolkit 11+)

**Parallel Frameworks**

- OpenMP API
- OpenMPI Suite

**Python Environment**

- Python 3.x
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

---

## Authors

**Μαρία Δρούγκα**  
Department of Computer Engineering & Informatics, University of Patras
