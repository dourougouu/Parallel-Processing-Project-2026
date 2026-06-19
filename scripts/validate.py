#!/usr/bin/env python3
"""
validate.py — Ελέγχει ορθότητα υλοποίησης vs Python/sklearn reference.

Χρήση:
    python3 scripts/validate.py <input.bin> <output.bin> <N> <D> <mode>

Παράδειγμα:
    python3 scripts/validate.py data/small.bin out.bin 1000000 32 standard
    python3 scripts/validate.py data/small.bin out.bin 1000000 32 minmax
"""

import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    if len(sys.argv) < 6:
        print(__doc__)
        sys.exit(1)

    input_bin  = sys.argv[1]
    output_bin = sys.argv[2]
    N          = int(sys.argv[3])
    D          = int(sys.argv[4])
    mode       = sys.argv[5].lower()

    if mode not in ("standard", "minmax"):
        print(f"[ERROR] mode must be 'standard' or 'minmax', got: {mode}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Validation: {mode.upper()} | N={N}, D={D}")
    print(f"  Input : {input_bin}")
    print(f"  Output: {output_bin}")
    print(f"{'='*55}")

    # Φόρτωση δεδομένων
    try:
        X = np.fromfile(input_bin,  dtype=np.float64).reshape(N, D)
        Y = np.fromfile(output_bin, dtype=np.float64).reshape(N, D)
    except Exception as e:
        print(f"[ERROR] Could not load files: {e}")
        sys.exit(1)

    # Reference από sklearn
    if mode == "standard":
        scaler = StandardScaler()
        Y_ref  = scaler.fit_transform(X)
    else:
        scaler = MinMaxScaler()
        Y_ref  = scaler.fit_transform(X)

    # Σύγκριση
    diff = np.abs(Y - Y_ref)
    max_err  = diff.max()
    mean_err = diff.mean()

    print(f"  Max  absolute error : {max_err:.2e}")
    print(f"  Mean absolute error : {mean_err:.2e}")

    # Κρίση
    THRESHOLD = 1e-9
    if max_err < THRESHOLD:
        print(f"  ✅ PASS (max error < {THRESHOLD})")
        return 0
    else:
        print(f"  ❌ FAIL (max error >= {THRESHOLD})")
        return 1

if __name__ == "__main__":
    sys.exit(main())