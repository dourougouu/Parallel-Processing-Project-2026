# Διαβάζει CSV απο το benchmark.sh και παραγει γραφήματα για οπτικοποίηση αποτελεσμάτων
# Χρήση
# python3 scripts/plot_visual_results.py results/benchmark_small_YYYYMMDD.csv results/
# Τα αποτελέσματα που παράγει είναι:
# ~ time_<dataset>_<mode>.png : Χρόνοι εκτέλεσης
# ~ speedup_<dataset>_<mode>.png : Speedup VS Serial 
# ~ scaling_<dataset>_<mode>.png : Strong scaling OpenMP/MPI
# ~ comparison_<dataset>.png : Σύγκριση
# ~ summary_<dataset>_<mode>.png : Πίνακας για το report

import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

mpl_order = ["serial", "simd", "openmp", "mpi", "cuda"]

mpl_labels = {"serial":"Serial",
              "simd":"SIMD\n(AVX2)",
              "openmp":"OpenMP",
              "mpi":"MPI", 
              "cuda":"CUDA"}

colors = { 
    "serial": "#607D8B",
    "simd": "#2196F3",
    "openmp": "#4CAF50",
    "mpi": "#FF9800",
    "cuda": "#9C27B0",
}

markers = {
    "serial": "o",
    "simd": "s",
    "openmp": "^",
    "mpi": "D",
    "cuda": "*"
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
    "figure.dpi":       120,
})

# I/O
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["time_secs"] = pd.to_numeric(df["time_secs"], errors="coerce")
    df["threads_or_procs"] = pd.to_numeric(df["threads_or_procs"], errors="coerce").fillna(1).astype(int)
    return df.dropna(subset=["time_secs"])

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["impl", "variant", "mode", "threads_or_procs", "dataset", "N", "D"],
                as_index=False)
              .agg(avg_time=("time_secs", "mean"),
                   min_time=("time_secs", "min"),
                   max_time=("time_secs", "max"),
                   runs=("time_secs", "count")
                )
            )

def compute_speedup(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, dataset), g in agg.groupby(["mode", "dataset"]):
        ser = g[g["impl"]=="serial"]
        if ser.empty:
            continue
        
        serial_t = ser["avg_time"].values[0]
        for _, r in g.iterrows():
            rows.append({**r.to_dict(), "speedup": serial_t/r["avg_time"], "serial_time": serial_t})
    return pd.DataFrame(rows)

# 1ο Γράφημα: Execution Time
def plot_time(df_speed: pd.DataFrame, out_dir:str, mode: str, dataset: str):
    sub = df_speed[ (df_speed["mode"]==mode)  &  (df_speed["dataset"]==dataset) ].copy()
    
    if sub.empty:
        return

    def make_label(r):
        if r["impl"] in ("serial", "simd", "cuda"):
            return mpl_labels.get(r["impl"], r["impl"])
        return f"{r['impl'].upper()}\n×{int(r['threads_or_procs'])}"
    

    sub["label"] = sub.apply(make_label, axis=1)
    sub = sub.sort_values(["impl","threads_or_procs"])
 
    fig, ax = plt.subplots(figsize=(max(10, len(sub)*0.9), 5))
    x = np.arange(len(sub))
    bar_colors_list = [colors.get(r["impl"],"#aaa") for _,r in sub.iterrows()]
 
    bars = ax.bar(x, sub["avg_time"], color=bar_colors_list,
                  edgecolor="white", linewidth=0.6, zorder=3)
 
    # Error bars (min/max)
    yerr_lo = sub["avg_time"] - sub["min_time"]
    yerr_hi = sub["max_time"] - sub["avg_time"]
    ax.errorbar(x, sub["avg_time"],
                yerr=[yerr_lo, yerr_hi],
                fmt="none", color="#333", capsize=4, linewidth=1.2, zorder=4)
 
    # Τιμές πάνω από μπάρες
    for bar, val in zip(bars, sub["avg_time"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+bar.get_height()*0.02,
                f"{val:.3f}s", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
 
    ax.set_xticks(x)
    ax.set_xticklabels(sub["label"], fontsize=9)
    ax.set_ylabel("Χρόνος εκτέλεσης (s)", fontsize=11)
    ax.set_title(f"Χρόνος εκτέλεσης — {mode.capitalize()} Scaler | "
                 f"{dataset.capitalize()} dataset", fontsize=13, fontweight="bold", pad=12)
 
    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(color=colors[k], label=mpl_labels.get(k,k))
                  for k in mpl_order if k in sub["impl"].values]
    ax.legend(handles=legend_els, fontsize=9, loc="upper right", framealpha=0.9)
 
    fig.tight_layout()
    fname = os.path.join(out_dir, f"time_{dataset}_{mode}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")
 
 
# ── Γράφημα 2: Speedup line chart ────────────────────────────────────────────
def plot_speedup(df_sp: pd.DataFrame, out_dir: str, mode: str, dataset: str):
    sub = df_sp[(df_sp["mode"]==mode) & (df_sp["dataset"]==dataset)].copy()
    if sub.empty:
        return
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    max_threads = 1
    for impl, grp in sub.groupby("impl"):
        if impl == "serial":
            continue
        grp = grp.sort_values("threads_or_procs")
        x = grp["threads_or_procs"].astype(int).values
        y = grp["speedup"].values
        c = colors.get(impl,"#999")
        m = markers.get(impl,"o")
        ax.plot(x, y, marker=m, color=c, label=mpl_labels.get(impl,impl),
                linewidth=2.2, markersize=8, zorder=4)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.2f}×", (xi, yi),
                        textcoords="offset points", xytext=(6,5),
                        fontsize=8.5, color=c, fontweight="bold")
        max_threads = max(max_threads, int(x.max()))
 
    # Ideal speedup
    if max_threads > 1:
        ideal = np.array([1, max_threads])
        ax.plot(ideal, ideal, "k--", linewidth=1, alpha=0.35, label="Ideal", zorder=2)
 
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Threads / Processes", fontsize=11)
    ax.set_ylabel("Speedup (× vs Serial)", fontsize=11)
    ax.set_title(f"Speedup — {mode.capitalize()} Scaler | {dataset.capitalize()} dataset",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
 
    # X-axis: log2 scale αν έχει πολλές τιμές
    if max_threads >= 4:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
 
    fig.tight_layout()
    fname = os.path.join(out_dir, f"speedup_{dataset}_{mode}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")
 
 
# ── Γράφημα 3: Strong Scaling (OpenMP & MPI μαζί) ─────────────────────────────
def plot_strong_scaling(df_sp: pd.DataFrame, out_dir: str, mode: str, dataset: str):
    sub = df_sp[(df_sp["mode"]==mode) & (df_sp["dataset"]==dataset)
                & (df_sp["impl"].isin(["openmp","mpi"]))].copy()
    if sub.empty:
        return
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
 
    for ax, col, ylabel, fmt in [
            (ax1, "avg_time", "Χρόνος εκτέλεσης (s)", ".3f"),
            (ax2, "speedup",  "Speedup (× vs Serial)", ".2f")]:
 
        max_t = 1
        for impl, grp in sub.groupby("impl"):
            grp = grp.sort_values("threads_or_procs")
            x = grp["threads_or_procs"].astype(int).values
            y = grp[col].values
            c = colors[impl]
            m = markers[impl]
            ax.plot(x, y, marker=m, color=c, label=mpl_labels[impl],
                    linewidth=2.2, markersize=8)
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:{fmt}}", (xi, yi),
                            textcoords="offset points", xytext=(5,5),
                            fontsize=8, color=c)
            max_t = max(max_t, int(x.max()))
 
        if col == "speedup" and max_t > 1:
            ideal = np.array([1, max_t])
            ax.plot(ideal, ideal, "k--", lw=1, alpha=0.35, label="Ideal")
 
        ax.set_xlabel("Threads / Processes", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        if max_t >= 4:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.legend(fontsize=10)
 
    fig.suptitle(f"Strong Scaling — {mode.capitalize()} | {dataset.capitalize()} dataset",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"scaling_{dataset}_{mode}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")
 
 
# ── Γράφημα 4: Standard vs MinMax σύγκριση (best per impl) ───────────────────
def plot_mode_comparison(df_sp: pd.DataFrame, out_dir: str, dataset: str):
    modes_avail = df_sp["mode"].unique()
    if len(modes_avail) < 2:
        return
 
    rows = []
    for mode in ["standard","minmax"]:
        sub = df_sp[(df_sp["mode"]==mode) & (df_sp["dataset"]==dataset)]
        if sub.empty:
            continue
        for impl, g in sub.groupby("impl"):
            best = g.loc[g["avg_time"].idxmin()]
            rows.append({"impl":impl,"mode":mode,
                         "time":best["avg_time"],"speedup":best["speedup"]})
    if not rows:
        return
 
    cmp = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    bar_colors = {"standard":"#42A5F5","minmax":"#EF5350"}
 
    for ax, col, ylabel, fmt in [
            (axes[0], "time",    "Χρόνος εκτέλεσης (s)", ".3f"),
            (axes[1], "speedup", "Speedup (× vs Serial)", ".2f")]:
 
        pivot = cmp.pivot(index="impl", columns="mode", values=col)
        pivot = pivot.reindex([i for i in mpl_order if i in pivot.index])
 
        x = np.arange(len(pivot))
        width = 0.35
        for idx, (m, c) in enumerate(bar_colors.items()):
            if m not in pivot.columns:
                continue
            vals = pivot[m].values
            offset = (idx - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=m.capitalize(),
                          color=c, edgecolor="white", linewidth=0.7)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()+bar.get_height()*0.02,
                            f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)
 
        ax.set_xticks(x)
        ax.set_xticklabels([mpl_labels.get(i,i) for i in pivot.index],
                           rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Standard vs MinMax — {ylabel}", fontsize=11, pad=8)
        ax.legend(title="Mode", fontsize=9)
 
    fig.suptitle(f"Σύγκριση Scaler Modes — {dataset.capitalize()} dataset",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"comparison_{dataset}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")
 
 
# ── Πίνακας κειμένου για report ───────────────────────────────────────────────
def save_text_table(df_sp: pd.DataFrame, out_dir: str, mode: str, dataset: str):
    sub = df_sp[(df_sp["mode"]==mode) & (df_sp["dataset"]==dataset)].copy()
    if sub.empty:
        return
    sub = sub.sort_values(["impl","threads_or_procs"])
 
    lines = []
    lines.append(f"Αποτελέσματα: {mode.capitalize()} Scaler | {dataset.capitalize()} dataset")
    lines.append("="*72)
    lines.append(f"{'Υλοποίηση':<12} {'Threads/Procs':<15} {'Avg Time (s)':<15} "
                 f"{'Min (s)':<12} {'Max (s)':<12} {'Speedup':<10} {'Runs'}")
    lines.append("-"*72)
    for _, r in sub.iterrows():
        lines.append(
            f"{r['impl']:<12} {int(r['threads_or_procs']):<15} "
            f"{r['avg_time']:<15.4f} {r['min_time']:<12.4f} "
            f"{r['max_time']:<12.4f} {r['speedup']:<10.3f} {int(r['runs'])}")
    lines.append("="*72)
 
    fname = os.path.join(out_dir, f"summary_{dataset}_{mode}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ {fname}")
    # Και στο terminal
    print("\n" + "\n".join(lines))
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
 
    csv_path = sys.argv[1]
    out_dir  = sys.argv[2]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
 
    print(f"\n[plot_results] Loading: {csv_path}")
    df_raw = load_csv(csv_path)
    if df_raw.empty:
        print("[ERROR] CSV is empty or could not be parsed.")
        sys.exit(1)
 
    df_agg = aggregate(df_raw)
    df_sp  = compute_speedup(df_agg)
 
    print(f"[plot_results] {len(df_raw)} runs, "
          f"{df_raw['impl'].nunique()} impls, "
          f"{df_raw['mode'].nunique()} modes\n")
 
    datasets = df_sp["dataset"].unique()
    modes    = df_sp["mode"].unique()
 
    for dataset in datasets:
        for mode in modes:
            print(f"\n── {mode.upper()} | {dataset.upper()} ──")
            plot_time(df_sp, out_dir, mode, dataset)
            plot_speedup(df_sp, out_dir, mode, dataset)
            plot_strong_scaling(df_sp, out_dir, mode, dataset)
            save_text_table(df_sp, out_dir, mode, dataset)
        plot_mode_comparison(df_sp, out_dir, dataset)
 
    print(f"\n[plot_results] ✓ All outputs saved → {out_dir}/")
 
 
if __name__ == "__main__":
    main()