"""
Accuracy vs KV budget curves, with matched-score budget-saving arrows.

For each model, plots SnapKV / PyramidKV / Proposed and marks, at one reference
score, how much less budget the proposed method needs (arrow). Qwen3 has a
non-monotonic region, which is shaded and excluded from the arrow.

Run from the project root:
  python make_budget_curve.py
Reads:  results/kvpress_performance/{model}_perf.json
Writes: figures/fig_budget_score.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['font.size'] = 11
os.makedirs("figures", exist_ok=True)

MODELS = {
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
    "Qwen3-8B": "Qwen3-8B",
}
C = {"prop": "#2E7D32", "snap": "#00838F", "pyr": "#E65100", "full": "#1565C0"}

def load(p):
    if not os.path.exists(p):
        print(f"  [missing] {p}"); return None
    return json.load(open(p))

def curve(d, method_key, is_prop=False):
    pts = []
    if method_key not in d: return pts
    for tag, info in d[method_key].items():
        tasks = info["tasks"]
        score = np.mean([tasks[t]["score"] for t in tasks])
        if is_prop:
            buds = [tasks[t]["budget"] for t in tasks if tasks[t].get("budget")]
            bud = np.mean(buds) if buds else info.get("epsilon", 0)
        else:
            bud = info.get("keep", 0)
        pts.append((bud, score))
    return sorted(pts)

def budget_at_score(pts, target):
    """Budget needed to reach `target` score (linear interp; assumes monotone)."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if target < min(ys) or target > max(ys): return None
    for i in range(1, len(ys)):
        if (ys[i-1] <= target <= ys[i]) or (ys[i] <= target <= ys[i-1]):
            if ys[i] == ys[i-1]: return xs[i]
            return xs[i-1] + (xs[i]-xs[i-1]) * (target-ys[i-1]) / (ys[i]-ys[i-1])
    return None

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
axmap = {m: axes[i] for i, m in enumerate(MODELS.values())}

# Reference score for the matched-budget arrow
HLINE = {"Mistral-7B": 0.40, "LLaMA-3.1-8B": 0.32, "Qwen3-8B": 0.177}
# Non-monotonic budget region to shade
NONMONO = {"Qwen3-8B": (0.18, 0.24)}
# Arrows only for models with a monotone curve (Qwen3 excluded)
ARROW_MODELS = {"Mistral-7B", "LLaMA-3.1-8B"}

for mkey, mname in MODELS.items():
    d = load(f"results/kvpress_performance/{mkey}_perf.json")
    if d is None: continue
    ax = axmap[mname]
    if mname in NONMONO:
        lo, hi = NONMONO[mname]
        ax.axvspan(lo, hi, color="#EEEEEE", alpha=0.7, zorder=0)
    for key, lab, col, mk_ in [("SnapKV", "SnapKV", C["snap"], "s"),
                               ("PyramidKV", "PyramidKV", C["pyr"], "^"),
                               ("Proposed_M1", "Proposed", C["prop"], "o")]:
        pts = curve(d, key, key == "Proposed_M1")
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk_, label=lab,
                    color=col, lw=2, ms=6, zorder=3)
    if "FullKV" in d:
        fk = d["FullKV"].get("FullKV", {}).get("tasks", {})
        if fk:
            fs = np.mean([fk[t]["score"] for t in fk])
            ax.axhline(fs, color=C["full"], ls=":", lw=1.3, label="FullKV", alpha=0.6)
    tgt = HLINE[mname]
    if mname in ARROW_MODELS:
        sp = curve(d, "SnapKV"); pp = curve(d, "Proposed_M1", True)
        sb = budget_at_score(sp, tgt); pb = budget_at_score(pp, tgt)
        if sb and pb and pb < sb:
            y0, y1 = ax.get_ylim(); off = (y1-y0)*0.045
            ax.axhline(tgt, color="gray", ls="--", lw=1, alpha=0.6, zorder=1)
            ya = tgt + off*0.5
            ax.annotate("", xy=(pb, ya), xytext=(sb, ya),
                        arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.8), zorder=6)
            ax.text((sb+pb)/2, ya+off*0.5, f"-{(sb-pb)/sb*100:.0f}% budget\nat matched score",
                    ha="center", va="bottom", fontsize=8, color="#D32F2F", fontweight="bold", zorder=6)
            ax.scatter([pb, sb], [tgt, tgt], color=["#2E7D32", C["snap"]], s=45, zorder=5, edgecolor="white", lw=1)
    ax.set_title(mname, fontsize=11); ax.set_xlabel("KV budget (fraction kept)")
    if ax is axes[0]: ax.set_ylabel("LongBench score (5-task avg)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right"); ax.grid(alpha=0.25)
    if mname in NONMONO:
        lo, hi = NONMONO[mname]; y0, y1 = ax.get_ylim()
        ax.text((lo+hi)/2, y0+(y1-y0)*0.04, "non-\nmonotonic", fontsize=7,
                ha="center", va="bottom", color="#999", style="italic")

fig.suptitle("Score vs KV budget: at matched score, Proposed needs less budget (= less memory)",
             fontsize=12.5, y=1.00)
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.13, wspace=0.22)
fig.savefig("figures/fig_budget_score.png", dpi=200, bbox_inches="tight")
print("saved figures/fig_budget_score.png")