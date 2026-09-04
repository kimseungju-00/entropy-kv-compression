"""
Matched-performance KV budget saving vs SnapKV, for the role-weighted variant
(Proposed) and the final method (Proposed_M1), across the three models.

Reads accuracy-budget curves from the perf JSON and, at each matched score,
compares the budget each method needs against SnapKV. Uses only the monotonic
region of each curve. No GPU measurement.

Run from the project root:
  python compare_full_m1.py
"""
import json, os, sys, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODELS = {"Mistral-7B-Instruct-v0.3": "Mistral", "Llama-3.1-8B-Instruct": "LLaMA", "Qwen3-8B": "Qwen3"}
TASKS = ["qasper", "hotpotqa", "gov_report", "trec", "lcc"]

def curve_from(d, method_key, is_prop=False):
    pts = []
    if method_key not in d: return pts
    for tag, info in d[method_key].items():
        tasks = info["tasks"]
        sc = np.mean([tasks[t]["score"] for t in TASKS if t in tasks])
        if is_prop:
            buds = [tasks[t]["budget"] for t in tasks if tasks[t].get("budget")]
            b = np.mean(buds) if buds else 0
        else:
            b = info.get("keep", 0)
        pts.append((b, sc))
    return sorted(pts)

def monotonic(pts):
    if len(pts) < 2: return pts
    mono = [pts[0]]
    for b, s in pts[1:]:
        if s >= mono[-1][1]: mono.append((b, s))
    return mono

def interp_bud(pts, tgt):
    pts = sorted(pts); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if not pts or tgt < min(ys) or tgt > max(ys): return None
    for i in range(1, len(ys)):
        if ys[i-1] <= tgt <= ys[i]:
            if ys[i] == ys[i-1]: return xs[i]
            return xs[i-1] + (xs[i]-xs[i-1]) * (tgt-ys[i-1]) / (ys[i]-ys[i-1])
    return None

for mkey, mname in MODELS.items():
    d = json.load(open(f"results/kvpress_performance/{mkey}_perf.json"))
    snap = monotonic(curve_from(d, "SnapKV"))
    full = monotonic(curve_from(d, "Proposed", True))
    m1 = monotonic(curve_from(d, "Proposed_M1", True))
    print(f"\n{'='*60}\n{mname}\n{'='*60}")
    print(f"SnapKV curve: {[(round(b,3), round(s,3)) for b, s in snap]}")
    print(f"Proposed    : {[(round(b,3), round(s,3)) for b, s in full]}")
    print(f"Proposed_M1 : {[(round(b,3), round(s,3)) for b, s in m1]}")

    def savings(prop_pts, label):
        rows = []
        for pb, ps in prop_pts:
            sb = interp_bud(snap, ps)
            if sb: rows.append((ps, pb, sb, (sb-pb)/sb*100))
        if rows:
            avg = np.mean([r[3] for r in rows])
            print(f"\n  [{label} vs SnapKV, budget saving at matched score]")
            for ps, pb, sb, sv in rows:
                print(f"    score {ps:.3f}: {label} {pb:.3f} vs SnapKV {sb:.3f} -> {sv:+.1f}%")
            print(f"    -> {label} mean saving: {avg:+.1f}%")
            return avg
        print(f"  [{label}] no overlapping score range"); return None

    fa = savings(full, "Proposed"); ma = savings(m1, "Proposed_M1")
    if fa is not None and ma is not None:
        better = "Proposed_M1" if ma > fa else "Proposed"
        print(f"\n  ==> {mname}: Proposed {fa:+.1f}% vs Proposed_M1 {ma:+.1f}%  [{better} better]")