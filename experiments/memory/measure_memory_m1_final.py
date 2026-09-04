"""
Matched-performance KV cache memory for the final method vs SnapKV.

No GPU measurement: reads the accuracy-budget curves from the perf JSON
(Proposed_M1 and SnapKV) and, at each matched score, converts the keep ratio
to KV cache memory in GB (memory scales linearly with the kept fraction).
Uses only the monotonic region of each curve.

Run from the project root:
  python measure_memory_m1_final.py
Output: results/kvpress_memory_m1/<model>_m1mem_final.json
"""
import json, os, sys, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoConfig

MODELS = {"Mistral-7B-Instruct-v0.3": ("Mistral-7B", "mistralai/Mistral-7B-Instruct-v0.3"),
          "Llama-3.1-8B-Instruct": ("LLaMA-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
          "Qwen3-8B": ("Qwen3-8B", "Qwen/Qwen3-8B")}
TASKS = ["qasper", "hotpotqa", "gov_report", "trec", "lcc"]
CTX = 8000
os.makedirs("results/kvpress_memory_m1", exist_ok=True)

def curve(d, key, is_prop=False):
    pts = []
    if key not in d: return pts
    for tag, info in d[key].items():
        t = info["tasks"]; sc = np.mean([t[x]["score"] for x in TASKS if x in t])
        if is_prop:
            b = [t[x]["budget"] for x in t if t[x].get("budget")]; b = np.mean(b) if b else 0
        else:
            b = info.get("keep", 0)
        pts.append((b, sc))
    return sorted(pts)

def mono(pts):
    """Keep only the points where score is non-decreasing in budget."""
    if len(pts) < 2: return pts
    m = [pts[0]]
    for b, s in pts[1:]:
        if s >= m[-1][1]: m.append((b, s))
    return m

def interp(pts, tgt):
    pts = sorted(pts); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if not pts or tgt < min(ys) or tgt > max(ys): return None
    for i in range(1, len(ys)):
        if ys[i-1] <= tgt <= ys[i]:
            if ys[i] == ys[i-1]: return xs[i]
            return xs[i-1] + (xs[i]-xs[i-1]) * (tgt-ys[i-1]) / (ys[i]-ys[i-1])
    return None

def full_kv_gb(model_id):
    c = AutoConfig.from_pretrained(model_id)
    nl = c.num_hidden_layers; nkv = c.num_key_value_heads
    hd = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    return 2 * nl * nkv * hd * CTX * 2 / 1024**3  # 2 for K/V, 2 bytes for bf16

for mkey, (mname, mid) in MODELS.items():
    d = json.load(open(f"results/kvpress_performance/{mkey}_perf.json"))
    snap = mono(curve(d, "SnapKV")); m1 = mono(curve(d, "Proposed_M1", True))
    if not m1:
        print(f"{mname}: no Proposed_M1, skipping"); continue
    fg = full_kv_gb(mid)
    print(f"\n{'='*58}\n{mname} (FullKV KV {fg:.3f} GB, ctx {CTX})\n{'='*58}")
    print(f"{'score':>7}{'M1 bgt':>9}{'Snap bgt':>10}{'M1 GB':>9}{'Snap GB':>10}{'saving':>9}")
    pts = []
    for mb, ms in m1:
        sb = interp(snap, ms)
        if sb:
            mg = mb*fg; sg = sb*fg; sv = (sg-mg)/sg*100
            print(f"{ms:>7.3f}{mb:>9.3f}{sb:>10.3f}{mg:>9.3f}{sg:>10.3f}{sv:>+8.1f}%")
            pts.append({"score": ms, "m1_budget": mb, "snap_budget": sb,
                        "m1_gb": mg, "snap_gb": sg, "save_pct": sv})
    if pts:
        print(f"  mean saving: {np.mean([p['save_pct'] for p in pts]):+.1f}%")
    json.dump({"ctx": CTX, "full_kv_gb": fg, "points": pts},
              open(f"results/kvpress_memory_m1/{mname}_m1mem_final.json", "w"), indent=2)
print("\nSaved: results/kvpress_memory_m1/*_m1mem_final.json")