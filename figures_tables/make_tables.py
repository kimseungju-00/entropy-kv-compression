"""
Generate paper tables (LaTeX .tex and Markdown .md) from the result JSON files.

Run from the project root:
  python make_tables.py

Tables:
  perf_full_<model>       - full accuracy grid (all methods x budgets x 5 tasks)
  perf_main_budget_<model>- accuracy at a matched budget (~20%)
  perf_main_matched_<model>- KV budget needed for a matched score
  memory                  - matched-performance KV memory (final method vs SnapKV)
  theory_tvdelta          - TV=delta check and delta-entropy correlation
  layer_hetero            - layer entropy heterogeneity
  efficiency_<model>      - latency and memory by context length
  ablation_<model>        - component analysis
"""
import json, os, numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs("tables", exist_ok=True)
MODELS = {
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
    "Qwen3-8B": "Qwen3-8B",
}
TASKS = ["qasper", "hotpotqa", "gov_report", "trec", "lcc"]
TASK_LABEL = {"qasper": "Qasper", "hotpotqa": "HotpotQA", "gov_report": "GovReport", "trec": "TREC", "lcc": "LCC"}

def load(p):
    if not os.path.exists(p): return None
    return json.load(open(p))

def save(name, tex, md):
    with open(f"tables/{name}.tex", "w") as f: f.write(tex)
    with open(f"tables/{name}.md", "w") as f: f.write(md)
    print(f"saved tables/{name}.tex, tables/{name}.md")

def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join(["---"]*len(headers)) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out

def tex_table(headers, rows, caption, label, align=None):
    ncol = len(headers); align = align or ("l" + "r"*(ncol-1))
    out = "\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{tab:" + label + "}\n"
    out += "\\begin{tabular}{" + align + "}\n\\toprule\n"
    out += " & ".join(headers) + " \\\\\n\\midrule\n"
    for r in rows:
        out += " & ".join(str(c) for c in r) + " \\\\\n"
    out += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return out

# --- Full accuracy grid (per model) ---
def perf_tables():
    alld = {}
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_performance/{mkey}_perf.json")
        if d: alld[mname] = d
    if not alld: print("  [perf tables skipped] no perf files"); return
    for mname, d in alld.items():
        headers = ["Method", "Budget"] + [TASK_LABEL[t] for t in TASKS] + ["Avg"]
        rows = []
        for method in ["FullKV", "Random", "StreamingLLM", "SnapKV", "Knorm", "PyramidKV", "AdaSnapKV", "Proposed_M1"]:
            if method not in d: continue
            for tag, info in sorted(d[method].items(), key=lambda x: x[1].get("keep", x[1].get("epsilon", 0))):
                tasks = info["tasks"]
                scores = [tasks[t]["score"] for t in TASKS if t in tasks]
                avg = np.mean(scores)
                if method == "Proposed_M1":
                    buds = [tasks[t]["budget"] for t in tasks if tasks[t].get("budget")]
                    bud = f"{np.mean(buds):.3f}" if buds else f"eps{info.get('epsilon','')}"
                elif method == "FullKV":
                    bud = "1.000"
                else:
                    bud = f"{info.get('keep',0):.3f}"
                row = [method, bud] + [f"{tasks[t]['score']:.4f}" if t in tasks else "-" for t in TASKS] + [f"{avg:.4f}"]
                rows.append(row)
        cap = f"Accuracy on LongBench ({mname}), all methods and budgets (100 samples/task)."
        save(f"perf_full_{mname}",
             tex_table(headers, rows, cap, f"perf_full_{mname.replace('.','').replace('-','')}"),
             f"### Accuracy - {mname} (full)\n\n" + md_table(headers, rows))

def memory_table():
    headers = ["Model", "Matched score", "Proposed (GB)", "SnapKV (GB)", "Reduction"]
    rows = []
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_memory_m1/{mname}_m1mem_final.json")
        if not d: continue
        pts = d["points"]
        for p in pts:
            rows.append([mname, f"{p['score']:.4f}", f"{p['m1_gb']:.3f}",
                         f"{p['snap_gb']:.3f}", f"{p['save_pct']:+.1f}\\%"])
        avg = np.mean([p["save_pct"] for p in pts])
        rows.append([f"\\textbf{{{mname} avg}}", "", "", "", f"\\textbf{{{avg:+.1f}\\%}}"])
    if not rows: print("  [memory table skipped]"); return
    cap = ("KV cache memory at matched performance: Proposed vs SnapKV (8k context). "
           "Qwen3 has a narrow score range (0.16-0.18); its lowest-score point yields an "
           "unreliable interpolation and is reported for transparency.")
    md_rows = [[c.replace("\\%", "%").replace("\\textbf{", "**").replace("}", "**") if "\\" in str(c) else c for c in r] for r in rows]
    save("memory", tex_table(headers, rows, cap, "memory"),
         "### KV Cache Memory Reduction at Matched Performance\n\n" + md_table(headers, md_rows))

def theory_table():
    headers = ["Model", "rho", "|TV-delta| max", "Pearson(delta,H)"]
    rows = []
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_theory/{mkey}_delta.json")
        if not d: continue
        res = d.get("results", d)
        for r in sorted([k for k in res.keys() if k != "window"], key=float):
            info = res[r]
            gap = info.get("tv_delta_gap", 0)
            rows.append([mname, r, f"{gap:.1e}", f"{info['pearson_delta_H']:.3f}"])
    if not rows: print("  [theory table skipped]"); return
    cap = "Theory verification: TV=delta identity (|TV-delta|) and delta-entropy correlation across compression ratios."
    save("theory_tvdelta", tex_table(headers, rows, cap, "theory"),
         "### Theory Verification (TV=delta and delta-entropy correlation)\n\n" + md_table(headers, rows))

def layer_table():
    headers = ["Model", "Layers", "Min entropy (layer)", "Max entropy (layer)", "Max/Min ratio"]
    rows = []
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_theory/{mkey}_layerent.json")
        if not d: continue
        rows.append([mname, d.get("n_layers", len(d.get("layer_entropy", []))),
                     f"{d['min']:.3f} (L{d['min_layer']})",
                     f"{d['max']:.3f} (L{d['max_layer']})",
                     f"{d['ratio']:.2f}x"])
    if not rows: print("  [layer table skipped]"); return
    cap = "Layer-wise attention entropy heterogeneity."
    save("layer_hetero", tex_table(headers, rows, cap, "layerhetero"),
         "### Layer-wise Entropy Heterogeneity\n\n" + md_table(headers, rows))

def efficiency_table():
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_efficiency_m1/{mkey}.json")
        if not d: continue
        res = d["results"]
        lengths = sorted(res.keys(), key=int)
        methods = ["FullKV", "SnapKV_20", "Proposed_M1"]
        headers = ["Context", "Method", "Latency (s)", "KV+act (GB)", "vs FullKV mem"]
        rows = []
        for L in lengths:
            full_mem = res[L].get("FullKV", {}).get("kv_act_gb", None)
            for m in methods:
                if m not in res[L]: continue
                info = res[L][m]
                mem = info["kv_act_gb"]
                vs = f"{(1-mem/full_mem)*100:+.0f}\\%" if (full_mem and m != "FullKV") else "-"
                rows.append([L if m == "FullKV" else "", m,
                             f"{info['latency_s']:.3f}", f"{mem:.3f}", vs])
        cap = (f"Efficiency ({mname}): latency and KV+activation memory by context length "
               f"(64-token decode, eps={d.get('epsilon','')}). Proposed matches SnapKV latency "
               "while reducing memory vs FullKV.")
        md_rows = [[str(c).replace("\\%", "%").replace("-", "-") for c in r] for r in rows]
        save(f"efficiency_{mname}",
             tex_table(headers, rows, cap, f"eff_{mname.replace('.','').replace('-','')}"),
             f"### Efficiency - {mname} (latency & memory by context length)\n\n" + md_table(headers, md_rows))

# --- Representative accuracy at a matched budget (~20%) ---
def perf_main_fixed_budget(target_budget=0.20):
    order = ["FullKV", "Random", "StreamingLLM", "SnapKV", "Knorm", "PyramidKV", "AdaSnapKV", "Proposed_M1"]
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_performance/{mkey}_perf.json")
        if not d: continue
        headers = ["Method", "Budget"] + [TASK_LABEL[t] for t in TASKS] + ["Avg"]
        rows = []
        for method in order:
            if method not in d: continue
            best = None; bestdiff = 1e9
            for tag, info in d[method].items():
                tasks = info["tasks"]
                if method == "Proposed_M1":
                    buds = [tasks[t]["budget"] for t in tasks if tasks[t].get("budget")]
                    b = np.mean(buds) if buds else info.get("epsilon", 0)
                elif method == "FullKV":
                    b = 1.0
                else:
                    b = info.get("keep", 0)
                if method == "FullKV":
                    best = (b, info); break
                diff = abs(b - target_budget)
                if diff < bestdiff: bestdiff = diff; best = (b, info)
            if best is None: continue
            b, info = best
            tasks = info["tasks"]
            avg = np.mean([tasks[t]["score"] for t in TASKS if t in tasks])
            row = [method, f"{b:.3f}"] + [f"{tasks[t]['score']:.4f}" if t in tasks else "-" for t in TASKS] + [f"{avg:.4f}"]
            rows.append(row)
        cap = f"Accuracy at matched budget (~{int(target_budget*100)}\\%) on LongBench ({mname}, 100 samples)."
        save(f"perf_main_budget_{mname}",
             tex_table(headers, rows, cap, f"perfmain_{mname.replace('.','').replace('-','')}"),
             f"### Accuracy at ~{int(target_budget*100)}% budget - {mname}\n\n" + md_table(headers, rows))

# --- Representative: KV budget needed for a matched score (monotonic region) ---
def perf_main_matched_score():
    def monotonic_curve(pts):
        pts = sorted(pts)
        if len(pts) < 2: return pts
        mono = [pts[0]]
        for b, s in pts[1:]:
            if s >= mono[-1][1]: mono.append((b, s))
        return mono
    def interp_budget(pts, target):
        pts = sorted(pts); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if not pts or target < min(ys) or target > max(ys): return None
        for i in range(1, len(ys)):
            if (ys[i-1] <= target <= ys[i]) or (ys[i] <= target <= ys[i-1]):
                if ys[i] == ys[i-1]: return xs[i]
                return xs[i-1] + (xs[i]-xs[i-1]) * (target-ys[i-1]) / (ys[i]-ys[i-1])
        return None
    def curve(d, method):
        pts = []
        if method not in d: return pts
        for tag, info in d[method].items():
            tasks = info["tasks"]
            score = np.mean([tasks[t]["score"] for t in TASKS if t in tasks])
            if method == "Proposed_M1":
                buds = [tasks[t]["budget"] for t in tasks if tasks[t].get("budget")]
                b = np.mean(buds) if buds else 0
            else:
                b = info.get("keep", 0)
            pts.append((b, score))
        return pts
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_performance/{mkey}_perf.json")
        if not d: continue
        pp_mono = monotonic_curve(curve(d, "Proposed_M1"))
        if not pp_mono: continue
        methods = ["SnapKV", "PyramidKV", "Proposed_M1"]
        mono_curves = {m: monotonic_curve(curve(d, m)) for m in methods}
        targets = sorted(set(round(s, 3) for _, s in pp_mono))
        headers = ["Matched score"] + methods + ["Prop vs SnapKV"]
        rows = []
        for tgt in targets:
            row = [f"{tgt:.3f}"]; buds = {}
            for m in methods:
                b = interp_budget(mono_curves[m], tgt); buds[m] = b
                row.append(f"{b:.3f}" if b else "-")
            if buds.get("SnapKV") and buds.get("Proposed_M1"):
                red = (buds["SnapKV"] - buds["Proposed_M1"]) / buds["SnapKV"] * 100
                row.append(f"{red:+.1f}" + chr(92) + "%")
            else:
                row.append("-")
            rows.append(row)
        cap = f"KV budget needed for matched score ({mname}, monotonic region): Proposed vs baselines (lower budget = less memory)."
        md_rows = [[str(c).replace(chr(92)+"%", "%") for c in r] for r in rows]
        save(f"perf_main_matched_{mname}",
             tex_table(headers, rows, cap, f"perfmatch_{mname.replace('.','').replace('-','')}"),
             f"### KV budget for matched score - {mname} (monotonic region)" + chr(10)+chr(10) + md_table(headers, md_rows))

def ablation_table():
    ABTASKS = ["qasper", "hotpotqa", "gov_report", "trec", "lcc"]
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_ablation/{mkey}_abl.json")
        if not d: continue
        res = d["results"]
        variants = [("Full", "Full (proposed)"), ("WO_Head", "w/o head classification"),
                    ("WO_Adapt", "w/o adaptive budget$^\\dagger$")]
        headers = ["Variant"] + [TASK_LABEL[t] for t in ABTASKS] + ["Avg", "Budget"]
        rows = []
        for key, label in variants:
            if key not in res: continue
            r = res[key]
            scores = [r[t]["score"] for t in ABTASKS if t in r]
            buds = [r[t]["budget"] for t in ABTASKS if t in r and r[t].get("budget")]
            avg = np.mean(scores); bud = np.mean(buds) if buds else 0
            row = [label] + [f"{r[t]['score']:.4f}" if t in r else "-" for t in ABTASKS] + [f"{avg:.4f}", f"{bud:.3f}"]
            rows.append(row)
        cap = (f"Component analysis ({mname}, " + chr(949) + f"=0.5, 100 samples). "
               "At matched budget, removing the adaptive components does not reduce accuracy "
               "(w/o adaptive budget performs on par with the full method), indicating that these "
               "components determine the per-input budget automatically rather than improving raw "
               "accuracy. Removing head classification raises both accuracy and budget, i.e. it "
               "regulates the budget size. The role of the adaptive components is thus automatic "
               "budget selection; their memory benefit is quantified separately in the "
               "matched-performance memory comparison. "
               "$^\\dagger$w/o layer differentiation is identical to w/o adaptive budget "
               "(both reduce to a single per-input budget in our 1-pass design).")
        md_rows = [[str(c).replace("$^\\dagger$", "(dagger)").replace(chr(92)+"dagger", "") for c in r] for r in rows]
        md_cap = ("Note: at matched budget, removing the adaptive components does not reduce accuracy "
                  "(w/o adaptive budget is on par with full), so these components perform automatic "
                  "per-input budget selection rather than improving raw accuracy. Removing head "
                  "classification raises both accuracy and budget (it regulates budget size). Their "
                  "memory benefit is quantified separately in the matched-performance memory "
                  "comparison. (dagger: w/o layer differentiation = w/o adaptive budget in the 1-pass design.)")
        save(f"ablation_{mname}",
             tex_table(headers, rows, cap, f"abl_{mname.replace('.','').replace('-','')}"),
             f"### Component Analysis (Ablation) - {mname}" + chr(10)+chr(10) + md_table(headers, md_rows) + chr(10) + md_cap + chr(10))

if __name__ == "__main__":
    print("=== generating tables ===")
    perf_tables()
    memory_table()
    theory_table()
    layer_table()
    efficiency_table()
    perf_main_fixed_budget()
    perf_main_matched_score()
    ablation_table()
    print("=== done. see tables/ ===")