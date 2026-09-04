"""
Accuracy of the final method (EntropyBudgetPress) on LongBench.

Reuses the exact evaluation helpers from run_performance.py; only the method
is EntropyBudgetPress (mean-entropy budget). Baselines are not re-measured.
Results are merged into results/kvpress_performance/<model>_perf.json under the
"Proposed_M1" key, alongside the existing baseline and ProposedOnePassPress runs.

Usage:
  quick check : CUDA_VISIBLE_DEVICES=0 python run_performance_m1.py --model <m> --n 1 --tasks hotpotqa
  full run    : CUDA_VISIBLE_DEVICES=0 python run_performance_m1.py --model <m> --n 100
"""
import sys, os, argparse, json, time, gc
import numpy as np
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import transformers
transformers.logging.set_verbosity_error()
import logging
logging.getLogger("kvpress").setLevel(logging.ERROR)
from transformers import pipeline
import kvpress
from src.methods.proposed_kvpress import EntropyBudgetPress
from src.eval.longbench_loader import load_task
from src.eval.evaluator import score_sample, TASK_MAX_NEW, DATASET2PROMPT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--tasks", nargs="+", default=["qasper","hotpotqa","gov_report","trec","lcc"])
ap.add_argument("--max_seq", type=int, default=10000)
ap.add_argument("--outdir", default="results/kvpress_performance")
args = ap.parse_args()
RAW_TASKS = {"lcc", "repobench-p"}
TASKS = args.tasks
mname = args.model.split("/")[-1]

def split(s, task):
    tmpl = DATASET2PROMPT[task]; pre, post = tmpl.split("{context}", 1)
    return pre + s["context"], post.replace("{input}", s.get("input",""))
def truncate(tok, ctx, q, ms):
    qi = tok(q, add_special_tokens=False)["input_ids"]; b = ms - len(qi) - 64
    ci = tok(ctx, add_special_tokens=False)["input_ids"]
    if len(ci) > b:
        h = b//2; ci = ci[:h]+ci[-h:]; ctx = tok.decode(ci, skip_special_tokens=True)
    return ctx, q

print(f"Loading {args.model}  [GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}]")
pipe = pipeline("kv-press-text-generation", model=args.model, device="cuda",
                torch_dtype=torch.bfloat16, model_kwargs={"attn_implementation":"flash_attention_2"})
tok = pipe.tokenizer
saved = tok.chat_template
if tok.bos_token is None: tok.bos_token = ""
print("Loaded\n")

def set_tmpl(task): tok.chat_template = None if task in RAW_TASKS else saved
def fin(ans, task): return ans if task in RAW_TASKS else ans.strip()

def run_task(task, make_press):
    """Same as run_performance.py."""
    set_tmpl(task); mn = TASK_MAX_NEW.get(task, 32)
    sc, budgets = [], []
    press = make_press() if make_press is not None else None
    if press is not None and hasattr(press, "_budget_log"):
        press._budget_log = []
    for s in load_task(task, max_samples=args.n):
        ctx, q = split(s, task); ctx, q = truncate(tok, ctx, q, args.max_seq)
        try:
            kw = {"press": press} if press is not None else {}
            ans = pipe(ctx, question=q, max_new_tokens=mn, **kw)["answer"]
            a = s.get("answers", []); a = [a] if isinstance(a, str) else a
            sc.append(score_sample(fin(ans, task), a, task, s.get("all_classes")))
        except Exception as e:
            print(f"      sample error: {type(e).__name__} {str(e)[:50]}")
            continue
    avg_sc = float(np.mean(sc)) if sc else 0.0
    avg_bud = None
    if press is not None and hasattr(press, "_budget_log") and press._budget_log:
        avg_bud = float(np.mean(press._budget_log))
    torch.cuda.empty_cache()
    return avg_sc, avg_bud, len(sc)

EPSILONS = [0.3, 0.4, 0.5, 0.7, 1.0]

# Load existing perf JSON (if any) and add/replace the "Proposed_M1" key
outpath = os.path.join(args.outdir, f"{mname}_perf.json")
if os.path.exists(outpath):
    with open(outpath) as f: results = json.load(f)
    print(f"Loaded existing perf JSON: {outpath} (methods: {list(results.keys())})")
else:
    results = {}
    print("No existing perf JSON, creating new")

results["Proposed_M1"] = {}
print(f"\n=== {mname} | final method accuracy (n={args.n}) ===\n")
for eps in EPSILONS:
    tag = f"Proposed_M1_e{eps}"
    results["Proposed_M1"][tag] = {"epsilon": eps, "tasks": {}}
    print(f"[{tag}]")
    for task in TASKS:
        t0 = time.time()
        sc, bud, ns = run_task(task, lambda e=eps: EntropyBudgetPress(epsilon=e))
        results["Proposed_M1"][tag]["tasks"][task] = {"score": sc, "budget": bud, "n": ns}
        print(f"    {task:12s} score={sc:.4f} budget={bud if bud else 0:.4f} (n={ns}, {time.time()-t0:.0f}s)")
    gc.collect(); torch.cuda.empty_cache()

os.makedirs(args.outdir, exist_ok=True)
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved (merged): {outpath}")