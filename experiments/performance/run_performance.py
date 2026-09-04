"""
Baseline and proposed-method accuracy on LongBench.

Methods and keep ratios:
  FullKV(100%), Random(10/30), StreamingLLM(10/20/30), Knorm(10/30),
  SnapKV(5/10/15/20/30/50), PyramidKV(5/10/15/20/30/50),
  AdaSnapKV(10/20/30, skipped on Qwen3), ProposedOnePassPress(eps 0.3/0.4/0.5/0.7/1.0)

Usage:
  quick check : CUDA_VISIBLE_DEVICES=0 python run_performance.py --model <m> --n 1 --tasks hotpotqa
  full run    : CUDA_VISIBLE_DEVICES=0 python run_performance.py --model <m> --n 100
Output: results/kvpress_performance/<model>_perf.json
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
from src.methods.proposed_kvpress import ProposedOnePassPress
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
is_qwen = "Qwen" in args.model

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

# Code-completion tasks use the raw prompt (no chat template)
def set_tmpl(task): tok.chat_template = None if task in RAW_TASKS else saved
def fin(ans, task): return ans if task in RAW_TASKS else ans.strip()

def keep2ratio(keep): return round(1.0 - keep, 4)

def run_task(task, make_press):
    """Run one task over all samples; return mean score, mean budget (if any), n."""
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

# Method specs: (name, keep -> press factory, [keep values])
def make_specs():
    specs = [("FullKV", None, [1.0]),
             ("Random", lambda k: kvpress.RandomPress(compression_ratio=keep2ratio(k)), [0.10,0.30]),
             ("StreamingLLM", lambda k: kvpress.StreamingLLMPress(compression_ratio=keep2ratio(k)), [0.10,0.20,0.30]),
             ("Knorm", lambda k: kvpress.KnormPress(compression_ratio=keep2ratio(k)), [0.10,0.30]),
             ("SnapKV", lambda k: kvpress.SnapKVPress(compression_ratio=keep2ratio(k)), [0.05,0.10,0.15,0.20,0.30,0.50]),
             ("PyramidKV", lambda k: kvpress.PyramidKVPress(compression_ratio=keep2ratio(k)), [0.05,0.10,0.15,0.20,0.30,0.50])]
    if not is_qwen:
        specs.append(("AdaSnapKV", lambda k: kvpress.AdaKVPress(kvpress.SnapKVPress(compression_ratio=keep2ratio(k))), [0.10,0.20,0.30]))
    return specs

EPSILONS = [0.3, 0.4, 0.5, 0.7, 1.0]
results = {}

print(f"=== {mname} | accuracy (n={args.n}, {len(TASKS)} tasks) ===\n")

for name, builder, keeps in make_specs():
    results[name] = {}
    for keep in keeps:
        tag = "FullKV" if name == "FullKV" else f"{name}_{int(keep*100)}"
        results[name][tag] = {"keep": keep, "tasks": {}}
        print(f"[{tag}]")
        for task in TASKS:
            mk = (lambda k=keep, b=builder: b(k)) if builder is not None else None
            t0 = time.time()
            sc, bud, ns = run_task(task, mk)
            results[name][tag]["tasks"][task] = {"score": sc, "budget": bud, "n": ns}
            print(f"    {task:12s} score={sc:.4f} (n={ns}, {time.time()-t0:.0f}s)")
        gc.collect(); torch.cuda.empty_cache()

results["Proposed"] = {}
for eps in EPSILONS:
    tag = f"Proposed_e{eps}"
    results["Proposed"][tag] = {"epsilon": eps, "tasks": {}}
    print(f"[{tag}]")
    for task in TASKS:
        t0 = time.time()
        sc, bud, ns = run_task(task, lambda e=eps: ProposedOnePassPress(epsilon=e))
        results["Proposed"][tag]["tasks"][task] = {"score": sc, "budget": bud, "n": ns}
        print(f"    {task:12s} score={sc:.4f} budget={bud if bud else 0:.4f} (n={ns}, {time.time()-t0:.0f}s)")
    gc.collect(); torch.cuda.empty_cache()

os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir, f"{mname}_perf.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {args.outdir}/{mname}_perf.json")