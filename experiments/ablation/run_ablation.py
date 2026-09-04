"""
Component ablation of ProposedOnePassPress.

Full = ProposedOnePassPress (entropy token selection + head classification +
adaptive per-layer budget). Variants remove one component each:
  WO_Head  : remove head-role classification (uniform head weights)
  WO_Adapt : remove the adaptive budget (fixed keep ratio)
  WO_Layer : remove layer differentiation (same keep ratio for all layers)

All variants are compared at the same epsilon; WO_Adapt and WO_Layer use the
mean budget of Full as their fixed keep ratio, so the comparison is at matched
budget. Each variant's realized budget is also recorded.

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_ablation.py --model <m> --n 100 --eps 0.5
Output: results/kvpress_ablation/<model>_abl.json
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
from dataclasses import dataclass

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--eps", type=float, default=0.5)
ap.add_argument("--tasks", nargs="+", default=["qasper","hotpotqa","gov_report","trec","lcc"])
ap.add_argument("--max_seq", type=int, default=10000)
ap.add_argument("--outdir", default="results/kvpress_ablation")
args = ap.parse_args()

RAW_TASKS = {"lcc", "repobench-p"}

# Ablation variants: inherit ProposedOnePassPress, override _layer_keep_ratio only.

@dataclass
class WOHeadPress(ProposedOnePassPress):
    """Remove head-role classification: budget from the plain mean entropy."""
    def _layer_keep_ratio(self, entropy):
        H_norm = entropy / (entropy.max() + 1e-9)
        phi = H_norm.mean().item()
        g = phi ** (1.0 / max(self.epsilon, 0.01))
        keep = self.min_ratio + (self.max_ratio - self.min_ratio) * g
        return float(np.clip(keep, self.min_ratio, self.max_ratio))

@dataclass
class WOAdaptPress(ProposedOnePassPress):
    """Remove the adaptive budget: use a fixed keep ratio (set at runtime)."""
    fixed_keep: float = 0.20
    def _layer_keep_ratio(self, entropy):
        return float(self.fixed_keep)

@dataclass
class WOLayerPress(ProposedOnePassPress):
    """Remove layer differentiation: same keep ratio for every layer."""
    fixed_keep: float = 0.20
    def _layer_keep_ratio(self, entropy):
        return float(self.fixed_keep)

mname = args.model.split("/")[-1]
print(f"Loading {args.model}  [GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}]")
pipe = pipeline("kv-press-text-generation", model=args.model, device="cuda",
                torch_dtype=torch.bfloat16, model_kwargs={"attn_implementation":"flash_attention_2"})
tok = pipe.tokenizer
saved = tok.chat_template
if tok.bos_token is None: tok.bos_token = ""
print("Loaded\n")

def split(s, task):
    tmpl = DATASET2PROMPT[task]; pre, post = tmpl.split("{context}", 1)
    return pre + s["context"], post.replace("{input}", s.get("input",""))
def truncate(ctx, q, ms):
    qi = tok(q, add_special_tokens=False)["input_ids"]; b = ms - len(qi) - 64
    ci = tok(ctx, add_special_tokens=False)["input_ids"]
    if len(ci) > b:
        h = b//2; ci = ci[:h]+ci[-h:]; ctx = tok.decode(ci, skip_special_tokens=True)
    return ctx, q
def set_tmpl(task): tok.chat_template = None if task in RAW_TASKS else saved
def fin(ans, task): return ans if task in RAW_TASKS else ans.strip()

def run_variant(name, make_press):
    """Run one variant over all tasks; return per-task score and budget."""
    out = {}
    for task in args.tasks:
        set_tmpl(task); mn = TASK_MAX_NEW.get(task, 32)
        press = make_press(); press._budget_log = []
        sc = []
        for s in load_task(task, max_samples=args.n):
            ctx, q = split(s, task); ctx, q = truncate(ctx, q, args.max_seq)
            try:
                ans = pipe(ctx, question=q, press=press, max_new_tokens=mn)["answer"]
                a = s.get("answers", []); a = [a] if isinstance(a, str) else a
                sc.append(score_sample(fin(ans, task), a, task, s.get("all_classes")))
            except Exception as e:
                print(f"      error: {type(e).__name__} {str(e)[:40]}"); continue
        bud = float(np.mean(press._budget_log)) if press._budget_log else None
        out[task] = {"score": float(np.mean(sc)) if sc else 0.0, "budget": bud, "n": len(sc)}
        print(f"    {task:12s} score={out[task]['score']:.4f} budget={bud if bud else 0:.4f} (n={len(sc)})")
        torch.cuda.empty_cache()
    return out

results = {}
eps = args.eps

# Full first, to get the mean budget used as the fixed keep for WO_Adapt / WO_Layer
print(f"=== {mname} | ablation (eps={eps}, n={args.n}) ===\n[Full]")
results["Full"] = run_variant("Full", lambda: ProposedOnePassPress(epsilon=eps))
full_buds = [v["budget"] for v in results["Full"].values() if v["budget"]]
fixed = float(np.mean(full_buds)) if full_buds else 0.20
print(f"  -> Full mean budget {fixed:.4f} (used as fixed keep for WO_Adapt / WO_Layer)\n")

print("[WO_Head]")
results["WO_Head"] = run_variant("WO_Head", lambda: WOHeadPress(epsilon=eps))
print()
print(f"[WO_Adapt] (fixed keep={fixed:.3f})")
results["WO_Adapt"] = run_variant("WO_Adapt", lambda: WOAdaptPress(epsilon=eps, fixed_keep=fixed))
print()
print(f"[WO_Layer] (fixed keep={fixed:.3f})")
results["WO_Layer"] = run_variant("WO_Layer", lambda: WOLayerPress(epsilon=eps, fixed_keep=fixed))

# Summary
print(f"\n{'='*55}\nSummary (delta vs Full, 5-task mean)\n{'='*55}")
def tavg(d): return np.mean([v["score"] for v in d.values()])
full_avg = tavg(results["Full"])
print(f"{'variant':<20}{'mean':>10}{'vs Full':>12}")
print(f"{'Full':<20}{full_avg:>10.4f}{'-':>12}")
for key, label in [("WO_Head","WO_Head"),("WO_Adapt","WO_Adapt"),("WO_Layer","WO_Layer")]:
    a = tavg(results[key])
    print(f"{label:<20}{a:>10.4f}{a-full_avg:>+12.4f}")

os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir, f"{mname}_abl.json"), "w") as f:
    json.dump({"eps": eps, "fixed_keep": fixed, "results": results}, f, indent=2)
print(f"\nSaved: {args.outdir}/{mname}_abl.json")