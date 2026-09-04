"""
Theory verification: TV(a, a_tilde) = delta, and delta vs attention entropy.

Entropy and delta are computed from the same window-averaged attention used by
the method (mean over the last `window` query positions), so the theoretical
quantities match what the compressor actually sees.

Checks:
  (1) TV == delta  (exact identity)
  (2) correlation between delta and head entropy
  (3) monotonicity of delta across compression ratios rho

Usage:
  CUDA_VISIBLE_DEVICES=0 python verify_delta.py --model <name> --n 15 --max_seq 2000
"""
import sys, os, argparse, json, math
import numpy as np
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.eval.longbench_loader import load_task
from src.eval.evaluator import DATASET2PROMPT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
ap.add_argument("--n", type=int, default=15)
ap.add_argument("--max_seq", type=int, default=2000)
ap.add_argument("--window", type=int, default=32, help="observation window (matches the method: 32)")
ap.add_argument("--rhos", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5])
ap.add_argument("--outdir", default="results/kvpress_theory")
args = ap.parse_args()

TASKS = ["qasper", "hotpotqa", "gov_report"]

print(f"Loading {args.model}  [GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}]")
tok = AutoTokenizer.from_pretrained(args.model)
if tok.bos_token is None: tok.bos_token = ""
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="eager")
model.eval()
print(f"Loaded (observation window = {args.window})\n")

def entropy(p, eps=1e-12):
    return -(p * (p + eps).log()).sum(-1)

per_rho = {r: {"H": [], "delta": [], "tv": [], "max_tv_gap": 0.0} for r in args.rhos}
W = args.window

def process(attn_full):
    """attn_full: (n_heads, T, T) full attention for one layer (eager).

    Average the last W query rows to get a per-key importance (n_heads, T),
    matching the method's attn.mean(dim=-2), then compute entropy, delta, TV.
    """
    nh, T, _ = attn_full.shape
    if T <= W:
        return
    A = attn_full[:, -W:, :].float().mean(dim=1)   # (n_heads, T)
    A = A / (A.sum(-1, keepdim=True) + 1e-12)       # normalize to a key distribution
    H = entropy(A)                                   # (n_heads,)
    for rho in args.rhos:
        k = max(1, int(T * rho))
        topv, topi = A.topk(k, dim=-1)
        m = topv.sum(-1)
        delta = 1.0 - m
        a_tilde = torch.zeros_like(A)
        a_tilde.scatter_(1, topi, topv / (m.unsqueeze(-1) + 1e-12))
        tv = 0.5 * (A - a_tilde).abs().sum(-1)
        d = per_rho[rho]
        d["H"].extend(H.cpu().tolist())
        d["delta"].extend(delta.cpu().tolist())
        d["tv"].extend(tv.cpu().tolist())
        d["max_tv_gap"] = max(d["max_tv_gap"], (tv - delta).abs().max().item())

def hook(module, inp, out):
    if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
        process(out[1][0].detach())  # (n_heads, T, T)
hooks = [layer.self_attn.register_forward_hook(hook) for layer in model.model.layers]

def build_input(task, s):
    tmpl = DATASET2PROMPT[task]; pre, post = tmpl.split("{context}", 1)
    text = pre + s["context"] + post.replace("{input}", s.get("input",""))
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.shape[1] > args.max_seq:
        h = args.max_seq // 2
        ids = torch.cat([ids[:, :h], ids[:, -h:]], dim=1)
    return ids.to("cuda")

mname = args.model.split("/")[-1]
print(f"=== {mname} | TV=delta and delta-vs-entropy (window={W}, n={args.n}/task) ===")
for task in TASKS:
    for s in load_task(task, max_samples=args.n):
        ids = build_input(task, s)
        with torch.no_grad():
            model(ids, output_attentions=True, use_cache=False)
        torch.cuda.empty_cache()
    print(f"  [{task}] done")
for h in hooks: h.remove()

print(f"\n{'rho':>5}{'mean_d':>9}{'mean_TV':>9}{'|TV-d|max':>12}{'corr(d,H)':>12}{'std_d':>10}")
print("-"*60)
results = {}
for rho in args.rhos:
    d = per_rho[rho]
    H = np.array(d["H"]); delta = np.array(d["delta"]); tv = np.array(d["tv"])
    if len(H) > 2 and H.std() > 1e-9 and delta.std() > 1e-9:
        r = float(np.corrcoef(H, delta)[0, 1])
    else:
        r = float("nan")
    results[rho] = {"delta_mean": float(delta.mean()), "tv_mean": float(tv.mean()),
                    "tv_delta_gap": d["max_tv_gap"], "pearson_delta_H": r,
                    "delta_std": float(delta.std()), "n": len(H)}
    print(f"{rho:>5}{delta.mean():>9.4f}{tv.mean():>9.4f}{d['max_tv_gap']:>12.2e}{r:>12.4f}{delta.std():>10.4f}")

os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir, f"{mname}_delta.json"),"w") as f:
    json.dump({"window": W, "results": results}, f, indent=2)
print(f"\nSaved: {args.outdir}/{mname}_delta.json")
print(f"Note: window={W} attention matches the method, so entropy is consistent.")
print(" |TV-delta| ~ 0 confirms the identity; positive corr(d,H); delta grows as rho shrinks.")