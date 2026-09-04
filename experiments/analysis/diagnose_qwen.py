"""
Head entropy distribution across models (supplementary analysis).

Compares the per-head attention entropy of each model to characterize why
Qwen3 behaves differently. Reports the overall entropy mean/std, the mean
within-layer spread across heads, and the across-layer spread. Entropy is
computed from the same window-averaged attention used by the method.

Usage:
  CUDA_VISIBLE_DEVICES=0 python diagnose_qwen.py --model <m> --n 15 --max_seq 1500
Output: results/kvpress_diag/<model>_headent.json
"""
import sys, os, argparse, json
import numpy as np, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import transformers; transformers.logging.set_verbosity_error()
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.eval.longbench_loader import load_task
from src.eval.evaluator import DATASET2PROMPT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--n", type=int, default=15)
ap.add_argument("--max_seq", type=int, default=1500)
ap.add_argument("--window", type=int, default=32)
ap.add_argument("--outdir", default="results/kvpress_diag")
args = ap.parse_args()
TASKS = ["qasper", "hotpotqa", "gov_report"]; W = args.window

print(f"Loading {args.model}")
tok = AutoTokenizer.from_pretrained(args.model)
if tok.bos_token is None: tok.bos_token = ""
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager").eval()
nL = model.config.num_hidden_layers
print(f"Loaded ({nL} layers)\n")

def entropy(p, eps=1e-12): return -(p * (p + eps).log()).sum(-1)

# Per-layer list of head-entropy arrays (one per sample)
layer_head_ent = {i: [] for i in range(nL)}

def hook_factory(idx):
    def hook(m, i, o):
        if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
            attn = o[1][0].float()  # (H, T, T)
            T = attn.shape[1]
            if T <= W: return
            A = attn[:, -W:, :].mean(dim=1)  # (H, T), window-averaged (matches the method)
            A = A / (A.sum(-1, keepdim=True) + 1e-12)
            H = entropy(A)  # (H,) per-head entropy
            layer_head_ent[idx].append(H.cpu().numpy())
    return hook
hooks = [model.model.layers[i].self_attn.register_forward_hook(hook_factory(i)) for i in range(nL)]

def build(task, s):
    tmpl = DATASET2PROMPT[task]; pre, post = tmpl.split("{context}", 1)
    text = pre + s["context"] + post.replace("{input}", s.get("input",""))
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.shape[1] > args.max_seq:
        h = args.max_seq // 2; ids = torch.cat([ids[:, :h], ids[:, -h:]], dim=1)
    return ids.to("cuda")

mname = args.model.split("/")[-1]
for task in TASKS:
    for s in load_task(task, max_samples=args.n):
        with torch.no_grad(): model(build(task, s), output_attentions=True, use_cache=False)
        torch.cuda.empty_cache()
    print(f"  [{task}] done")
for h in hooks: h.remove()

# Aggregate: overall entropy, within-layer head spread, across-layer spread
all_ent = []; layer_within_std = []
for i in range(nL):
    if not layer_head_ent[i]: continue
    arr = np.concatenate(layer_head_ent[i])
    all_ent.append(arr)
    per_sample_std = [np.std(x) for x in layer_head_ent[i]]
    layer_within_std.append(np.mean(per_sample_std))
allE = np.concatenate(all_ent)

print(f"\n{'='*55}\n{mname} head-entropy distribution\n{'='*55}")
print(f"overall entropy: mean {allE.mean():.3f}, std {allE.std():.3f}, range {allE.min():.3f}~{allE.max():.3f}")
print(f"within-layer head std (mean): {np.mean(layer_within_std):.3f}")
print(f"across-layer mean-entropy std: {np.std([a.mean() for a in all_ent]):.3f}")

os.makedirs(args.outdir, exist_ok=True)
json.dump({"model": mname, "overall_mean": float(allE.mean()), "overall_std": float(allE.std()),
           "within_layer_head_std": float(np.mean(layer_within_std)),
           "across_layer_std": float(np.std([a.mean() for a in all_ent])),
           "n_layers": nL},
          open(f"{args.outdir}/{mname}_headent.json", "w"), indent=2)
print(f"\nSaved: {args.outdir}/{mname}_headent.json")