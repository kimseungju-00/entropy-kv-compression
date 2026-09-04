"""
Layer-wise attention entropy heterogeneity.

Measures the mean attention entropy of each layer and the max/min ratio across
layers. Entropy is computed from the same window-averaged attention used by the
method, so it is consistent with the compressor.

Usage:
  CUDA_VISIBLE_DEVICES=0 python measure_layer_entropy.py --model <name> --n 20 --max_seq 2000
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
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--max_seq", type=int, default=2000)
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
n_layers = model.config.num_hidden_layers
print(f"Loaded ({n_layers} layers)\n")

def entropy(p, eps=1e-12):
    return -(p * (p + eps).log()).sum(-1)

# Accumulate per-layer entropy over all heads and samples
layer_ent_sum = np.zeros(n_layers)
layer_ent_cnt = np.zeros(n_layers)

W = 32  # observation window (matches the method)
def hook_factory(idx):
    def hook(module, inp, out):
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            attn = out[1][0].float()  # (n_heads, T, T)
            T = attn.shape[1]
            if T <= W:
                return
            # Average the last W query rows -> (n_heads, T), same as attn.mean(dim=-2)
            A = attn[:, -W:, :].mean(dim=1)
            A = A / (A.sum(-1, keepdim=True) + 1e-12)
            H = entropy(A)  # (n_heads,)
            layer_ent_sum[idx] += H.sum().item()
            layer_ent_cnt[idx] += H.numel()
    return hook

hooks = [model.model.layers[i].self_attn.register_forward_hook(hook_factory(i))
         for i in range(n_layers)]

def build_input(task, s):
    tmpl = DATASET2PROMPT[task]; pre, post = tmpl.split("{context}", 1)
    text = pre + s["context"] + post.replace("{input}", s.get("input",""))
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.shape[1] > args.max_seq:
        h = args.max_seq // 2
        ids = torch.cat([ids[:, :h], ids[:, -h:]], dim=1)
    return ids.to("cuda")

mname = args.model.split("/")[-1]
print(f"=== {mname} | layer entropy heterogeneity (n={args.n}/task) ===")
for task in TASKS:
    for s in load_task(task, max_samples=args.n):
        ids = build_input(task, s)
        with torch.no_grad():
            model(ids, output_attentions=True, use_cache=False)
        torch.cuda.empty_cache()
    print(f"  [{task}] done")
for h in hooks: h.remove()

layer_ent = layer_ent_sum / np.maximum(layer_ent_cnt, 1)
emin, emax = layer_ent.min(), layer_ent.max()
imin, imax = int(layer_ent.argmin()), int(layer_ent.argmax())
ratio = emax / emin
mean_ent = layer_ent.mean()

print(f"\nmin entropy: {emin:.3f} (L{imin})")
print(f"max entropy: {emax:.3f} (L{imax})")
print(f"max/min ratio: {ratio:.2f}x")
print(f"overall mean: {mean_ent:.3f}")
print(f"\nper-layer entropy:")
for i in range(n_layers):
    bar = "#" * int(layer_ent[i]/emax*30)
    print(f"  L{i:2d}: {layer_ent[i]:.3f} {bar}")

os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir, f"{mname}_layerent.json"), "w") as f:
    json.dump({"layer_entropy": layer_ent.tolist(), "min": float(emin), "max": float(emax),
               "min_layer": imin, "max_layer": imax, "ratio": float(ratio),
               "mean": float(mean_ent), "n_layers": n_layers}, f, indent=2)
print(f"\nSaved: {args.outdir}/{mname}_layerent.json")
print("Note: a larger ratio means entropy (hence compression loss delta) varies more across layers.")