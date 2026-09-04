"""
Latency and memory of the final method, by context length.

Measures per-length latency and peak memory for FullKV, SnapKV, and the final
method (EntropyBudgetPress). Generation length is fixed (min = max) so the
latency reflects prefill + fixed decode only, not variable generation length.

Usage:
  CUDA_VISIBLE_DEVICES=0 python measure_efficiency_m1.py --model <m> --epsilon 0.7 --lengths 4000 8000 12000
Output: results/kvpress_efficiency_m1/<model>.json
"""
import sys, os, argparse, json, time
import numpy as np
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import transformers
transformers.logging.set_verbosity_error()
import logging
logging.getLogger("kvpress").setLevel(logging.ERROR)
from transformers import pipeline
from kvpress import SnapKVPress, KnormPress
from src.methods.proposed_kvpress import EntropyBudgetPress
from src.eval.longbench_loader import load_task
from src.eval.evaluator import DATASET2PROMPT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
ap.add_argument("--epsilon", type=float, default=0.7)
ap.add_argument("--lengths", type=int, nargs="+", default=[4000, 8000, 12000])
ap.add_argument("--gen", type=int, default=64)
ap.add_argument("--repeat", type=int, default=5)
ap.add_argument("--outdir", default="results/kvpress_efficiency_m1")
args = ap.parse_args()

print(f"Loading {args.model}  [GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}]")
pipe = pipeline("kv-press-text-generation", model=args.model, device="cuda",
                torch_dtype=torch.bfloat16, model_kwargs={"attn_implementation": "flash_attention_2"})
tok = pipe.tokenizer
if tok.bos_token is None: tok.bos_token = ""
torch.cuda.synchronize()
weight_mem = torch.cuda.memory_allocated() / 1e9
print(f"Loaded (weights {weight_mem:.2f} GB), fixed generation length = {args.gen}\n")

raw = load_task("hotpotqa", max_samples=10)
big_context = " ".join(s["context"] for s in raw)
tmpl = DATASET2PROMPT["hotpotqa"]; pre, post = tmpl.split("{context}", 1)
question = post.replace("{input}", raw[0].get("input",""))
def make_input(length):
    ids = tok(big_context, add_special_tokens=False)["input_ids"][:length]
    return pre + tok.decode(ids, skip_special_tokens=True)

def time_mem(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time(); fn(); torch.cuda.synchronize()
    return time.time() - t0, torch.cuda.max_memory_allocated() / 1e9

GEN_KW = dict(max_new_tokens=args.gen, min_new_tokens=args.gen)
def run_fullkv(ctx): return pipe(ctx, question=question, press=KnormPress(0.0), **GEN_KW)
def run_snapkv(ctx): return pipe(ctx, question=question, press=SnapKVPress(compression_ratio=0.8), **GEN_KW)
def run_m1(ctx):     return pipe(ctx, question=question, press=EntropyBudgetPress(epsilon=args.epsilon), **GEN_KW)
METHODS = [("FullKV", run_fullkv), ("SnapKV_20", run_snapkv), ("Proposed_M1", run_m1)]

results = {}; mname = args.model.split("/")[-1]
print(f"=== {mname} efficiency (gen={args.gen}, {args.repeat} runs, eps={args.epsilon}) ===")
for length in args.lengths:
    ctx = make_input(length)
    actual = len(tok(ctx, add_special_tokens=False)["input_ids"])
    print(f"\n[context ~{length} (actual {actual})]")
    print(f"{'Method':<14}{'latency(s)':>12}{'peak GB':>10}{'KV+act GB':>12}")
    results[length] = {}
    for label, fn in METHODS:
        try: fn(ctx)  # warmup
        except Exception as e:
            print(f"{label:<14} failed: {str(e)[:40]}"); continue
        dts, peaks = [], []
        for _ in range(args.repeat):
            dt, peak = time_mem(lambda: fn(ctx)); dts.append(dt); peaks.append(peak)
        dt_m = np.mean(dts); peak_m = np.mean(peaks); kv_m = peak_m - weight_mem
        results[length][label] = {"latency_s": float(dt_m), "peak_gb": float(peak_m),
                                   "kv_act_gb": float(kv_m), "latency_std": float(np.std(dts))}
        print(f"{label:<14}{dt_m:>12.3f}{peak_m:>10.2f}{kv_m:>12.2f}")

os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir, f"{mname}.json"), "w") as f:
    json.dump({"weight_gb": weight_mem, "epsilon": args.epsilon, "gen_fixed": args.gen, "results": results}, f, indent=2)
print(f"\nSaved: {args.outdir}/{mname}.json")