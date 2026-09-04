import os
import json
from pathlib import Path
from typing import Optional

# LongBench data directory. Override with the LONGBENCH_DATA env var, or place
# the .jsonl files under <project_root>/data/longbench/data/.
DATA_DIR = Path(os.environ.get(
    "LONGBENCH_DATA",
    Path(__file__).resolve().parents[2] / "data" / "longbench" / "data",
))

TASKS = {
    "single_doc_qa":   ["qasper", "narrativeqa", "multifieldqa_en"],
    "multi_doc_qa":    ["hotpotqa", "2wikimqa", "musique"],
    "summarization":   ["gov_report", "qmsum", "multi_news"],
    "few_shot":        ["trec", "triviaqa", "samsum"],
    "synthetic":       ["passage_count", "passage_retrieval_en"],
    "code":            ["lcc", "repobench-p"],
}

ALL_TASKS = [t for tasks in TASKS.values() for t in tasks]


def load_task(
    task: str,
    max_samples: Optional[int] = None,
    max_tokens: int = 31500,
) -> list[dict]:
    """Load one task, dropping samples whose context exceeds max_tokens."""
    fpath = DATA_DIR / f"{task}.jsonl"
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} not found")

    samples = []
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line.strip())
            est_tokens = len(s["context"]) / 4  # rough token estimate (chars / 4)
            if est_tokens <= max_tokens:
                samples.append(s)

    if max_samples:
        samples = samples[:max_samples]

    return samples


def load_all_tasks(
    max_samples_per_task: int = 50,
    max_tokens: int = 31500,
) -> dict[str, list[dict]]:
    """Load all tasks."""
    result = {}
    for task in ALL_TASKS:
        try:
            samples = load_task(task, max_samples_per_task, max_tokens)
            result[task] = samples
            print(f"  {task:<25}: {len(samples)} loaded")
        except FileNotFoundError:
            print(f"  {task:<25}: file not found (skipped)")
    return result


if __name__ == "__main__":
    print("=== LongBench data load test ===")
    print("\nLoading all tasks (50 per task):")
    data = load_all_tasks(max_samples_per_task=50)
    total = sum(len(v) for v in data.values())
    print(f"\n{len(data)} tasks, {total} samples loaded")

    print("\nSingle-task sample (first qasper):")
    sample = data["qasper"][0]
    print(f"  context length: {len(sample['context']):,} chars")
    print(f"  question: {sample['input'][:80]}...")
    print(f"  answers: {sample['answers']}")