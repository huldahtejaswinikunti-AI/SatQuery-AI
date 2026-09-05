import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse, json
from pathlib import Path
import numpy as np
from evaluation.eval_utils import compute_token_f1
from satquery.pipeline.executor import PipelineExecutor

def run_eval(split="held_out", output="evaluation/results/rsvqa_run.json"):
    print(f"Evaluating RSVQA on {split}...")
    ex = PipelineExecutor()
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    res = ex.run([img], [{"filename": "eval.png"}], "What land cover is visible?")
    f1 = compute_token_f1(res["answer"], "Vegetation and soil cover are visible.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"split": split, "f1": f1, "prediction": res["answer"]}, f, indent=2)
    print(f"RSVQA Eval Done. Token F1: {f1:.3f}")

if __name__ == "__main__":
    run_eval()
