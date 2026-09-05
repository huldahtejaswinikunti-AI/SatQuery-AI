import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
from pathlib import Path
import numpy as np
from evaluation.eval_utils import compute_token_f1
from satquery.pipeline.executor import PipelineExecutor

def run_eval(split="held_out", output="evaluation/results/cdvqa_run.json"):
    print(f"Evaluating CDVQA on {split}...")
    ex = PipelineExecutor()
    t1 = np.full((64, 64, 3), 50, dtype=np.uint8)
    t2 = np.full((64, 64, 3), 150, dtype=np.uint8)
    res = ex.run([t1, t2], [{"filename": "t1.png"}, {"filename": "t2.png"}], "What changed?")
    f1 = compute_token_f1(res["answer"], "Built-up area expanded.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"split": split, "f1": f1, "prediction": res["answer"]}, f, indent=2)
    print(f"CDVQA Eval Done. Token F1: {f1:.3f}")

if __name__ == "__main__":
    run_eval()
