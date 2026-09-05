import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
from pathlib import Path
import numpy as np
from evaluation.eval_utils import compute_bleu_1
from satquery.pipeline.executor import PipelineExecutor

def run_eval(output="evaluation/results/caption_eval.json"):
    ex = PipelineExecutor()
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    res = ex.run([img], [{"filename": "eval.png"}], "Describe the scene")
    bleu = compute_bleu_1("A high resolution satellite scene.", res["answer"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"bleu_1": bleu, "caption": res["answer"]}, f, indent=2)
    print(f"Caption Eval Done. BLEU-1: {bleu:.3f}")

if __name__ == "__main__":
    run_eval()
