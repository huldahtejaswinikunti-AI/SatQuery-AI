from __future__ import annotations
import json
from datetime import datetime
from typing import Any

class ReportGenerator:
    @staticmethod
    def generate_markdown(payload: dict[str, Any]) -> str:
        tr = payload.get("trace", {})
        task = tr.get("task", "Remote Sensing Analysis")
        conf_tag = tr.get("confidence_tag", "UNKNOWN")
        conf_score = tr.get("confidence", 0.0)
        q = payload.get("query", "")
        ans = payload.get("answer", "")
        tools = ", ".join(tr.get("tools_invoked", []))
        return f"""# SatQuery AI ? Remote Sensing Analysis Report
**Task:** `{task}` | **Confidence:** `{conf_tag}` ({conf_score*100:.1f}%)

## User Query
> "{q}"

## Grounded Answer
{ans}

## Auditable Tools
`{tools}`
""".strip()

    @staticmethod
    def generate_json(payload: dict[str, Any]) -> str:
        def _sanitize(o):
            import numpy as np
            if isinstance(o, np.ndarray): return f"<ndarray shape={o.shape}>"
            if isinstance(o, dict): return {k: _sanitize(v) for k, v in o.items()}
            if isinstance(o, list): return [_sanitize(x) for x in o]
            return o
        return json.dumps(_sanitize(payload), indent=2)
