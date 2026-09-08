"""PDF report generator for SatQuery AI analysis results.

Generates an executive-ready 1-2 page PDF summary using fpdf2.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

from fpdf import FPDF

logger = logging.getLogger(__name__)


def _sanitize(text: Any) -> str:
    """Sanitize text for standard PDF Latin-1 fonts."""
    if text is None:
        return ""
    s = str(text)
    # Replace common typographic unicode characters
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2026": "...",
        "\u2022": "*",
        "\u00a9": "(c)",
        "\u00ae": "(R)",
        "\u2122": "TM",
        "\u2714": "[OK]",
        "\u2713": "[OK]",
        "\u2718": "[X]",
        "\u26a0": "[!]",
        "\u25b6": ">",
        "\u2192": "->",
    }
    for orig, rep in replacements.items():
        s = s.replace(orig, rep)
    # Filter out anything else outside Latin-1
    return s.encode("latin-1", "replace").decode("latin-1")


class SatQueryPDF(FPDF):
    def header(self) -> None:
        logo_path = Path(__file__).parent / "assets" / "logo.png"
        if logo_path.exists():
            try:
                self.image(str(logo_path), x=10, y=8, w=16)
                self.set_x(28)
            except Exception as e:
                logger.debug("Could not load logo in PDF: %s", e)
                self.set_x(10)
        else:
            self.set_x(10)

        self.set_font("Helvetica", "B", 14)
        self.set_text_color(24, 43, 73)
        self.cell(0, 7, "SatQuery AI -- Remote Sensing Analysis Report", new_x="LMARGIN", new_y="NEXT")
        self.set_x(28 if logo_path.exists() else 10)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "ISRO / SAC -- SIH 2026 (PS 26167)", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(226, 232, 240)
        self.line(10, 24, 200, 24)
        self.ln(6)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        page_str = f"Page {self.page_no()}/{{nb}} | SatQuery AI Verification Harness"
        self.cell(0, 10, page_str, align="C")


def generate_pdf_report(
    result: dict[str, Any],
    query: str,
    images_meta: list[dict[str, Any]] | None = None,
) -> bytes:
    """Generate a clean 1-2 page PDF report from a pipeline execution result.

    Parameters
    ----------
    result:
        Pipeline result dictionary (answer, trace, confidence, overlay, etc.)
    query:
        Natural-language user query
    images_meta:
        Optional metadata dictionaries for the input images

    Returns
    -------
    bytes
        Raw PDF file content suitable for streaming or download buttons.
    """
    try:
        pdf = SatQueryPDF(orientation="P", unit="mm", format="A4")
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=18)

        # 1. Metadata / Timestamp bar
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, f"Generated: {now_str}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # 2. Query Section
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, "Query:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(0, 5, _sanitize(query or "N/A"))
        pdf.ln(3)

        # 3. Answer Section
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, "Analysis Answer:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(15, 23, 42)
        answer = result.get("answer", "No answer generated.")
        pdf.multi_cell(0, 5, _sanitize(answer))
        pdf.ln(3)

        # 4. Confidence & Verification Metrics
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, "Verification & Confidence Assessment:", new_x="LMARGIN", new_y="NEXT")

        conf_tag = result.get("confidence_tag") or result.get("confidence") or "UNKNOWN"
        raw_conf = result.get("raw_confidence")
        consensus = result.get("consensus_score")
        sem_cons = result.get("semantic_consistency")
        chg_dir = result.get("change_direction")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(51, 65, 85)
        conf_line = f"Confidence Tag: {_sanitize(conf_tag)}"
        if raw_conf is not None:
            conf_line += f"  (Raw: {float(raw_conf):.3f})"
        if consensus is not None:
            conf_line += f"  |  Consensus Score: {float(consensus):.1%}"
        if sem_cons is not None:
            conf_line += f"  |  Semantic Consistency: {_sanitize(sem_cons)}"
        pdf.cell(0, 5, conf_line, new_x="LMARGIN", new_y="NEXT")

        if chg_dir:
            pdf.cell(0, 5, f"Change Direction: {_sanitize(chg_dir)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # 5. Image Metadata Table
        if images_meta:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 6, f"Input Images ({len(images_meta)} loaded):", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            for idx, meta in enumerate(images_meta, 1):
                bands = meta.get("bands", meta.get("channels", "N/A"))
                shape = meta.get("shape", "N/A")
                crs = meta.get("crs", "N/A")
                date = meta.get("date", meta.get("timestamp", "N/A"))
                sensor = meta.get("sensor", meta.get("modality", "N/A"))
                line = f"Image #{idx}: Sensor={_sanitize(sensor)}, Bands={bands}, Shape={shape}, Date={_sanitize(date)}, CRS={_sanitize(crs)}"
                pdf.cell(0, 4, line, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

        # 6. Trace & Specialists Summary
        trace = result.get("trace", {})
        if trace:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 6, "Execution & Specialist Routing Trace:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)

            task = trace.get("task", "N/A")
            tools = trace.get("tools_invoked", [])
            tools_str = " -> ".join(str(t) for t in tools) if tools else "Direct specialist"
            exec_time = trace.get("execution_time_seconds")
            time_str = f"{exec_time:.2f}s" if exec_time is not None else "N/A"

            pdf.cell(0, 4, f"Task Identified: {_sanitize(task)}  |  Execution Duration: {time_str}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 4, f"Specialists / Tools: {_sanitize(tools_str)}", new_x="LMARGIN", new_y="NEXT")

            # Verified facts if present
            verified_facts = result.get("verified_facts") or trace.get("verified_facts") or []
            if verified_facts:
                pdf.ln(1)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(0, 5, f"Verified Facts ({len(verified_facts)} statements checked):", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 8)
                for vf in verified_facts[:8]:  # show up to 8
                    status = "[VERIFIED]" if vf.get("verified", True) else "[UNVERIFIED]"
                    stmt = vf.get("fact") or vf.get("statement") or str(vf)
                    pdf.multi_cell(0, 4, f"  {status} {_sanitize(stmt)}")

        # Output bytes
        out = pdf.output()
        return bytes(out)

    except Exception as exc:
        logger.warning("Failed to generate PDF report: %s", exc, exc_info=True)
        return b""
