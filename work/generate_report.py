"""Generate the final PDF report for the SHL assessment recommender."""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "shl_assessment_recommender_report.pdf"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleAccent",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=colors.HexColor("#f4a641"),
            spaceAfter=12,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionAccent",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0f1a2d"),
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyAccent",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13,
            textColor=colors.HexColor("#10213b"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallAccent",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#4a5870"),
        )
    )
    return styles


def _header_footer(canvas, doc):
    canvas.saveState()
    width, height = LETTER
    canvas.setFillColor(colors.HexColor("#0b1424"))
    canvas.rect(0, height - 34, width, 34, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#f4a641"))
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(doc.leftMargin, height - 22, "SHL Assessment Recommender - Implementation Report")
    canvas.setFillColor(colors.HexColor("#6e7d99"))
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(width - doc.rightMargin, 18, f"Page {canvas.getPageNumber()}")
    canvas.setStrokeColor(colors.HexColor("#d7deea"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 40, width - doc.rightMargin, 40)
    canvas.restoreState()


def _diagram() -> Drawing:
    d = Drawing(500, 120)
    fill = colors.HexColor("#e9f1ff")
    stroke = colors.HexColor("#17304f")
    accent = colors.HexColor("#f4a641")
    accent2 = colors.HexColor("#4fbfb1")
    items = [
        (10, 40, 90, 42, "Catalog JSON", accent),
        (120, 40, 90, 42, "Normalize", accent2),
        (230, 40, 90, 42, "FAISS", accent),
        (340, 40, 150, 42, "Planner + /chat", accent2),
    ]
    for x, y, w, h, label, color in items:
      d.add(Rect(x, y, w, h, rx=10, ry=10, fillColor=color, strokeColor=stroke, strokeWidth=1))
      d.add(String(x + w / 2, y + h / 2 - 4, label, textAnchor="middle", fontName="Helvetica-Bold", fontSize=10, fillColor=colors.HexColor("#08111d")))
    for x1, x2 in [(100, 120), (210, 230), (320, 340)]:
      d.add(Line(x1, 61, x2, 61, strokeColor=stroke, strokeWidth=1.5))
      d.add(Line(x2 - 8, 65, x2, 61, strokeColor=stroke, strokeWidth=1.5))
      d.add(Line(x2 - 8, 57, x2, 61, strokeColor=stroke, strokeWidth=1.5))
    d.add(String(250, 14, "Stateless conversation history is re-sent on every request.", textAnchor="middle", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor("#4a5870")))
    return d


def build_report(output_path: Path = OUTPUT_PATH) -> Path:
    styles = _styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.72 * inch,
        title="SHL Assessment Recommender Report",
        author="Codex",
    )

    story = []
    story.append(Paragraph("SHL Assessment Recommender", styles["TitleAccent"]))
    story.append(Paragraph("Implementation report for the conversational SHL Individual Test Solutions assistant.", styles["BodyAccent"]))
    story.append(Spacer(1, 0.16 * inch))

    summary = Table(
        [
            ["Scope", "Stateless FastAPI service with grounded recommendations, refinement, comparison, and refusal behavior."],
            ["Catalog", "Normalizes the SHL JSON catalog endpoint and caches a processed dataset plus FAISS index."],
            ["UI", "A responsive single-page interface is served at the root route and calls /chat directly."],
            ["Deployment", "Dockerized with optional Gemini, Anthropic, or OpenRouter provider wiring through environment variables."],
        ],
        colWidths=[1.15 * inch, 5.85 * inch],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f6fb")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#10213b")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d7deea")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d7deea")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.2),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        ),
    )
    story.append(summary)
    story.append(Spacer(1, 0.16 * inch))
    story.append(Paragraph("Architecture", styles["SectionAccent"]))
    story.append(_diagram())
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            "The backend keeps conversation state out of the server. Each request includes the full message transcript, which the planner uses to decide whether to clarify, recommend, refine, compare, or refuse.",
            styles["BodyAccent"],
        )
    )

    story.append(Paragraph("Key decisions", styles["SectionAccent"]))
    bullets = ListFlowable(
        [
            ListItem(Paragraph("Hybrid retrieval uses local TF-IDF plus FAISS, with metadata boosts for role, seniority, duration, and test-family signals.", styles["BodyAccent"])),
            ListItem(Paragraph("The catalog source of truth is the SHL JSON endpoint. Records are cached as raw and normalized JSON to keep startup fast.", styles["BodyAccent"])),
            ListItem(Paragraph("Prompt templates live in app/prompts/ so the behavior can be tuned without hardcoding prompt text into the service.", styles["BodyAccent"])),
            ListItem(Paragraph("Recommendation output stays deterministic and schema-compliant even when an LLM is configured for reply polishing.", styles["BodyAccent"])),
        ],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
    )
    story.append(bullets)

    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Verification", styles["SectionAccent"]))
    verif_table = Table(
        [
            ["Unit + integration tests", "13 passing tests covering schema, recommendation, refinement, comparison, refusal, and retrieval smoke checks."],
            ["UI", "Responsive root page with quick prompts, local conversation history, and recommendations rendered as cards."],
            ["PDF report", "Generated with reportlab and verified via pdftoppm rendering."],
        ],
        colWidths=[1.45 * inch, 5.55 * inch],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f6fb")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d7deea")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d7deea")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.9),
                ("LEADING", (0, 0), (-1, -1), 11.6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        ),
    )
    story.append(verif_table)

    story.append(PageBreak())
    story.append(Paragraph("Deployment and operational notes", styles["SectionAccent"]))
    story.append(
        Paragraph(
            "The project ships with a Dockerfile, Compose file, Render and Railway configs, and a README that explains the Gemini, Anthropic, and OpenRouter environment variables. The same container can serve both the UI and the API on port 7860.",
            styles["BodyAccent"],
        )
    )
    story.append(Spacer(1, 0.10 * inch))
    story.append(
        Paragraph(
            "Gemini is optional but supported. If GEMINI_API_KEY is present and the provider mode is auto, the app prefers Gemini for final reply polishing. If no key is configured, the service falls back to a deterministic template path.",
            styles["BodyAccent"],
        )
    )
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph("Limitations", styles["SectionAccent"]))
    story.append(
        Paragraph(
            "The local embedding backend is intentionally lightweight and lexical-heavy. That makes the app fast and portable, but a dedicated semantic embedding model could improve recall on harder, more abstract traces.",
            styles["BodyAccent"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("Future improvements", styles["SectionAccent"]))
    future_bullets = ListFlowable(
        [
            ListItem(Paragraph("Swap in a hosted or local semantic embedding model once the runtime dependency set is stable.", styles["BodyAccent"])),
            ListItem(Paragraph("Add trace-driven prompt tuning and per-intent evaluation reports.", styles["BodyAccent"])),
            ListItem(Paragraph("Expand alias handling for acronyms and product-family shorthand in the catalog.", styles["BodyAccent"])),
        ],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
    )
    story.append(future_bullets)
    story.append(Spacer(1, 0.10 * inch))
    story.append(
        Paragraph(
            "This report was generated directly from the implementation state in the workspace and is intended to be concise enough for the assignment submission limit.",
            styles["SmallAccent"],
        )
    )

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output_path


if __name__ == "__main__":
    path = build_report()
    print(path)
