"""Generate a concise submission PDF with form answers and implementation notes."""

from __future__ import annotations

from pathlib import Path

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
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "shl_submission_package.pdf"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleAccent",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=colors.HexColor("#f4a641"),
            alignment=TA_LEFT,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionAccent",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=16,
            textColor=colors.HexColor("#0f1a2d"),
            spaceBefore=8,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyAccent",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#10213b"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallAccent",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.3,
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
    canvas.drawString(doc.leftMargin, height - 22, "SHL AI Hiring - Submission Package")
    canvas.setFillColor(colors.HexColor("#6e7d99"))
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(width - doc.rightMargin, 18, f"Page {canvas.getPageNumber()}")
    canvas.setStrokeColor(colors.HexColor("#d7deea"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 40, width - doc.rightMargin, 40)
    canvas.restoreState()


def _answer_table(styles):
    data = [
        [
            Paragraph("<b>Did the solution meet the expectations?</b>", styles["BodyAccent"]),
            Paragraph(
                "Yes. The app asks clarifying questions, returns SHL recommendations, refines the shortlist, compares assessments, and includes evaluation checks.",
                styles["BodyAccent"],
            ),
        ],
        [
            Paragraph("<b>Public base URL</b>", styles["BodyAccent"]),
            Paragraph(
                "Not deployed permanently from this workspace. Use Railway or Render to create a stable URL before final submission.",
                styles["BodyAccent"],
            ),
        ],
        [
            Paragraph("<b>Cold start delay</b>", styles["BodyAccent"]),
            Paragraph(
                "N/A until deployed. On free tiers that sleep, expect a short first-request delay. Use an always-on plan to avoid it.",
                styles["BodyAccent"],
            ),
        ],
        [
            Paragraph("<b>LLM used</b>", styles["BodyAccent"]),
            Paragraph("Gemini 2.5 Flash via the Gemini API.", styles["BodyAccent"]),
        ],
        [
            Paragraph("<b>AI tools used</b>", styles["BodyAccent"]),
            Paragraph(
                "OpenAI Codex for implementation, debugging, and testing; Gemini API for response polishing.",
                styles["BodyAccent"],
            ),
        ],
    ]
    table = Table(
        data,
        colWidths=[1.9 * inch, 5.15 * inch],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f6fb")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#10213b")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d7deea")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d7deea")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("LEADING", (0, 0), (-1, -1), 11.4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        ),
    )
    return table


def build_pdf(output_path: Path = OUTPUT_PATH) -> Path:
    styles = _styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.72 * inch,
        title="SHL AI Hiring Submission Package",
        author="Codex",
    )

    story = []
    story.append(Paragraph("SHL AI Hiring Submission Package", styles["TitleAccent"]))
    story.append(
        Paragraph(
            "A compact, submission-ready summary for the assessment portal. It answers the form prompts and summarizes the implementation and verification work.",
            styles["BodyAccent"],
        )
    )
    story.append(Spacer(1, 0.14 * inch))
    story.append(Paragraph("Form answers", styles["SectionAccent"]))
    story.append(_answer_table(styles))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Submission note", styles["SectionAccent"]))
    story.append(
        Paragraph(
            "The public base URL should be filled in separately after deployment. This PDF intentionally avoids hosting instructions.",
            styles["SmallAccent"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Approach summary", styles["SectionAccent"]))
    approach_bullets = ListFlowable(
        [
            ListItem(Paragraph("Built a stateless FastAPI service. Each /chat request includes the transcript so the backend does not rely on session memory.", styles["BodyAccent"])),
            ListItem(Paragraph("Used the SHL catalog JSON endpoint as the source of truth, cached normalized records, and indexed the catalog for grounded retrieval.", styles["BodyAccent"])),
            ListItem(Paragraph("The planner classifies requests into clarify, recommend, compare, or refuse, then applies catalog-driven filters and comparison logic.", styles["BodyAccent"])),
            ListItem(Paragraph("Prompt templates keep the assistant concise and recruiter-like. Gemini is only used to polish the final reply, not to invent recommendations.", styles["BodyAccent"])),
            ListItem(Paragraph("Evaluation uses pytest scenarios plus a smoke script covering clarification, recommendation, refinement, comparison, and guardrail behavior.", styles["BodyAccent"])),
            ListItem(Paragraph("What did not work initially: Gemini sometimes returned JSON wrapped in code fences. I normalized the reply text so the UI receives clean recruiter language.", styles["BodyAccent"])),
            ListItem(Paragraph("Measured improvement by running the full test suite, smoke checks, and rendering the PDF output to verify layout quality.", styles["BodyAccent"])),
        ],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
    )
    story.append(approach_bullets)
    story.append(Spacer(1, 0.11 * inch))
    story.append(Paragraph("Evaluation notes", styles["SectionAccent"]))
    evaluation_bullets = ListFlowable(
        [
            ListItem(Paragraph("The regression suite passed end to end after the Gemini reply normalization fix.", styles["BodyAccent"])),
            ListItem(Paragraph("A smoke run confirmed clarification, recommendation, refinement, comparison, and guardrail behavior.", styles["BodyAccent"])),
            ListItem(Paragraph("The PDF output was rendered to images and visually checked for layout issues.", styles["BodyAccent"])),
            ListItem(Paragraph("The UI serves the chat experience from the root route and stays stateless across turns.", styles["BodyAccent"])),
            ListItem(Paragraph("The final implementation remains grounded in the catalog JSON and does not invent assessments outside the source data.", styles["BodyAccent"])),
        ],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
    )
    story.append(evaluation_bullets)
    story.append(Spacer(1, 0.11 * inch))
    story.append(
        Paragraph(
            "If you need the deployment steps separately, they can be provided outside this PDF so the submission file stays focused on the solution summary.",
            styles["SmallAccent"],
        )
    )

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output_path


if __name__ == "__main__":
    print(build_pdf())
