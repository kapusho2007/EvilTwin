"""Build report/Milestone3_Report.docx from report/Milestone3_Report.md.
Style as in "Technical Report - Evil Twin": Calibri, black, bold headings; 1.15 line spacing and
0.9" margins (to keep the main body within the 15-page limit); real Word heading styles
(navigation pane / TOC); every figure and table captioned."""
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

import config as C

REPORT = C.ROOT / "report"
MD, OUT = REPORT / "Milestone3_Report.md", REPORT / "Milestone3_Report.docx"
FONT = "Calibri"
HEAD_PT = {1: 18, 2: 14, 3: 12}
TEXT_W = 6.7                      # inches between the 0.9" margins (Letter)


def runs(par, text, size=11, bold=False):
    for part in re.split(r"(\*\*.+?\*\*|`.+?`|\*.+?\*)", text.replace("\\|", "|")):
        if not part:
            continue
        r = par.add_run(part.strip("*`") if part[:1] in "*`" else part)
        r.bold = bold or part.startswith("**")
        r.italic = part.startswith("*") and not part.startswith("**")
        r.font.name, r.font.size = FONT, Pt(size)
        r.font.color.rgb = RGBColor(0, 0, 0)


def caption(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.0               # captions single-spaced, body 1.5
    runs(p, text.strip("*"), size=10)
    for r in p.runs:
        r.italic = True


def table(doc, rows):
    """Full text width; column widths follow the longest cell (capped), 8 pt for wide tables."""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style, t.alignment, t.autofit = "Table Grid", WD_TABLE_ALIGNMENT.CENTER, False
    size = 8 if ncol >= 5 else 9
    span = [i for i, r in enumerate(rows) if i and ncol > 1 and not any(r[1:])]   # section rows
    body = [r for i, r in enumerate(rows) if i not in span]
    longest = [max(max(len(r[j]) for r in body), 4) for j in range(ncol)]
    mar = OxmlElement("w:tblCellMar")                # narrow cell padding (0.03")
    for side in ("left", "right"):
        e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), "43"); e.set(qn("w:type"), "dxa"); mar.append(e)
    t._tbl.tblPr.append(mar)
    room = TEXT_W / (0.062 if size == 8 else 0.07) - 1.5 * ncol       # characters per line, minus padding
    cap = max(longest)
    while cap > 6 and sum(min(n, cap) for n in longest) > room:   # the longest columns wrap first
        cap -= 1
    need = [min(n, cap) + 1.5 for n in longest]
    widths = [TEXT_W * n / sum(need) for n in need]
    for j, col in enumerate(t.columns):             # grid widths (LibreOffice reads these)
        col.width = Inches(widths[j])
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            pf = c.paragraphs[0].paragraph_format
            pf.line_spacing, pf.space_after = 1.0, Pt(0)
            runs(c.paragraphs[0], cell, size=size, bold=(i == 0))
            if i == 0 or i in span:
                sh = OxmlElement("w:shd")
                sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "F2F2F2")
                c._tc.get_or_add_tcPr().append(sh)
    for i in span:
        t.cell(i, 0).merge(t.cell(i, ncol - 1))


def render(doc, lines):
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln or ln.startswith("<!--"):           # blank lines and Markdown comments
            i += 1; continue
        m = re.match(r"^(#{1,3})\s+(.*)", ln)
        img = re.match(r"^!\[(.*)\]\((.*)\)$", ln)
        if m:
            if m.group(2).startswith("Appendix A"):
                doc.add_page_break()                      # appendices start on a new page
            h = doc.add_heading(level=len(m.group(1)))
            runs(h, m.group(2), size=HEAD_PT[len(m.group(1))], bold=True)
        elif img:
            doc.add_picture(str(REPORT / img.group(2)), width=Inches(5.6))
            caption(doc, img.group(1))
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[-:\s|]+\|$", lines[i].strip()):
                    rows.append([c.strip() for c in re.split(r"(?<!\\)\|", lines[i].strip().strip("|"))])
                i += 1
            table(doc, rows)
            continue
        elif ln.startswith("*Table") or ln.startswith("*Figure"):
            caption(doc, ln)
        elif re.match(r"^\d+\.\s", ln):
            num, body = re.match(r"^(\d+\.)\s+(.*)", ln).groups()   # literal number: Word's List Number
            p = doc.add_paragraph()                                # would continue across the document
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.3), Inches(-0.3)
            runs(p, f"{num}\t{body}")
        elif ln.startswith(("- ", "* ")):
            runs(doc.add_paragraph(style="List Bullet"), ln[2:])
        else:
            runs(doc.add_paragraph(), ln)
        i += 1


def title_page(doc):
    def line(text, size, bold=False):
        p = doc.add_paragraph(); runs(p, text, size=size, bold=bold)
    line("Technical Report - Evil Twin (MITRE ATT&CK T1557.004)", 24, True)
    line("Milestone 3: Pipeline Implementation, Evaluation & Intelligent Ensemble Optimization", 14, True)
    for label, value in [
        ("Project Title", "Applied AI/ML Architectures for Cyber Intrusion Detection: Evil Twin / Rogue AP"),
        ("Group", "3"),
        ("Team Members", "Michael Alpin, Iftach Samel, Itay Rubanenko, Shaked Sarbstein "
                         "(student IDs and emails: TO PROVIDE)"),
        ("Dataset A", "AWID3 [5], Evil_Twin subset (802.11 L1+L2, WPA2)"),
        ("Dataset B", "RIDS (Saini et al. 2022 [4]), RogueAP class (802.11 L1+L2, WPA3)"),
        ("Code Repository & Branch", "TO PROVIDE (repository URL + branch)"),
        ("Model Artifact Archive", "TO PROVIDE (model-artifact archive link)"),
    ]:
        line(label, 12, True); line(value, 11)
    doc.add_page_break()


def main():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = FONT, Pt(11)
    # 1.15 spacing (standard for a page-limited technical report) keeps the main body within the
    # 15-page limit without cutting content; captions/tables stay single-spaced (set per element).
    st.paragraph_format.line_spacing = 1.15
    st.paragraph_format.space_after = Pt(4)
    for sec in doc.sections:                              # 0.9" margins (was 1.0") to fit the limit
        sec.left_margin = sec.right_margin = Inches(0.9)
        sec.top_margin = sec.bottom_margin = Inches(0.9)
    title_page(doc)
    render(doc, MD.read_text(encoding="utf-8").splitlines())
    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
