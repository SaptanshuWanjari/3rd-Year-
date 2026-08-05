"""
docx_template.py — ML lab report builder for the ml-lab-report skill.

Fill CONFIG below, then run with:
    ~/.venvs/lab-tools/bin/python docx_template.py
"""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ---------------------------------------------------------------- CONFIG ---
CONFIG = {
    "out_path": "",                                  # .../Lab_N_Section_Roll_Name.docx
    "title_line": "# ML Lab 1",                      # "# ML Lab N"
    "name_line": "## Name: Saptanshu Wanjari",
    "class_line": "### Class: A-A2",                 # Section-Batch
    "roll_line": "### Roll No: 18",
    "info_table": [                                  # (label, value) pairs, page 1
        ("Lab No.", "1"),
        ("Lab Aim and Objective", "..."),
        ("Dataset or Resources Used", "..."),
        ("Date of Performance", "DD Month YYYY"),
    ],
    "theory": [                                      # (section_heading, body_text)
        ("1.1 What is Data Preprocessing?", "..."),
    ],
    "theory_bullets": ["...", "..."],                # bullets under 1.2
    "code_py_path": "",                              # Lab_....py from nbconvert --to script
    "github_urls": [],                               # notebook URL, .py URL
    "materials": ["Hardware: ...", "Software: ..."], # bullets under 3.1
    "procedure_steps": ["1. ...", "2. ..."],         # steps under 3.2
    "shots": [                                       # (caption, png_filename, short_desc)
        ("Output 1 – ...", "cell_1.png", "2-3 student-style sentences."),
    ],
    "viva": [("Q1. ...", "Solution: ...")],          # 4-5 (question, answer)
    "shots_dir": "shots",                            # folder with the screenshots
}
# --------------------------------------------------------------------------


doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)
for s in doc.sections:
    s.top_margin = Inches(0.9); s.bottom_margin = Inches(0.9)
    s.left_margin = Inches(1.0); s.right_margin = Inches(1.0)


def heading(text, size=14, space_before=14):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before); p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text); r.bold = True; r.font.size = Pt(size); r.font.color.rgb = RGBColor(0, 0, 0)


def sub(text, size=12):
    heading(text, size=size, space_before=8)


def para(text, size=11, bold=False, align=None, space_after=6):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    r = p.add_run(text); r.bold = bold; r.font.size = Pt(size)


def bullets(items, size=11):
    for it in items:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3); p.paragraph_format.space_after = Pt(4)
        r = p.add_run("•  " + it); r.font.size = Pt(size)


def add_code(code, size=8):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(4); pf.space_after = Pt(6); pf.left_indent = Inches(0.1)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "F2F2F2")
    pPr.append(shd)
    pbdr = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}"); el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:color"), "BFBFBF")
        pbdr.append(el)
    pPr.append(pbdr)
    lines = code.strip("\n").split("\n")
    for i, line in enumerate(lines):
        run = p.add_run(line); run.font.name = "Consolas"; run.font.size = Pt(size)
        rf = run._element.rPr.get_or_add_rFonts(); rf.set(qn("w:ascii"), "Consolas"); rf.set(qn("w:hAnsi"), "Consolas")
        if i < len(lines) - 1:
            run.add_break()


def add_pic(path, width_in=6.0):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width_in))


def add_page_field(paragraph):
    run = paragraph.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    run._r.append(f1); run._r.append(instr); run._r.append(f2)


# footer page number only (NO header)
fp = doc.sections[0].footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = fp.add_run("Page "); fr.font.size = Pt(9)
add_page_field(fp)

# title block
para(CONFIG["title_line"], size=17, bold=True, space_after=4)
para(CONFIG["name_line"], size=13, bold=True, space_after=2)
para(CONFIG["class_line"], size=12, bold=True, space_after=2)
para(CONFIG["roll_line"], size=12, bold=True, space_after=10)

tbl = doc.add_table(rows=len(CONFIG["info_table"]), cols=2)
tbl.style = "Table Grid"
for i, (k, v) in enumerate(CONFIG["info_table"]):
    c0, c1 = tbl.rows[i].cells
    c0.width = Inches(1.9); c1.width = Inches(4.6)
    r0 = c0.paragraphs[0].add_run(k); r0.bold = True
    r1 = c1.paragraphs[0].add_run(v)
    r0.font.size = Pt(10); r1.font.size = Pt(10)
doc.add_page_break()

# 1. Theory
heading("1. Theory")
for h, body in CONFIG["theory"]:
    sub(h); para(body)
if CONFIG.get("theory_bullets"):
    sub("1.2 Techniques Applied")
    for b in CONFIG["theory_bullets"]:
        bullets([b])
doc.add_page_break()

# 2. Code with GitHub URL
heading("2. Code with GitHub URL")
para("The complete code for this lab is written in a Jupyter Notebook and uploaded to GitHub:")
for url in CONFIG["github_urls"]:
    p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.2)
    r = p.add_run(url); r.font.name = "Consolas"; r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x00, 0x00, 0xCC); r.font.underline = True
para("The complete written code is given below:", bold=True)
add_code(open(CONFIG["code_py_path"]).read())
doc.add_page_break()

# 3. Procedures and Materials
heading("3. Procedures and Materials Used")
sub("3.1 Materials Used")
bullets(CONFIG["materials"])
sub("3.2 Procedure")
for s in CONFIG["procedure_steps"]:
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(3)
    p.add_run(s).font.size = Pt(11)
doc.add_page_break()

# 4. Output Screenshots
heading("4. Output Screenshots with Descriptions")
for caption, fname, desc in CONFIG["shots"]:
    sub(caption, size=11)
    add_pic(f'{CONFIG["shots_dir"]}/{fname}')
    para(desc, size=10.5)
doc.add_page_break()

# 5. Viva Questions
heading("5. Viva Questions and Solutions")
for q, a in CONFIG["viva"]:
    p = doc.add_paragraph(); r = p.add_run(q); r.bold = True; r.font.size = Pt(11)
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(10)
    r = p.add_run(a); r.font.size = Pt(11)

doc.save(CONFIG["out_path"])
print("saved:", CONFIG["out_path"])
