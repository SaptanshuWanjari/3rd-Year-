---
name: ml-lab-report
description: Use when preparing or updating a Machine Learning lab submission in the 3rd-Year repo (folders like ML/lab/8-7-Lab-1): build/execute a preprocessing notebook for a dataset, generate the Word report (.docx) with title block, code+GitHub URL, short student-style output descriptions, viva Q&A, and push code to GitHub. Trigger words: "lab report", "lab docx", "make lab", "submit lab", "preprocessing lab".
---

# ML Lab Report

Build a college ML lab submission: executed notebook + `.py` script + screenshots + `.docx` report, pushed to GitHub. Formatting and voice rules below are mandatory.

## Inputs

**Ask the user (cannot be picked from files):** use the `question` tool.
- Lab number and title (e.g. Lab 1, "Data Preprocessing").
- Working folder under `ML/lab/*-Lab-N/` (confirm even if one folder looks obvious).
- Student details: Name, Section, Roll No, Batch (Class line = `Section-Batch`, e.g. `A-A2`).
- Date of performance.
- Push to GitHub? (yes/no)
- Anything extra: how many screenshots, viva count (default 4–5), extra sections.

**Derive from files (do NOT ask):**
- Dataset: the `.csv` inside the working folder.
- GitHub remote: `git remote -v` (origin URL).
- Python env: `<repo>/.venv/bin/python` (pandas/sklearn/jupyter) and `~/.venvs/lab-tools/bin/python` (python-docx, playwright, pillow).
- Chromium: newest `~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome`.
- Existing notebook in the folder (fix/extend it rather than recreating).

## Workflow

1. Read the folder + dataset. Confirm inputs. Delete stale drafts (`main.ipynb`, `main.py`) once the named notebook exists.
2. Build/extend the notebook `Lab_N_Section_Roll_Name.ipynb` covering: load+inspect, missing values, drop irrelevant columns, duplicates, outliers (IQR + box plots), label-encode categoricals, EDA plots, feature scaling, train/test split.
3. Execute it: `<repo>/.venv/bin/jupyter nbconvert --to notebook --execute --inplace "Lab_....ipynb"`.
4. Export the plain script: `<repo>/.venv/bin/jupyter nbconvert --to script "Lab_....ipynb"` → `Lab_....py`. This `.py` is the "written code" for the docx and GitHub.
5. Export HTML: `<repo>/.venv/bin/jupyter nbconvert --to html "Lab_....ipynb"`.
6. Screenshot each code cell output with playwright (light background only — verify corners are white). Script pattern below.
7. Fill `docx_template.py` (in this skill folder) CONFIG: info table, theory, procedures, shots (caption/image/desc), viva, github URLs. Run with `~/.venvs/lab-tools/bin/python docx_template.py`.
8. Commit + push code (notebook, `.py`, html, docx) to GitHub. Put notebook + script URLs in the docx (step 7 fills them). Commit message must be academic, not developer-style: `Lab N done`, `Lab N file done`, `Lab N assignment made` (no emojis, no "feat/fix", no bodies).
9. Verify (see Verification).

## Screenshot script pattern

```python
from playwright.sync_api import sync_playwright
import glob, os
chrome = glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome"))[-1]
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, executable_path=chrome, args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=2)
    pg.goto("file://" + os.path.abspath("Lab_....html")); pg.wait_for_timeout(800)
    for i, cell in enumerate(pg.query_selector_all("div.jp-Cell.jp-CodeCell")):
        out = cell.query_selector("div.jp-OutputArea")
        if out: out.screenshot(path=f"shots/cell_{i}.png")
    b.close()
```

## docx formatting rules (non-negotiable)

- Filename: `Lab_N_Section_Roll_Name.docx` (e.g. `Lab_1_A_18_Saptanshu Wanjari.docx`).
- **No header on any page.**
- Top of page 1, markdown-style heading lines (bold, decreasing size):
  ```
  # ML Lab 1
  ## Name: Saptanshu Wanjari
  ### Class: A-A2
  ### Roll No: 18
  ```
- First-page info table (Table Grid): `Lab No.`, `Lab Aim and Objective`, `Dataset or Resources Used`, `Date of Performance`.
- Footer on every page: centered/right page-number field (`Page ` + PAGE field), nothing else.
- Sections in order: `1. Theory`, `2. Code with GitHub URL`, `3. Procedures and Materials Used`, `4. Output Screenshots with Descriptions`, `5. Viva Questions and Solutions`.
- Code section: GitHub URLs first (notebook + `.py`), then ONE continuous monospace block of the full generated `.py`. **No per-cell labels** ("Code Cell 1" etc.).
- Screenshots: 8–12 outputs, each with a caption and a SHORT description.

## Student voice (descriptions)

Write like a 2nd-year engineering student, not a professional: short sentences, plain words, "I"/"we", "so", state what you did and what you observed. 2–3 sentences max (~40–60 words). No marketing words (robust, comprehensive, seamless, thereby). Example: "This is df.head() which shows the first 5 rows. Cabin column is mostly empty, so there are missing values. Some columns like Name and Ticket are not useful so we drop them later." Viva answers can be slightly fuller but still plain.

## Verification

- Open the docx with python-docx: header paragraph must be empty; footer must contain `PAGE` field; exactly the expected number of `word/media/` images; no paragraph starts with "Code Cell".
- `git status` clean after push; confirm files on GitHub via `gh api`.
- Describe every screenshot by checking the notebook cell's actual output (don't invent numbers); re-check key numbers (missing counts, shapes, split sizes) against cell outputs.
- `ponytail:chromium` cache path may drift — re-glob when the launch fails.

## Common mistakes

- AI-sounding long descriptions → rewrite in student voice (2–3 sentences).
- Black/dark screenshots → html theme is light by default; still verify corner pixels are (255,255,255).
- Running header leaking in → do not touch `section.header`; only set `section.footer`.
- Inline PAGE field broken → append fldChar begin / instrText `PAGE` / fldChar end inside one run's `rPr`.
- Forgetting the `.py` → docx code block must come from `nbconvert --to script`, not hand-copied cells.
