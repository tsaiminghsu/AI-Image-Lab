"""Render training/WORKFLOW_COMPARISON.md as a styled HTML page, and optionally as a PDF.

    python training/build_workflow_pdf.py          # writes the HTML
    python training/build_workflow_pdf.py --pdf    # also prints it to PDF with headless Chrome

Output goes to outputs/reports/ (git-ignored): Workflow模板差異對照表_2026-10.pdf, with the HTML
next to the other report sources in outputs/reports/source/. The Markdown is the source of truth;
this script only supports what that file uses: ##/### headings, pipe tables, bullet and numbered
lists, paragraphs, `code` and **bold**. Standard library only - it needs neither the ComfyUI venv
nor a GPU.
"""

import argparse
import html
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SOURCE_MD = os.path.join(HERE, "WORKFLOW_COMPARISON.md")
REPORTS_DIR = os.path.join(REPO_ROOT, "outputs", "reports")
SOURCE_DIR = os.path.join(REPORTS_DIR, "source")
BASENAME = "Workflow模板差異對照表_2026-10"
TITLE = "Workflow 模板差異對照表（2026-10）"
CHROME_CANDIDATES = (
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
)

# Same look as the other reports in outputs/reports/source/ (A4, Noto Sans TC, blue header rows).
# The @page margin boxes need Chrome 131 or newer.
CSS = """
  @page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @top-right { content: "%(title)s"; font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif; font-size: 7.5pt; color: #9aa3ad; }
    @bottom-center { content: counter(page) " / " counter(pages); font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif; font-size: 8pt; color: #9aa3ad; }
  }
  @page :first { @top-right { content: none; } }
  :root { --ink: #1d2733; --muted: #5d6b7a; --accent: #1f4e79; --rule: #d6dde5; --zebra: #f5f7fa; }
  html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { font-family: "Noto Sans TC", "Microsoft JhengHei", "PingFang TC", sans-serif; font-size: 10.2pt;
         line-height: 1.7; color: var(--ink); margin: 0; }
  a { color: #1a5fb4; text-decoration: none; }
  .cover { padding: 6mm 0 4mm; border-bottom: 2.5pt solid var(--accent); margin-bottom: 6mm; }
  .kicker { font-size: 8.5pt; letter-spacing: 0.18em; color: var(--accent); font-weight: 700; }
  h1 { font-size: 23pt; line-height: 1.3; margin: 2mm 0 3mm; font-weight: 700; }
  .meta { font-size: 9pt; color: var(--muted); }
  .meta span + span::before { content: "·"; margin: 0 0.6em; color: #b8c1cb; }
  h2 { font-size: 14pt; color: var(--accent); margin: 8mm 0 2.5mm; padding-bottom: 1.2mm;
       border-bottom: 0.8pt solid var(--rule); break-after: avoid; }
  h2 .num { color: #9aa9b8; font-weight: 400; margin-right: 0.4em; }
  h3 { font-size: 10.8pt; margin: 4mm 0 1.5mm; break-after: avoid; }
  p { margin: 0 0 2.5mm; }
  p.note { font-size: 8.8pt; color: var(--muted); }
  ul, ol { margin: 0 0 3mm; padding-left: 5.5mm; }
  li { margin-bottom: 1.2mm; }
  li::marker { color: var(--accent); }
  table { width: 100%%; border-collapse: collapse; margin: 1mm 0 4mm; font-size: 8.6pt; line-height: 1.5; }
  thead { display: table-header-group; }
  th { background: var(--accent); color: #fff; text-align: left; font-weight: 700; padding: 1.6mm 2mm;
       border: 0.6pt solid var(--accent); }
  td { padding: 1.5mm 2mm; border: 0.6pt solid var(--rule); vertical-align: top; }
  tbody tr:nth-child(even) td { background: var(--zebra); }
  tr { break-inside: avoid; }
  code { font-family: Consolas, "Noto Sans TC", monospace; font-size: 0.92em; background: #eef1f5;
         padding: 0 0.8mm; border-radius: 0.6mm; }
  td code, th code { background: transparent; padding: 0; }
""" % {"title": TITLE}

LIST_ITEM = re.compile(r"^(- |\d+\. )")


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)


def split_row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def render_body(markdown):
    """Markdown text -> HTML fragment. h2 headings are numbered 01, 02, ... in document order."""
    lines = markdown.split("\n")
    out = []
    i = 0
    section = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            i += 1
        elif line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
            i += 1
        elif line.startswith("## "):
            section += 1
            heading = re.sub(r"^\d+\.\s*", "", line[3:])
            out.append(f'<h2><span class="num">{section:02d}</span>{inline(heading)}</h2>')
            i += 1
        elif line.startswith("|"):
            header = split_row(line)
            i += 2  # skip the |---| separator
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
        elif LIST_ITEM.match(line):
            tag = "ol" if line[0].isdigit() else "ul"
            items = []
            while i < len(lines) and (LIST_ITEM.match(lines[i]) or lines[i].startswith("   ")):
                if LIST_ITEM.match(lines[i]):
                    items.append(LIST_ITEM.sub("", lines[i], count=1))
                else:  # indented continuation of the previous item
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
        elif line.strip():
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "#", "- ")) \
                    and not re.match(r"^\d+\. ", lines[i]):
                paragraph.append(lines[i])
                i += 1
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
        else:
            i += 1
    return "\n".join(out)


def build_html(markdown):
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>{TITLE}</title>
<style>{CSS}</style>
</head>
<body>
<section class="cover">
  <div class="kicker">模板對照</div>
  <h1>{TITLE}</h1>
  <div class="meta"><span>資料來源：training/workflow_template*.json、training/comfyui_client.py</span><span>只讀程式與模板，未實際生成</span></div>
</section>
{render_body(markdown)}
</body>
</html>
"""


def find_chrome(explicit=None):
    for path in ((explicit,) if explicit else CHROME_CANDIDATES):
        if path and os.path.isfile(path):
            return path
    raise SystemExit("找不到 Chrome；用 --chrome 指定 chrome.exe 的路徑")


def print_pdf(html_path, pdf_path, chrome):
    # --headless plus a private --user-data-dir: without them chrome.exe hands the job to an
    # already-open Chrome window and exits without writing the file.
    profile = os.path.join(os.environ.get("TEMP", REPO_ROOT), "report-chrome")
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", "file:///" + html_path.replace("\\", "/")],
        check=True, capture_output=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pdf", action="store_true", help="also print the HTML to a PDF with headless Chrome")
    parser.add_argument("--chrome", help="path to chrome.exe (default: the usual install locations)")
    args = parser.parse_args(argv)

    with open(SOURCE_MD, encoding="utf-8") as f:
        page = build_html(f.read())
    os.makedirs(SOURCE_DIR, exist_ok=True)
    html_path = os.path.join(SOURCE_DIR, BASENAME + ".html")
    with open(html_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    print(html_path)
    if args.pdf:
        pdf_path = os.path.join(REPORTS_DIR, BASENAME + ".pdf")
        print_pdf(html_path, pdf_path, find_chrome(args.chrome))
        print(pdf_path)


if __name__ == "__main__":
    sys.exit(main())
