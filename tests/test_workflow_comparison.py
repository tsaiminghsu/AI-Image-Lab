"""training/WORKFLOW_COMPARISON.md must keep covering every workflow template, and must still render.

The comparison is a hand-written reading of training/workflow_template*.json and comfyui_client.py,
so it rots the moment someone adds a template. Adding a template file fails this test until the
document names it; the build test makes sure the Markdown subset the PDF script understands is
still all the document uses.
"""

import glob
import os
import re

import build_workflow_pdf as builder


def _document():
    with open(builder.SOURCE_MD, encoding="utf-8") as f:
        return f.read()


def test_every_workflow_template_is_named_in_the_comparison():
    document = _document()
    missing = []
    for path in sorted(glob.glob(os.path.join(builder.HERE, "workflow_template*.json"))):
        name = os.path.basename(path)
        # the base template is spelled in full; the others by their suffix, e.g. `_txt2img_zimage`
        suffix = name[len("workflow_template") : -len(".json")]
        if (suffix or name) not in document:
            missing.append(name)
    assert not missing, f"WORKFLOW_COMPARISON.md does not mention: {missing}"


def test_comparison_renders_every_section_and_table():
    document = _document()
    page = builder.build_html(document)
    assert page.count("<h2>") == document.count("\n## ")
    separators = [line for line in document.splitlines() if re.fullmatch(r"\|[-| ]+\|", line)]
    assert page.count("<table>") == len(separators)
    # an unsupported construct would leak through as literal markup
    assert "**" not in page and "```" not in page
