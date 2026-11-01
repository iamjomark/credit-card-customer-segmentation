"""Tiny helper to build notebooks where every code cell has a 3-line explainer above it."""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def explain(what: str, look_for: str, meaning: str) -> str:
    """Build the three-line markdown block that goes above each code cell."""
    return (
        f"**What this cell does:** {what}  \n"
        f"**What to look for:** {look_for}  \n"
        f"**What the output means:** {meaning}"
    )


def step(cells: list, what: str, look_for: str, meaning: str, code: str) -> None:
    """Add an explainer markdown cell and the code cell right below it."""
    cells.append(new_markdown_cell(explain(what, look_for, meaning)))
    cells.append(new_code_cell(code.strip("\n")))


def text(cells: list, markdown: str) -> None:
    """Add a plain markdown cell (headings, findings, notes)."""
    cells.append(new_markdown_cell(markdown.strip("\n")))


def save(cells: list, path: Path) -> None:
    """Write the notebook to disk with a standard python3 kernel."""
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
