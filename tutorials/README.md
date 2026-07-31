# BoolForge Tutorials

BoolForge tutorials are available in three forms:

- Source files in src/
- Interactive notebooks in tutorials_ipynb/
- Complete PDF manual in BoolForge_Tutorials.pdf

The tutorials are designed to be reproducible, maintainable, and easy to extend.

Rather than authoring Jupyter notebooks directly, all tutorials are written as Python scripts using the Jupytext percent format. This provides readable source files, clean version control history, and deterministic notebook generation.

---

## Repository structure

```text
src/
    tutorial00_preface.py
    tutorial01_Boolean_functions.py
    ...
    tutorial12_dynamics_decomposition.py

src_utils/
    abstract.md
    frontmatter.md
    backmatter.md
    citation_note.md
    references.bib

build_tutorials.sh
Makefile
```

The files in `src/` are the single source of truth.

---

## Tutorial source format

Each tutorial is written as a Python file using the Jupytext percent format:

```python
# %% [markdown]
# # Tutorial title
#
# Markdown content

# %%
import boolforge as bf
```

This approach provides:

* readable Git diffs,
* editor and IDE support,
* no hidden notebook state,
* deterministic notebook generation.

---

## Generating Jupyter notebooks

Tutorial notebooks are generated automatically from the Python sources:

```bash
make tutorials
```

or equivalently

```bash
jupytext --sync src/*.py
```

The generated notebooks are stored in:

```text
tutorials_ipynb/
```

These notebooks are intended for interactive use by readers of the tutorials.

---

## Generating HTML tutorials

Similarly, HTML notebooks can be generated automatically from the Python sources:

```bash
make html
```

---

## Building the tutorial PDF

A complete tutorial document can be generated with:

```bash
make pdf
```

This:

1. Generates notebooks from the Python sources.
2. Converts notebooks to Markdown.
3. Combines all tutorials into a single document.
4. Creates a LaTeX source file using Pandoc.
5. Compiles the document with LaTeX.

The resulting PDF is:

```text
BoolForge_Tutorials.pdf
```

A supplementary-material version can be generated with:

```bash
make supplement
```

which creates:

```text
BoolForge_Tutorials_Supplement.pdf
```

---

## Build artifacts

The following directories are generated automatically and should not be edited manually:

```text
tutorials_ipynb/
build_latex_standalone/
build_latex_supplement/
```

Only the files in `src/` and `src_utils/` should be modified directly.

---

## Why this workflow?

This workflow provides:

* **Reproducibility** — all tutorial outputs are generated from source.
* **Maintainability** — tutorials are edited as Python code rather than notebook JSON.
* **Transparency** — Git diffs remain readable.
* **Flexibility** — tutorials can be consumed as notebooks or as a complete PDF.
* **Reliability** — publication-quality PDF documentation is generated automatically.

The tutorials are therefore treated as executable scientific documentation rather than static examples.
