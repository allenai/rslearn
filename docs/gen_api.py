"""Generate API reference pages for the rslearn package.

This runs at build time via the mkdocs-gen-files plugin. It walks the
``rslearn`` package, emits one Markdown stub per module containing a
mkdocstrings ``:::`` directive, and writes a ``SUMMARY.md`` that
mkdocs-literate-nav turns into the navigation for the API Reference section.

griffe (used by mkdocstrings) parses the source statically, so the heavy
runtime dependencies are not required for this to work.
"""

from pathlib import Path

import mkdocs_gen_files

# Package to document, relative to the repository root.
PACKAGE = "rslearn"

# Vendored / third-party implementations that are copied into the tree. They
# generally lack rslearn-style docstrings, so we keep them out of the reference.
EXCLUDED_PREFIXES = (
    "rslearn.models.galileo",
    "rslearn.models.presto",
    "rslearn.models.detr",
    "rslearn.models.clay",
    "rslearn.models.croma",
    "rslearn.models.panopticon_data",
    "rslearn.models.olmoearth_pretrain",
)

# Skip any module whose final component matches one of these names.
EXCLUDED_STEMS = ("single_file_galileo", "single_file_presto")

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / PACKAGE

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    parts = tuple(module_path.parts)

    # Turn package __init__ modules into the package page itself.
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = module_path.parent / "index.md"
    elif parts[-1] == "__main__":
        continue
    else:
        doc_path = module_path.with_suffix(".md")

    if not parts:
        continue

    identifier = ".".join(parts)

    if identifier.startswith(EXCLUDED_PREFIXES):
        continue
    if parts[-1] in EXCLUDED_STEMS:
        continue

    # doc_path is like "rslearn/dataset/dataset.md"; drop the leading package
    # segment so pages live under reference/.
    nav_parts = parts[1:] if len(parts) > 1 else ("rslearn",)
    full_doc_path = Path("reference", *doc_path.parts[1:])

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{identifier}`\n\n::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))
    nav[nav_parts] = doc_path.parts[1:] and Path(*doc_path.parts[1:]).as_posix()

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
