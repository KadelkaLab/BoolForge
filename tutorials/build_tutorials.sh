#!/usr/bin/env bash
set -euo pipefail


# Standalone PDF or supplement of journal article format?
MODE="${1:-standalone}"

# Keep build directory (for debugging or journal submission)?
KEEP_BUILD="${2:-false}"

case "$MODE" in
    standalone)
        PDF_NAME="BoolForge_Tutorials.pdf"
        TEX_NAME="BoolForge_Tutorials.tex"
        TITLE="Boolean Network Modeling in Systems Biology: A Hands-On Tutorial with BoolForge"
        BUILD_DIR="build_latex_standalone"
        ;;
    supplement)
        PDF_NAME="BoolForge_Tutorials_Supplement.pdf"
        TEX_NAME="BoolForge_Tutorials_Supplement.tex"
        TITLE="BoolForge Tutorials"
        BUILD_DIR="build_latex_supplement"
        ;;
    *)
        echo "Usage: $0 [standalone|supplement]"
        exit 1
        ;;
esac

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

UTILS_DIR="src_utils"
IPYNB_DIR="tutorials_ipynb"

MD_DIR="$BUILD_DIR/tutorials_md"
FIG_DIR="$BUILD_DIR/figures"

OUT_TEX="$BUILD_DIR/$TEX_NAME" 
HEADER_TEX="$UTILS_DIR/header.tex"

AUTHOR="Claus Kadelka"

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------

echo "==> Cleaning previous build artifacts"
rm -rf "$MD_DIR" "$FIG_DIR" "$BUILD_DIR"
mkdir -p "$MD_DIR" "$FIG_DIR" "$BUILD_DIR"

# ------------------------------------------------------------
# Convert notebooks → Markdown
# ------------------------------------------------------------

echo "==> Converting notebooks to Markdown"

for nb in "$IPYNB_DIR"/*.ipynb; do
  jupyter nbconvert \
    --to markdown \
    --TemplateExporter.exclude_input_prompt=True \
    --TemplateExporter.exclude_output_prompt=True \
    "$nb" \
    --output-dir "$MD_DIR"
done

# ------------------------------------------------------------
# Clean captions (portable sed)
# ------------------------------------------------------------

echo "==> Cleaning figure captions"

for f in "$MD_DIR"/*.md; do
    sed -i.bak 's/!\[png\]/![]/g' "$f"
    rm "$f.bak"
done


# ------------------------------------------------------------
# FIX 1: ensure headings start new lines
# ------------------------------------------------------------

echo "==> Fixing missing blank lines before headings"

for f in "$MD_DIR"/*.md; do
    sed -i.bak 's/^\(# .*\)/\n\1/g' "$f"
    rm "$f.bak"
done

# ------------------------------------------------------------
# FIX 2: remove accidental bold formatting
# ------------------------------------------------------------

echo "==> Removing accidental bold formatting"

for f in "$MD_DIR"/*.md; do
    sed -i.bak 's/\*\*/ /g' "$f"
    rm "$f.bak"
done

# ------------------------------------------------------------
# FIX 3: replace Unicode logical symbols 
# ------------------------------------------------------------

echo "==> Replacing Unicode logical symbols (∧, ∨, ¬)"

for f in "$MD_DIR"/*.md; do
    sed -i.bak 's/∧/\\wedge/g' "$f"
    sed -i.bak 's/∨/\\vee/g' "$f"
    sed -i.bak 's/¬/\\neg/g' "$f"
    
    rm "$f.bak"
done


# ------------------------------------------------------------
# Flatten figures → figures/
# ------------------------------------------------------------

echo "==> Flattening figure directories"

for md in "$MD_DIR"/*.md; do
    base=$(basename "$md" .md)
    files_dir="$MD_DIR/${base}_files"

    if [ -d "$files_dir" ]; then
        counter=0

        for img in "$files_dir"/*; do
            ext="${img##*.}"
            new_name="${base}_fig${counter}.${ext}"

            cp "$img" "$FIG_DIR/$new_name"

            sed -i.bak "s|${base}_files/$(basename "$img")|${FIG_DIR}/${new_name}|g" "$md"
            rm "$md.bak"

            ((counter++))
        done
    fi
done

rm -rf "$MD_DIR"/*_files

# ------------------------------------------------------------
# Build LaTeX
# ------------------------------------------------------------

echo "==> Building LaTeX"

files=()
for f in "$MD_DIR"/tutorial*.md; do
    if [[ "$f" != *tutorial00*.md && "$f" != *tutorial12*.md ]]; then
#    if [[ "$f" != *tutorial00*.md ]]; then
        files+=("$f")
    fi
done

{
    if [ "$MODE" = "standalone" ]; then
        cat "$UTILS_DIR"/abstract_standalone.md
        printf "\n\n"
        cat "$MD_DIR"/tutorial00_preface_standalone.md
        printf "\n\n"
        cat "$UTILS_DIR"/introduction_standalone.md
        printf "\n\n"
    else
        cat "$MD_DIR"/tutorial00_preface_supplement.md
        printf "\n\n"
    fi

    printf "\n\n"
    cat "$UTILS_DIR"/citation_note.md

    for f in "${files[@]}"; do
        cat "$f"
        printf "\n\n"
    done

    cat "$UTILS_DIR"/backmatter.md

} > "$MD_DIR/_combined.md"

pandoc \
    "$MD_DIR/_combined.md" \
    --resource-path="$MD_DIR" \
    --from markdown+yaml_metadata_block+tex_math_dollars+fenced_code_blocks \
    --toc \
    --standalone \
    --number-sections \
    --listings \
    --syntax-highlighting=none \
    --natbib --bibliography="$UTILS_DIR/references.bib" \
    -V title="$TITLE" \
    -V author="$AUTHOR" \
    -V date="$(date '+%B %d, %Y')" \
    -V geometry:margin=1in \
    -V documentclass=article \
    -H "$HEADER_TEX" \
    -o "$OUT_TEX"
    
echo "==> Converting verbatim to lstlisting (force syntax highlighting)"

sed -i.bak 's/\\begin{verbatim}/\\begin{lstlisting}/g' "$OUT_TEX"
sed -i.bak 's/\\end{verbatim}/\\end{lstlisting}/g' "$OUT_TEX"
rm "$OUT_TEX.bak"

echo "==> Compiling $OUT_TEX to generate PDF"

latexmk \
    -pdf \
    -aux-directory="$BUILD_DIR" \
    -output-directory="$BUILD_DIR" \
    "$OUT_TEX"

cp "$BUILD_DIR/${TEX_NAME%.tex}.pdf" .


echo "==> Cleaning up temporary files"

rm -rf "$MD_DIR"

if [ "$KEEP_BUILD" != "true" ]; then
    rm -rf "$BUILD_DIR" "$FIG_DIR"
fi

echo "==> Done"