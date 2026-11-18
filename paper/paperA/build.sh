#!/usr/bin/env bash
# Build paper/paperA/main.pdf in WSL (pdflatex from apt texlive). Compiles in a temp copy to keep the repo clean.
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
cp -r "$SRC"/. "$TMP"/
cd "$TMP"
pdflatex -interaction=nonstopmode main.tex > build1.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1 || true
grep -E "^!|undefined|Overfull" build.log | head -20 || true
grep "Output written" build.log
cp main.pdf "$SRC"/main.pdf
mkdir -p "$TMP"/pages && pdftoppm -r 80 -png main.pdf "$TMP"/pages/p >/dev/null 2>&1 && \
  mkdir -p "$SRC"/.preview && cp "$TMP"/pages/*.png "$SRC"/.preview/
rm -rf "$TMP"
