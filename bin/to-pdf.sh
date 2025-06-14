export PANDOC_PATH="/C/Users/Khor Kean Teng/Downloads/pandoc-3.7.0.2-windows-x86_64/pandoc-3.7.0.2"
export PATH="$PATH:$PANDOC_PATH"
pandoc evaluation/explainability.html \
  --pdf-engine=xelatex \
  -o evaluation/explainability.pdf