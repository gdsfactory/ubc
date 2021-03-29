SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = ubc
SOURCEDIR     = .
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile upload

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

upload:
	mkdir -p site
	rsync -rv build/html/ site/ubc

open:
	open build/html/index.html

see: html open


md:
	jupyter nbconvert --to markdown report.ipynb

pdf:
	pandoc --highlight=tango --filter pandoc-crossref --filter pandoc-citeproc --csl ieee.csl  --bibliography=library.bib JoaquinMatres.md -o JoaquinMatres.pdf

open-pdf:
	zathura report.pdf

pdfc:
	jupyter nbconvert --to pdf report.ipynb

bib:
	cp $(HOME)/wikis/library.bib .


execute:
	jupyter nbconvert --to notebook --execute my_notebook.ipynb


clean-latex:
	rm *.aux *.tex *.log *.odt *.blg *.out *.html *.json *.epub *.docx *.latex
