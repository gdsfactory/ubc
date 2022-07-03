md:
	jupyter nbconvert --to markdown report.ipynb

pdf:
	pandoc --highlight=tango --filter pandoc-crossref --filter pandoc-citeproc --csl ieee.csl  --bibliography=library.bib JoaquinMatres.md -o JoaquinMatres.pdf

open:
	zathura report.pdf

pdfc:
	jupyter nbconvert --to pdf report.ipynb

bib:
	cp $(HOME)/wikis/library.bib .


execute:
	jupyter nbconvert --to notebook --execute my_notebook.ipynb


clean:
	rm *.aux *.tex *.log *.odt *.blg *.out *.html *.json *.epub *.docx *.latex


.PHONY: all clean paper
