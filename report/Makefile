md:
	jupyter nbconvert --to markdown report.ipynb

pdf:
	pandoc --filter pandoc-citeproc --csl ieee.csl report.md -o JoaquinMatres.pdf

bib:
	cp $(HOME)/wikis/library.bib .


clean:
	rm *.aux *.tex *.log *.odt *.blg *.out *.html *.json *.epub *.docx *.latex


.PHONY: all clean paper
