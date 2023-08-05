format:
	jupytext --pipe black *.ipynb

sync:
	jupytext --sync *.ipynb

convert:
	jupytext *.ipynb --to *.py

clean:
	nbstripout --drop-empty-cells *.ipynb
	nbstripout --drop-empty-cells **/*.ipynb
