.PHONY: tables pdf clean

tables:
	python3 tables/make_tables.py

pdf: tables
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C
	rm -f tables/*.tex
