LANDSLIDE ?= landslide

presentation.html: slides.rst
	$(LANDSLIDE) -i -q $<

upload: presentation.html ganchev-dredze.png
	rm -rf _out
	mkdir -p _out
	cp *.png _out
	cp $< _out/index.html
	echo "make $@" | git commit-tree $$(python hash-tree.py _out) | \
	    xargs git update-ref gh-pages
	git push -f origin $$(git rev-parse gh-pages):refs/heads/gh-pages
