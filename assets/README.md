# Assets: blog post and paper

The write-ups of the study behind this code. Anyone can read them. Edits go
through pull requests from Synthpop members.

```
blog/                the blog post: full post, Part 1 and Part 2
                     (self-contained HTML pages, built from blog/src/)
paper/main.tex       paper source (two-column)
paper/refs.bib       bibliography
paper/figures/       figures used by main.tex
paper/main.pdf       built paper
```

## Blog

Open `blog/index.html`, `blog/part-1.html` or `blog/part-2.html` in a browser
to read. To edit, change the sources in `blog/src/` and rebuild with
`python assets/blog/build.py`; [`blog/README.md`](blog/README.md) explains
where everything lives, including the charts.

GitHub shows HTML files as source. To share a rendered version, publish the
pages (for example with GitHub Pages) or send them as attachments.

## Paper

Build from `paper/` with any TeX Live installation:

```sh
latexmk -pdf main.tex
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or use [Tectonic](https://tectonic-typesetting.github.io/), a single binary
that fetches packages on demand: `tectonic -X compile main.tex`.

Packages used (all in standard TeX Live): newtx, microtype, booktabs,
tabularx, longtable, graphicx, xcolor, enumitem, tikz, pgfplots, pifont,
natbib, hyperref, cleveref, caption.

Commit the rebuilt `main.pdf` with the source change. The repository rejects
files over 1 MB, and the PDF is close to that limit, so keep figures as
compact PNGs.
