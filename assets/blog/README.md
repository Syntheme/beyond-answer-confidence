# Blog post

One post, built as three pages from the same sources:

| Page | What it is |
| --- | --- |
| `index.html` | the full post |
| `part-1.html` | Part 1 of 2: what Jev's confidence tells you |
| `part-2.html` | Part 2 of 2: asking Jev directly |

The two parts are for publishing in two steps; the full post is the same text
in one page. Each page is a single self-contained file: text, styles, chart
data and chart code are all inside it, and there are no image files. Open a
page in a browser to read it; publish it by uploading that one file.

Parts link to each other as `part-1.html` and `part-2.html`. If they are
published under other names or addresses, change those links in
`src/content/` (search for `part-1.html` and `part-2.html`) and rebuild.

## Editing

Never edit the three pages directly; edit `src/` and rebuild:

```sh
python assets/blog/build.py           # rewrite the pages (standard library only)
python assets/blog/build.py --check   # fail if a page is out of date
```

Commit the sources and the rebuilt pages together. The `blog-build`
pre-commit hook and `tests/unit/test_blog_build.py` fail if they disagree.

```
src/
  page.html              page skeleton: <head>, notes list, scripts
  style.css              all styles
  content/
    header-full.html     title block of the full post
    header-part-1.html   title block of Part 1
    header-part-2.html   title block of Part 2
    intro.html           opening and summary (full post and Part 1)
    intro-part-2.html    recap that opens Part 2
    sections-1.html      the Part 1 sections
    end-part-1.html      closing paragraph of Part 1
    sections-2.html      the Part 2 sections
    conclusion.html      "What we learned" (full post and Part 2)
    methods.html         methods, limitations, provenance, disclosures
    notes.html           every footnote
  charts/
    data-1.json …        the numbers each chart draws
    charts-1.js …        the code that draws the charts
```

Which fragments make up which page is set in `PAGES` at the top of
`build.py`. Text shared by two pages (intro, sections, methods) lives in one
file, so an edit there changes both pages.

### Text

Edit the HTML in `src/content/`. Keep it plain: paragraphs, headings, lists,
links. The builder takes care of three things for you:

- **Section numbers.** A section is `<h2 id="some-id"><span
  class="eyebrow">Label</span>Heading</h2>`. The builder numbers the eyebrows
  01, 02, … on each page, so Part 2 starts again at 01.
- **Contents.** Each page's contents list is built from its `<h2>` headings,
  where `{{TOC}}` appears in the intro fragment.
- **Footnotes.** Write `<sup data-note="key"></sup>` in the text and put the
  note in `notes.html` as `<li data-note="key">…</li>`. Each page numbers its
  notes in reading order and lists only the ones it cites.

The build stops with a message if a footnote is missing or unused, a link
points to an id that isn't on the page, or two elements share an id.

### Charts

A chart is a slot in the text, `<div class="chart" id="chartDice"></div>`,
drawn by a function in `charts/charts-*.js` from the data in
`charts/data-*.json` (`charts-1.js` reads `data-1.json`, and so on).

- **Move a chart:** move its slot. A page can leave out any chart.
- **Change a title, axis label or caption:** edit the strings in the chart's
  function (search the `.js` files for the slot id).
- **Change a number:** edit the JSON, and change any sentence that quotes it.
  The numbers come from the experiment outputs and the paper; they should
  only change when a result changes.

The diagrams are inline SVG: Figure 1 is in `intro.html`, Figure 2 and the
setup diagrams in `sections-1.html`. Edit their text there.

### Check

After building, open each page in a browser at desktop and phone widths.
Every chart should draw, nothing should scroll sideways, and the browser
console should show no errors.
