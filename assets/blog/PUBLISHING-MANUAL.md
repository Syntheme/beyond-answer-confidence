# Publishing the blog post on synthpop.ai by hand

A guide for the marketing team when nobody can create a Webflow API token
(that needs site admin rights). You create the two posts yourself in the CMS
by copying and pasting from a **kit**: one web page, sent to you by
Engineering, with a **Copy** button for every field and every piece of the
post. No coding and no admin rights are needed. With a token, the shorter
route is [`PUBLISHING.md`](PUBLISHING.md).

Webflow occasionally renames buttons; if a label here doesn't match exactly,
look for the closest one.

## What you are publishing

The post *"Does a decision model (like Jev) know when it is guessing?"* in
two parts, as two items in the site's blog collection (the posts listed under
**Resources**):

| Item | Content |
| --- | --- |
| Part 1 | What Jev's confidence tells you (10 interactive charts, 3 diagrams) |
| Part 2 | Asking Jev directly (8 interactive charts) |

The post contains interactive charts: readers can hover over or tap them to
see values, and open a data table under each one. They are drawn by a small
script, loaded by one line added to the blog post template. The diagrams are
images.

## Who does what

| Step | Who |
| --- | --- |
| Send the kit | Engineering |
| Add the chart script line to the blog template (once) | Someone with Designer access |
| Create the two items and paste the post into them | Marketing (CMS access is enough) |
| Fill in the remaining fields (banner, author, date, ...) | Marketing |
| Check on the staging site | Marketing and Engineering |
| Publish to the live site | Marketing |
| Change the post's text or charts later | Engineering sends a new kit, Marketing pastes it |
| Change title, summary, images, SEO later | Marketing, directly in Webflow |

## 1. What is already set

| | Part 1 | Part 2 |
| --- | --- | --- |
| Title | Does a decision model (like Jev) know when it is guessing? (Post 1) | Does a decision model (like Jev) know when it is guessing? (Post 2) |
| Summary | We ran about 575,000 queries against Jev, a decision model. Its confidence worked on familiar tasks and stayed high when it had nothing to go on. | We asked Jev, a decision model, whether it knows the answer. Its replies mostly picked up surface clues; questions about the case itself held up. |
| URL | `/resources/does-a-decision-model-know-when-it-is-guessing-post-1` | `/resources/does-a-decision-model-know-when-it-is-guessing-post-2` |

The kit contains all of these. You can change the title and summary at any
time. The URLs are different: the two parts link to each other by them, so if
you want other URLs, **tell Engineering before they make the kit**; after
that, don't change them (see "Things to avoid").

Both parts are published **together**: Part 1 links to Part 2 at the top and
at the end, and those links would lead to a missing page if Part 2 came
later.

Decided with marketing:

- **Launch date:** 1 October 2026, both parts.
- **Author:** Sharath Shankaranarayana.
- **Type:** Article (like the other long-form posts, e.g. the "Agentic
  engineering at Synthpop" series).
- **Banner images:** marketing's own, one per part (600 × 400, used in
  listings and link previews).

The titles and summaries follow the site's style for a series, as in
"Agentic engineering at Synthpop: … (Post 1)".

## 2. Get the kit

Engineering sends you a file called `blog-paste-kit.html`. It isn't secret;
any channel is fine. Save it and **open it in your browser** (double-click
it). It has:

- at the top, the chart script line for the blog template;
- for each part, its **Name**, **Slug** and **Summary**, then the post body
  as numbered **blocks** (4 for Part 1, 3 for Part 2).

Each **Copy** button copies one field or block exactly; it turns green when
done. Always use the buttons rather than selecting text by hand: the blocks
are code, and a missing character breaks the layout or the charts. Keep the
kit open while you work through the steps below.

## 3. Add the chart script to the blog template (once)

This needs someone who can open the site in the **Designer**. The line is
also here:

```html
<script src="https://cdn.jsdelivr.net/gh/Syntheme/beyond-answer-confidence@v0.1.1/assets/blog/webflow/blog.js" integrity="sha384-Qf2LUCpcIeRutwEWXFt7pkKdlCUho8POXxAUS5D/UEx84e1GimGhBnPpl0UWzlEL" crossorigin="anonymous" defer></script>
```

1. Copy the line: use **Copy** next to "Before </body> tag" at the top of
   the kit (or the copy button at the top right of the box above, on GitHub).
2. Open the site in the **Designer** and open the **Pages** panel.
3. Under **CMS Collection pages**, hover over the blog post template and click
   the gear icon (**Settings**).
4. Scroll to **Custom code**. In the **Before `</body>` tag** box, paste the
   line on its own line, below anything already there. If a line containing
   `beyond-answer-confidence` is already there, replace it instead of adding
   a second one.
5. Click **Save**. It takes effect the next time the site is published.

The line must stay exactly as it is, on **one line**: it carries a security
check, and if a single character changes (a line break added when copying
from a chat, e-mail or terminal counts), browsers refuse the script and the
charts don't appear. The script runs only on blog posts and does nothing on
posts without our charts.

## 4. Create the two items

For Part 1:

1. Open **CMS** → the blog collection → **New item** (the "+" button).
2. Fill in **Name**, **Slug** and **Summary** with the kit's **Copy**
   buttons for Part 1. Check the Slug field afterwards: Webflow fills it in
   from the Name as you type, so paste the kit's slug over whatever is there.
3. Paste the body, one block at a time, **in the kit's order**:
   1. Click into the post body field, on an empty line.
   2. Click the **+** that appears at the left of the line and choose the
      **</>** icon (*HTML embed* or *Custom code*).
   3. In the kit, click **Copy** on **Block 1**. Paste into the code box
      that opened, and click **Save & close**. A grey box appears saying
      "This embed will only appear on the published site": that's right.
   4. Click on the empty line **below** that grey box (press Enter after it
      if there is none) and repeat with Block 2, and so on to the last block.

   Paste nothing else into the body, and type nothing between the blocks.
   If an empty line stays between two grey boxes, delete it if you can; it
   only adds a little space.
4. Fill in the other fields as for any other post:
   - **Type**: *Article*; and the **category**, which is also used in the
     breadcrumbs.
   - **Author**: Sharath Shankaranarayana.
   - **Date**: the launch date, 1 October 2026.
   - **Banner / thumbnail image**: your 600 × 400 banner for this part.
   - **SEO title and description**, if the collection has them separately;
     otherwise the Name and Summary are used.
   - Anything else the collection asks for (required fields are marked).
5. **Save** (not **Publish**).

Repeat for Part 2 with the Part 2 section of the kit (3 blocks).

The editor's field names may differ slightly from the ones above; when in
doubt, open an older published post and copy what it has. If the body
field's **+** menu has no **</>** option, your role can't add code to posts:
ask someone with Designer access to do step 3, or to change your role.

## 5. Check on the staging site

The editor and the Designer preview don't run the chart script, so charts
only show on a published page. Check on the staging address
(`….webflow.io`) first, which the public doesn't use:

1. Set both items to be published (in the item, **Stage for publish** or the
   equivalent; not **Publish now**, which can put the item on the live site
   straight away).
2. Click **Publish** at the top right, tick **only** the `webflow.io`
   address, untick the live domain, and publish.
3. Open each post on the staging address and go through the checklist below,
   on a computer and on a phone.

### Checklist

- [ ] Title, author, date, banner and summary are right.
- [ ] The post starts with the links "Open-source code", "Paper …" and
      ends with numbered notes; nothing is missing in between (a missing
      block shows as a jump in the text, or as missing notes at the end).
- [ ] Every chart draws (10 in Part 1, 8 in Part 2). Hovering (or tapping)
      a chart shows values; "Show data table" opens a table.
- [ ] The three diagrams in Part 1 show (no broken-image icons).
- [ ] The "Contents" box near the top links to the sections.
- [ ] The links between Part 1 and Part 2 open the other part.
- [ ] Footnote numbers jump to the notes at the end, and back.
- [ ] On a phone, text fits the screen; wide charts (they say so) and
      diagrams scroll sideways inside their box, the page itself doesn't.
- [ ] Sharing a link (e.g. pasting it into a chat) shows the right title and
      image.

Send Engineering the two staging addresses: they can check the pages too.
Tell them about anything that looks wrong (a screenshot and the page address
are enough).

## 6. Publish to the live site

1. Click **Publish**, tick the live domain (and `webflow.io`), and publish.
2. Open both live URLs and repeat a quick check: charts, diagrams, links.
3. Tell Engineering the posts are live.

## Changing the post later

- **Title, summary, images, author, SEO, category:** change them in Webflow
  and publish. Nothing else is needed.
- **Text, numbers or charts in the post:** ask Engineering. They change the
  source and send you a new kit (and, rarely, a new script line). In each
  item, delete **all** the grey boxes from the body and paste the new blocks
  as in step 4.3; leave the other fields alone. Then check on staging and
  publish as in steps 5 and 6. Always replace all the blocks, not just one:
  blocks from different kits don't fit together.
- **An urgent typo** while Engineering is not around: double-click the grey
  box that contains it, change only the words (not the code around them),
  save and publish. **Then tell Engineering**, otherwise the next kit brings
  the typo back.

## Things to avoid

- **Don't edit, reorder or restyle the grey boxes,** and don't paste the
  post's text into the body from anywhere but the kit's Copy buttons (for
  example from the standalone HTML pages). Webflow's editor removes parts of
  the formatting from pasted text, and the charts need the blocks exactly as
  they are.
- **Don't change a post's URL (slug) after it is published.** The other part
  links to it, and links shared on social media break. If a URL really must
  change, ask Engineering first: the other part needs a new kit.
- **Don't remove or edit the chart script line** in the blog template's
  custom code (the one containing `beyond-answer-confidence`); the charts
  disappear without it.

## Taking the post down

- **Hide one part:** open the item in the CMS and unpublish it (or set it back
  to draft) and publish the site. The post disappears; nothing is lost.
- **Go back to an earlier version of the text:** ask Engineering for the kit
  of that version and paste it as under "Changing the post later".

## Troubleshooting

| What you see | Likely cause | What to do |
| --- | --- | --- |
| Grey boxes in the editor | Normal; the post shows only on the published page | Nothing |
| No **</>** in the body's **+** menu | Your role can't add code to posts | Ask someone with Designer access |
| No charts on the published page, text is fine | The script line is missing from the blog template's custom code or was changed, or the site wasn't published after it was added | Check the line (step 3), publish again; if that doesn't help, tell Engineering |
| Raw code (`<div class=...`) shows as text on the page | A block was pasted into the body as text, not into a **</>** embed | Delete that text and paste the block again with **+** → **</>** |
| Part of the post is missing, or appears twice | A block was skipped, pasted twice or out of order | Delete all grey boxes in that item and paste the blocks again in order |
| Charts missing only for one person | A browser extension (ad or script blocker) | Try another browser; nothing to fix on the site |
| Broken-image icons instead of diagrams | The image files aren't online | Tell Engineering |
| A part's link to the other part goes to a missing page | The other part isn't published, or its URL differs from the kit's | Publish it, or check its slug against the kit |
| Spacing or fonts look different from the rest of the blog | A clash with the site's styles | Screenshot to Engineering |
| A text edit came back after an update | It was made in Webflow but not in the source | Tell Engineering what to change in the source |
