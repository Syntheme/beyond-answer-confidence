# Publishing the blog post on synthpop.ai

A guide for the marketing team. It explains what is being published, who
does what, and how to put the post live, check it, change it later and take
it down. No coding is needed on your side; the steps marked **Engineering**
are done by whoever runs the publishing scripts (see the Webflow section of
`README.md` next to this file).

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

(A single-page version of the whole post also exists. It is not published by
default; say if you want it as well.)

The post contains interactive charts: readers can hover over or tap them to
see values, and open a data table under each one. They are drawn by a small
script, loaded by one line added to the blog post template. The diagrams are
images.

## Who does what

| Step | Who |
| --- | --- |
| Decide titles, URLs, summaries, images, author, date, category | Marketing |
| Create the API token (once) | Webflow site admin |
| Put the posts into the CMS as drafts | Engineering |
| Add the chart script line to the blog template (once) | Someone with Designer access, with the line in this guide |
| Fill in the remaining fields (cover image, author, SEO, ...) | Marketing |
| Check on the staging site | Marketing and Engineering |
| Publish to the live site | Marketing |
| Change the post's text or charts later | Engineering, then Marketing publishes |
| Change title, summary, images, SEO later | Marketing, directly in Webflow |

## 1. What is already set, and what we need from you

### Already set (in the post's sources)

| | Part 1 | Part 2 |
| --- | --- | --- |
| Title | Does a decision model (like Jev) know when it is guessing? (Post 1) | Does a decision model (like Jev) know when it is guessing? (Post 2) |
| Summary | We ran about 575,000 queries against Jev, a decision model. Its confidence worked on familiar tasks and stayed high when it had nothing to go on. | We asked Jev, a decision model, whether it knows the answer. Its replies mostly picked up surface clues; questions about the case itself held up. |
| URL | `/resources/does-a-decision-model-know-when-it-is-guessing-post-1` | `/resources/does-a-decision-model-know-when-it-is-guessing-post-2` |

Engineering puts these into the items. You can change the title and summary
in Webflow at any time. The URLs are different: the two parts link to each
other by them, so if you want other URLs, **say so before the posts go in**;
after that, don't change them (see "Things to avoid").

Both parts are published **together**: Part 1 links to Part 2 at the top and
at the end, and those links would lead to a missing page if Part 2 came
later.

### Decided with marketing

- **Launch date:** 1 October 2026, both parts.
- **Author:** Sharath Shankaranarayana.
- **Type:** Article (like the other long-form posts, e.g. the "Agentic
  engineering at Synthpop" series).
- **Banner images:** marketing's own, one per part (600 × 400, used in
  listings and link previews).

### House style

The titles and summaries follow the site's style for a series, as in
"Agentic engineering at Synthpop: … (Post 1)": sentence case, "(Post 1)" and
"(Post 2)" at the end, URLs ending in `-post-1` and `-post-2`, and a summary
that reads on its own. Inside the post, the site's styles apply: headings are
shown In Title Case and links get the site's coloured gradient.

## 2. One-time setup (site admin)

### Create the API token

1. In Webflow, open the synthpop.ai site's **Site settings** →
   **Apps & integrations** → **API access** → **Generate API token**.
2. Name it, for example, "Blog publishing".
3. Give it these permissions and leave everything else at *No access*:

   | Permission | Access |
   | --- | --- |
   | CMS | Read and write |
   | Sites | Read-only |
   | Pages | Read-only |

4. Copy the token and pass it to Engineering **through the password manager**,
   never by e-mail or chat. Webflow shows it only once.
5. After the launch, the token can be deleted on the same screen and a new
   one made for the next update.

The token lets the scripts create and change CMS items. It cannot publish
anything: publishing always stays a click in Webflow by you. (It also cannot
add the chart script: Webflow allows that only by hand or through a
registered app, so it is pasted in once, below.)

### Check the blog template (usually nothing to do)

If existing blog posts show their text on the site, the blog template is
already set up and there is nothing to do. Engineering checks this with the
token before anything is changed.

### Add the chart script to the blog template

The charts are drawn by a script that the blog post template loads with this
one line:

```html
<script src="https://cdn.jsdelivr.net/gh/Syntheme/beyond-answer-confidence@v0.1.1/assets/blog/webflow/blog.js" integrity="sha384-Qf2LUCpcIeRutwEWXFt7pkKdlCUho8POXxAUS5D/UEx84e1GimGhBnPpl0UWzlEL" crossorigin="anonymous" defer></script>
```

Add it once:

1. Copy the line. On GitHub, use the **copy button** at the top right of the
   box above; that copies it exactly, on one line.
2. Open the site in the **Designer** and open the **Pages** panel.
3. Under **CMS Collection pages**, hover over the blog post template and click
   the gear icon (**Settings**).
4. Scroll to **Custom code**. In the **Before `</body>` tag** box, paste the
   line on its own line, below anything already there. If a line containing
   `beyond-answer-confidence` is already there, replace it instead of adding
   a second one.
5. Click **Save**. It takes effect the next time the site is published.

The line must stay exactly as above, on **one line**: it carries a security
check, and if a single character changes (a line break added when copying
from a chat, e-mail or terminal counts), browsers refuse the script and the
charts don't appear. The script runs only on blog posts and does nothing on
posts without our charts. Custom code needs a paid site plan.

If Engineering changes the charts later, they update the line here and tell
you; replace the old one in the same box.

## 3. Engineering puts the posts in as drafts

Engineering first shows what would be created (a dry run), then creates the
two items as **drafts** and tells you when they are there. Drafts are not
visible on the live site.

## 4. Fill in the remaining fields

The blog template already shows the title, author and date above every post,
so nothing needs adding to the template: you only fill in the item's fields.
The post body starts after the title on purpose; the title comes from the
item's **Name**.

1. Open **CMS** → the blog collection → the new Part 1 item. It is marked
   *Draft*.
2. The post body appears as a few **grey boxes** saying "This embed will
   only appear on the published site". That is normal: the post is inside
   them, and it shows on the published page, not in the editor. Leave them
   as they are.
3. Check the fields Engineering filled in, and change them if you like
   (later updates from Engineering won't overwrite them):
   - **Name**: the title, shown in the dark header and as the page and
     sharing title.
   - **Slug**: the URL. Don't change it (see "Things to avoid").
   - **Summary**: the short description in listings, search results and
     link previews.
4. Fill in the fields our posts on synthpop.ai show, the same way as for
   other posts:
   - **Type** label (other posts say *Article*) and **category**, which is also
     used in the breadcrumbs.
   - **Author**: picks the name and photo shown in the header.
   - **Date**: the date shown under the author. Use the launch date; both parts
     can have the same one.
   - **Banner / thumbnail image** (other posts use a 600 × 400 banner):
     shown in listings and as the image when the link is shared. Engineering
     can make one from a diagram in the post.
   - **SEO title and description**, if the collection has them separately;
     otherwise the Name and Summary are used.
   - Anything else the collection asks for (required fields are marked).
5. **Save** (not **Publish**). Repeat for Part 2.

The editor's field names may differ slightly from the ones above; when in
doubt, open an older published post and copy what it has.

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

- [ ] Title, author, date, cover image and summary are right.
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

Tell Engineering about anything that looks wrong (a screenshot and the page
address are enough).

## 6. Publish to the live site

1. Click **Publish**, tick the live domain (and `webflow.io`), and publish.
2. Open both live URLs and repeat a quick check: charts, diagrams, links.
3. Tell Engineering the posts are live.

## Changing the post later

- **Title, summary, images, author, SEO, category:** change them in Webflow
  and publish. Nothing else is needed.
- **Text, numbers or charts in the post:** ask Engineering. They change the
  source, put the new version into the same items, and tell you; you check on
  staging and publish as in steps 5 and 6. Only the post body changes; your
  fields stay as they are.
- **An urgent typo** while Engineering is not around: double-click the grey
  box that contains it, change only the words (not the code around them),
  save and publish. **Then tell Engineering**, otherwise their next update
  brings the typo back.

## Things to avoid

- **Don't delete, reorder or restyle the grey boxes,** and don't paste the
  post's text into the body from elsewhere (for example from the standalone
  HTML pages). Webflow's editor removes parts of the formatting when text is
  pasted, and the charts need the boxes as they are.
- **Don't change a post's URL (slug) after it is published.** The other part
  links to it, links shared on social media break, and the next update from
  Engineering would create a second copy instead of updating the post. If a
  URL really must change, ask Engineering first.
- **Don't remove or edit the chart script line** in the blog template's
  custom code (the one containing `beyond-answer-confidence`); the charts
  disappear without it.
- **Don't rename or delete the blog collection's body field** while the post
  is live.

## Taking the post down

- **Hide one part:** open the item in the CMS and unpublish it (or set it back
  to draft) and publish the site. The post disappears; nothing is lost.
- **Go back to an earlier version of the text:** ask Engineering; they can put
  any earlier version back into the item.

## Troubleshooting

| What you see | Likely cause | What to do |
| --- | --- | --- |
| Grey boxes in the editor | Normal; the post shows only on the published page | Nothing |
| No charts on the published page, text is fine | The script line is missing from the blog template's custom code or was changed, or the site wasn't published after it was added | Check the line (section 2), publish again; if that doesn't help, tell Engineering |
| Charts missing only for one person | A browser extension (ad or script blocker) | Try another browser; nothing to fix on the site |
| Broken-image icons instead of diagrams | The image files aren't online yet | Tell Engineering |
| A part's link to the other part goes to a missing page | The other part isn't published, or its URL was changed | Publish it, or ask Engineering to update the link |
| Spacing or fonts look different from the rest of the blog | A clash with the site's styles | Screenshot to Engineering |
| A text edit came back after an update | It was made in Webflow but not in the source | Tell Engineering what to change in the source |
