# Publishing the blog post on synthpop.ai

A guide for the marketing team. It explains what is being published, who
does what, and how to put the post live, check it, change it later and take
it down. No coding is needed on your side; the steps marked **Engineering**
are done by whoever runs the publishing scripts (see the Webflow section of
`README.md` next to this file).

Webflow occasionally renames buttons; if a label here doesn't match exactly,
look for the closest one.

## What you are publishing

The post *"Does a Decision Model (like Jev) know when it is guessing?"* in
two parts, as two items in the site's blog collection:

| Item | Content |
| --- | --- |
| Part 1 | What Jev's confidence tells you (10 interactive charts, 3 diagrams) |
| Part 2 | Asking Jev directly (8 interactive charts) |

(A single-page version of the whole post also exists. It is not published by
default; say if you want it as well.)

The post contains interactive charts: readers can hover over or tap them to
see values, and open a data table under each one. They are drawn by a small
script that is added to the site once. The diagrams are images.

## Who does what

| Step | Who |
| --- | --- |
| Decide titles, URLs, summaries, images, author, date, category | Marketing |
| Create the API token (once) | Webflow site admin |
| Put the posts into the CMS as drafts | Engineering |
| Install the chart script on the site (once) | Engineering, after marketing agrees |
| Fill in the remaining fields (cover image, author, SEO, ...) | Marketing |
| Check on the staging site | Marketing and Engineering |
| Publish to the live site | Marketing |
| Change the post's text or charts later | Engineering, then Marketing publishes |
| Change title, summary, images, SEO later | Marketing, directly in Webflow |

## 1. Decisions to make before we start

Send Engineering your answers to these; anything left open gets the default
shown.

- **URLs (slugs).** Defaults:
  - `/blog/does-a-decision-model-know-when-it-is-guessing-part-1`
  - `/blog/does-a-decision-model-know-when-it-is-guessing-part-2`

  The two parts link to each other by these URLs, so **choose them before the
  posts go in and don't change them afterwards** (see "Things to avoid").
- **Titles.** Default: the post title followed by "Part 1" / "Part 2". You
  can change them in Webflow at any time.
- **Summaries** (the short description in listings and search results).
  Defaults are written; you can change them in Webflow at any time.
- **Author, publication date, category or tags, cover image, social sharing
  image.** These are filled in by you in Webflow (step 4). Engineering can
  make a cover image from one of the post's diagrams if you want one.
- **One launch or two?** Part 1 links to Part 2 at the top and at the end. If
  Part 2 is published later, those links lead to a missing page until then.
  Either publish both parts together (recommended), or tell Engineering to
  remove the links from Part 1 until Part 2 is out.
- **House style.** The site's styles apply to the post. Two of them change
  how the post looks, and either way is fine:
  - headings are shown In Title Case (the post is written in sentence case);
  - links get the site's coloured gradient.

  Say if the post should keep the site style (default) or its own.

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
   | Custom code | Read and write |

4. Copy the token and pass it to Engineering **through the password manager**,
   never by e-mail or chat. Webflow shows it only once.
5. After the launch, the token can be deleted on the same screen and a new
   one made for the next update.

The token lets the scripts create and change CMS items and install the chart
script. It cannot publish anything: publishing always stays a click in
Webflow by you.

### Check the blog template (usually nothing to do)

If existing blog posts show their text on the site, the blog template is
already set up and there is nothing to do. Engineering checks this with the
token before anything is changed.

### Install the chart script

Engineering adds one script to the site footer (it appears as
**BeyondAnswerConfidenceBlog**). It does nothing on pages without the post,
so the rest of the site is unaffected. It needs the site's paid plan (custom
code), and it takes effect the next time the site is published.

## 3. Engineering puts the posts in as drafts

Engineering first shows what would be created (a dry run), then creates the
two items as **drafts** and tells you when they are there. Drafts are not
visible on the live site.

## 4. Fill in the remaining fields

1. Open **CMS** → the blog collection → the new Part 1 item.
2. The post body appears as a few **grey boxes** saying "This embed will
   only appear on the published site". That is normal: the post is inside
   them, and it shows on the published page, not in the editor.
3. Fill in the fields the scripts don't set: author, date, category, cover
   image, social sharing image, SEO title and description, and anything else
   the template uses. Adjust the title and summary if you like; later updates
   from Engineering won't overwrite them.
4. Save. Repeat for Part 2.

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
- **Don't remove the BeyondAnswerConfidenceBlog script** from the site; the
  charts disappear without it.
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
| No charts on the published page, text is fine | The chart script isn't installed, or the site wasn't published after it was installed | Publish the site again; if that doesn't help, tell Engineering |
| Charts missing only for one person | A browser extension (ad or script blocker) | Try another browser; nothing to fix on the site |
| Broken-image icons instead of diagrams | The image files aren't online yet | Tell Engineering |
| A part's link to the other part goes to a missing page | The other part isn't published, or its URL was changed | Publish it, or ask Engineering to update the link |
| Spacing or fonts look different from the rest of the blog | A clash with the site's styles | Screenshot to Engineering |
| A text edit came back after an update | It was made in Webflow but not in the source | Tell Engineering what to change in the source |
