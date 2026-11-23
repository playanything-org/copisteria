# copisteria, Play Anything's fork

This is [Play Anything](https://github.com/playanything-org)'s fork of
[**copisteria**](https://github.com/kobimusic/copisteria), KobiMusic's optical music recognition reader: a scanned
page of sheet music in, MusicXML out. We keep it for Inscript's music reading. All credit for the reader, its models
and its design goes to KobiMusic (https://kobi.music); see upstream's repository for what copisteria is, how well it
reads and how to use it, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) here for how it works.

- **Code:** upstream's `main`, with the changes listed below on top. Upstream is pulled in deliberately, not
  continuously.
- **Weights:** upstream's, unchanged: [kobimusic/copista-28m](https://huggingface.co/kobimusic/copista-28m). Nothing
  here is retrained.
- **License:** Apache-2.0, as upstream. `LICENSE` and `NOTICE` are upstream's and stay with the code. Every file
  changed or added here says so in its first line, as section 4(b) of the license asks.

## Changes

Reading (what a page's MusicXML holds):

- **A staff the line finder finds twice is one staff** (`front.py` `_without_doubles`). On a scan that bows or tilts,
  a staff line is two runs of rows and the finder groups each set of five into a staff, half a space apart. The
  system then held a staff the page does not print, and the page's parts were planned on it.
- **A shaded page is masked with its shading divided out** (`front.py` `page_mask`, `_shading`). One ink threshold
  for the whole page loses the line that opens a system in a photograph's bright patch, and the system read as two.
  A page whose paper tone varies by more than `SHADED` is masked flat; any other page keeps the mask it had.
- **A line with a percussion clef on it is a one-line staff** (`front.py` `_one_line_staves`). The finder knows
  five-line staves only, so a triangle's or a tambourine's staff was not there and its system read a staff short.
- **One staff out of line does not part a system the caller states** (`front.py` `_joined_around`). A layout's
  join was refused where the two staves at it did not line up bar for bar. A staff of rests in a band score shows
  three measure boxes for six bars, ending two or three staff spaces off, and lines up with neither neighbour: a
  stated ten-staff system was read as ten systems of one staff, page after page. The join now stands where the
  three staves above it and the three below it line up with each other: of the pairs across it that show three
  bar lines or more, at least half. Two systems with bar lines of their own have hardly such a pair.
- **No voice holds two events at once** (`write.py` `_unlapped`). On a part of several staves the model's onset
  places each event, and where it puts one before the last has ended (a hand's two voices read as one, an onset
  read wrong) the voice was written with a backup inside it. A reader of the score that keeps one event at a time
  in a voice drops what overlaps: on 326 production pages about 1,700 of 83,754 noteheads were written into a voice
  that was still sounding. What overlaps now goes to a voice of its own while the staff has one free; every event
  is still written at the time the model gave it. Ties are paired along the voice as it was read; the notes of a
  tie, and a run of notes with its beam and its tuplet, stay in one voice where they can; and the staff's
  directions are written with the voice whose next event they stand at.
- **A bar the writer cannot voice costs that bar, never the page** (`write.py` `_decode_rhythm`, `write`). The
  rhythm decoder times each event on a grid of 48 steps a quarter under each tuplet it may be; a dotted 128th is 2.25
  steps under every one, the decoder had no choice for it, and the page's whole writing failed (a 22-staff orchestral
  page delivered as empty bars). A value no choice times now steps to the next event at its nearest step under its
  own reading and is written as read (what a voice's last event leaves of the bar is scored as before); and a staff's bar whose voicing or timing still fails is written empty, as a staff with
  nothing read in it, and named in `facts["unvoiced"]`, while every other bar of the page stands.
- **A layout parts two staves the page joins by a line alone** (`front.py` `_line_alone`). A page pasted together
  from cuttings has a black edge down the left of two one-staff systems; the front end took it for the line that
  opens a system, the caller's layout (two systems of one staff) was refused as "parting staves the page joins", and
  one voice's two lines were written as two parts at once. A layout now parts such a join where no brace or bracket
  spans it, the two staves' bar lines plainly do not line up, and neither the staves around it nor the two sides'
  grid of bar lines say one system.
- **Successive systems keep their own staves** (`front.py` `_stacked`). Tight spacing and ink near the left
  margin can leave a gap uncertain even though no opening line or barline crosses it. Where the opening clefs
  and brace positions repeat a complete staff pattern, the front end puts the boundary before bar segmentation
  and part planning. A visible connector, a group symbol across the gap or shared note onsets keep the staves
  together. A matching prefix can begin a cropped last system where the page also shows its complete pattern.
  On a page of solo systems a lone ambiguous pair can be separated by the surrounding spacing and its own
  conflicting bar grids.
- **A chord read as separate notes is one chord, and each voice under a beam is beamed** (`write.py` `_unlapped`,
  `_attach`). A left hand's bass note and chord in turn stand under one beam; the model reads the chord's notes as
  events of their own at one onset. Written in voices of their own they lost the beam, which went to the voice
  that held most of its notes. Notes that stand one over the other, at one time and of one length, now join the
  note they sound with as its chord; and where a beam's notes are still written in two voices, each is beamed.
- **The edge of a binding's shadow is not a system's pale opening line** (`front.py` `_pale_at`, `_runs_on`).
  Upstream's pale-line finder (for a small render, whose opening line is one grey pixel the ink mask does not count)
  took the edge of a scan's binding shadow for one: with the frame taken off, the shadow's edge is pale for the
  page's whole height, with paper a staff space to either side, and every staff of the page was joined into one
  system (a part's eight lines read as eight instruments playing at once; 29 pages of one upload). A line now counts
  only where the page changes within a quarter of a staff space of it, back toward its paper or into darker ink,
  and where it does not go on past the page's first staff and past its last, as no system's line does. When the
  staff-line starts are visible, pale opening ink must also lie within one and a half staff spaces of those starts.
  This keeps a detached margin stroke from joining successive systems.
- **A staff the measure boxes make twice is one staff** (`front.py` `_clear_rows`). Where the line finder finds no
  staves (a photograph taken by hand), the staves are the rows of the detector's measure boxes, and it boxes a
  tilted or blurred staff's bars at two heights: two rows, one printed staff, and a system of three staves where two
  are printed. No two staves of a page overlap, so of two that share more than a tenth of a staff's height one goes,
  the one its boxes are less sure of, and its bars are the other's where that has none (two rows that each have
  notes of their own along the same stretch of the page are two staves printed close, and stay). The same tenth now holds for the line finder's doubles (`_without_doubles`,
  which asked for half).
- **Ink across a gap opens a system only where its staves' lines begin at it** (`front.py` `_meets_staves`,
  `_ink_links`). The search for the line that opens a system reaches three staff spaces left of the page's leftmost
  staff, and took any ink across a gap there for that line. In a photograph taken by hand, the edge of the binder
  or folder the sheet lies in, or of the sheet itself, runs down that margin past every gap. It joined every staff
  of a one-instrument part into one system, and the "line" it was taken for overrode the bar lines, which did not
  line up: one tenor drum part of six one-staff systems was delivered as four drum parts. Ink across a gap now
  counts as the line only where it is a stroke no wider than a brace and, within three staff spaces past it, each
  of the two staves shows its lines running on (three of them, or a one-line staff's one, a staff space or two off
  its measure boxes on a tilted page; on a faded scan whose lines the mask does not part, ink across most of it). An opening line, a bracket or a brace stands where its staves start. Where
  the only ink there is not such a stroke, the gap is white, unless the detector saw a brace or bracket over both
  staves.
- **A multi-bar rest outside every bar box is a bar the detector missed** (`front.py` `_fill_gaps`, `LOOSE`). On a
  page whose staves are the rows of the detector's measure boxes, a stretch of a staff with notes or rests outside
  every box became a bar of its own, and a multi-bar rest did not. A part's opening multi-bar rest, boxed only
  faintly, was then no bar: the staff began after it, without its clef (a percussion part read as treble), and its
  bars were not counted.
- **No pedal mark under a part of one staff** (`write.py` `_attach`). The detector calls a voice's lyrics pedal
  marks, syllable by syllable: a score for four voices and piano, its text in Chinese, was written with 380 pedal
  lines under its voices and none is printed. A pedal is under a keyboard's staves, so one is written only for a
  part of two staves or more.
- **A full score whose staves each show a few of one grid of bar lines is one system** (`front.py` `_one_grid`,
  `_loses_bars`). A big-band page under a "PREVIEW" watermark showed each of its 19 staves a different two or three
  of the system's bar lines, and the measure boxes' edges stood a beat off the lines. Each pair at a join the layout
  made was "apart", and the stated one-system layout was refused: the page was read as seven systems of five parts and
  14 instruments' music was lost. A join is also taken where the bar lines that the staves on each side show, taken
  together, fall on the lines the other side's staves show (from the boxes' edges or from the bar-line glyphs). A
  join's fewer bars are judged against what most of its staves show, not the one staff the watermark cut at every
  beat.
- **A system shaped as two of another on the page is two** (`front.py` `_repeats`). A choir-and-piano page whose
  systems hold two bars at the same places, with lyrics written by hand in the gap, read its first two systems as one
  of eight staves. A system whose opening clefs are another's (read apart on the same page) repeated is parted after
  each copy of three staves or more, where no ink line runs through the gap and no brace or bracket spans it, unless
  another system on the page has the joined system's whole shape.
- **Percussion staves of one or two lines stand three spaces apart** (`front.py` `_one_line_staves`). A Latin big
  band's bongos, congas, clave and maracas are printed three to five staff spaces apart, and only the first was taken:
  the others were held to a five-line staff's distance from it. A one-line staff is now held 2.5 staff spaces from
  another one-line staff's line (their boxes, a five-line staff's height, then touch: such neighbours are one system,
  `_links`), and a staff of two lines (its clef between them, a space from each) is taken on
  the nearer line, as before.
- **A keyboard staff read as percussion without its clef on the page is pitched** (`write.py` `_pitched_clefs`). The
  model read an organ's bass staff F, then percussion for a line, then F, and a piano's left hand as percussion,
  with no percussion clef glyph on either: the notes were written unpitched and the staff opened a part of its own. A
  percussion clef the page does not show is read as the staff's pitched clef where the staff is one of a braced part
  with a pitched staff, or reads pitched at least as often as percussion.
- **Hairpins are written once each, with their own stop** (`write.py` `_hairpins`). Every hairpin detection was
  written as a start at its box's left and a stop at its right, unnumbered: one hairpin detected twice opened a second
  wedge while the first was open, and the staff-join check downstream then refused to join a piano's two staves. A
  hairpin detected twice (one kind, one side) is the longer box; of two that overlap otherwise the first stops before
  the second; a sliver narrower than a staff space is none; and two at once (above and below the staff) are
  numbered apart. A short hairpin between two onsets still starts and stops at one instant.

What a caller may tell the front end (`front.py` `build`, `read.py` `read`; without it nothing changes):

- **`layout`**: the staves of each of the page's systems, top to bottom, as the caller knows them from a reading of
  the whole score at a glance (`_with_layout`). It groups the staves found into the counted systems, drops what is
  hardly a staff where more were found (an inset ossia, a line of text), and takes a staff the finder saw and left
  where fewer were. It is evidence, and may be wrong: it joins, it does not part staves the page joins; a join it
  adds has to fit the page (bar lines that line up, a gap narrower than one it leaves between systems, a system
  that keeps the bars its staves show); a staff that holds music is not dropped; `most` caps a system at the
  score's staves. A layout that cannot be met is not used, and `Layout.prior` says what was done and why.
  Where the caller counts more staves than were found, a staff of rests is taken too: the finder's own staves that
  hold nothing (it leaves them out as lines that are not a staff), five even rows of ink in a gap between two
  staves that are as wide as a staff (`_gap_staves`: a blurred or thick-lined staff the finder does not see), and,
  on a page whose staves are the rows of its measure boxes, the finder's staves that stand clear of those
  (`_lined_staves`; such a staff takes its neighbour's bars). Without a layout, or with one that counts no more
  than the page shows, none of them is a staff.
- **`insist`**: with a `layout`, the layout's systems are taken even where the page refuses them, for a caller that
  has a reading without it that falls well short of it and chooses between the two itself. On a photographed page
  the ink between systems can run them together (eight staves read as one system of three bars, against a count of
  four piano systems), and on a page with a few more or fewer staves than counted nothing of the count was used.
  Insisted on, the staves found are grouped as counted where they are as many, else parted into as many systems as
  counted at the page's widest gaps, those no brace or bracket spans and the page does not join first; a join made
  so stands even where its system holds fewer bars than its staves. A layout that counts no staves or more than
  `most` is still not used. `Layout.prior` says `insisted` and what the page said against it.

What a caller may ask of the writer (`write.py` `write`; without these arguments the MusicXML is unchanged):

- **`facts`**: a dict that is given what the writer decided and the MusicXML does not say: the part plan, each
  system's places, the measures as events, what `_chop` dropped or shortened, the rests written unprinted for time
  no event was read in, and the ties read and not written.
- **`align=False`**: every system's staves are counted into the parts from the top, as before `_align_systems`,
  for a caller that places a system's staves itself.
- **`roster`** (`roster.py`): the score's instruments as the caller knows them, each as its number of staves. A page
  whose every system the roster explains is planned on the roster; any other page is planned as without it, and
  `facts["roster"]` says why. A system of fewer staves is placed among the roster's instruments by its braces and
  opening clefs alone: where they fit it more than one way (two treble staves that could be any two of four voices),
  the page is not explained, since a guess would give players' music to others and leave them empty.
- **`clefs`, `among`** (`roster.py`): what the caller read in the score's print for that roster. `clefs` is the clef
  each of the roster's staves opens with, used where the page shows no whole system of its own to read it from (a
  clef with and without its octave mark is one clef there: a tenor's staff is read either way; a clef the caller left
  out is unknown, and a staff may as well be that instrument's). `among` says, for each staff of each of the page's
  systems, which of the roster's instruments it may be, as the label printed by the staff names them: "T." and "B."
  over the piano are the tenor and the bass, where braces and clefs could not say which two of four voices sing.
  Whose label a label is, is the caller's to decide; nothing here reads a name. A staff stands only at an instrument
  it may be, a whole system one of whose staves may not be the instrument at its place is not this roster's page, and
  for what remains the rule above holds: one fit, or the page is planned as without the roster. A row of `among` that
  does not fit its system says the caller's systems are not the ones found: the page is then planned as without the
  roster, since the roster came with what was read of other systems. Nothing the caller says is written as a part
  name.

The read path:

- **Checkpoints load with `weights_only=True`** (`read.py` `load_model`, `detect.py` `Detector`). Both checkpoints
  load that way and hold no unsafe global, so a weights file is never unpickled as code.
- **One detector for the process** (`pipeline.py` `detections`). A page that had no cached detections built a new
  `Detector` (a checkpoint load) for itself; the process now keeps one, the one `featcache` already kept.
