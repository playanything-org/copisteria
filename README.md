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
  and where it does not go on past the page's first staff and past its last, as no system's line does.
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
  its measure boxes on a tilted page). An opening line, a bracket or a brace stands where its staves start. Where
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
