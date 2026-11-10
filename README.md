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

What a caller may tell the front end (`front.py` `build`, `read.py` `read`; without it nothing changes):

- **`layout`**: the staves of each of the page's systems, top to bottom, as the caller knows them from a reading of
  the whole score at a glance (`_with_layout`). It groups the staves found into the counted systems, drops what is
  hardly a staff where more were found (an inset ossia, a line of text), and takes a staff the finder saw and left
  where fewer were. It is evidence, and may be wrong: it joins, it does not part staves the page joins; a join it
  adds has to fit the page (bar lines that line up, a gap narrower than one it leaves between systems, a system
  that keeps the bars its staves show); a staff that holds music is not dropped; `most` caps a system at the
  score's staves. A layout that cannot be met is not used, and `Layout.prior` says what was done and why.

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

The read path:

- **Checkpoints load with `weights_only=True`** (`read.py` `load_model`, `detect.py` `Detector`). Both checkpoints
  load that way and hold no unsafe global, so a weights file is never unpickled as code.
- **One detector for the process** (`pipeline.py` `detections`). A page that had no cached detections built a new
  `Detector` (a checkpoint load) for itself; the process now keeps one, the one `featcache` already kept.
