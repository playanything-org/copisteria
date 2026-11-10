# Changes in this fork

This is Play Anything's fork of [copisteria](https://github.com/kobimusic/copisteria) by KobiMusic, kept for
[Inscript](https://github.com/playanything-org)'s music reading. It follows upstream `main`; the `inscript` branch
carries the changes below on top of it. `LICENSE` and `NOTICE` are upstream's and stay with it. Every file changed
here says so in its first line, as section 4(b) of the Apache License 2.0 asks.

The weights are upstream's, unchanged: [kobimusic/copista-28m](https://huggingface.co/kobimusic/copista-28m).

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

What a caller may ask of the writer (`write.py` `write`; without these arguments the MusicXML is unchanged):

- **`facts`**: a dict that is given what the writer decided and the MusicXML does not say: the part plan, each
  system's places, the measures as events, what `_chop` dropped or shortened, the rests written unprinted for time
  no event was read in, and the ties read and not written.
- **`align=False`**: every system's staves are counted into the parts from the top, as before `_align_systems`,
  for a caller that places a system's staves itself.
- **`roster`** (`roster.py`): the score's instruments as the caller knows them, each as its number of staves. A page
  whose every system the roster explains is planned on the roster; any other page is planned as without it, and
  `facts["roster"]` says why.

The read path:

- **Checkpoints load with `weights_only=True`** (`read.py` `load_model`, `detect.py` `Detector`). Both checkpoints
  load that way and hold no unsafe global, so a weights file is never unpickled as code.
- **One detector for the process** (`pipeline.py` `detections`). A page that had no cached detections built a new
  `Detector` (a checkpoint load) for itself; the process now keeps one, the one `featcache` already kept.
