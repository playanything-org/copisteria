# Changes in this fork

This is Play Anything's fork of [copisteria](https://github.com/kobimusic/copisteria) by KobiMusic, kept for
[Inscript](https://github.com/playanything-org)'s music reading. It follows upstream `main`; the `inscript` branch
carries the changes below on top of it. `LICENSE` and `NOTICE` are upstream's and stay with it. Every file changed
here says so in its first line, as section 4(b) of the Apache License 2.0 asks.

The weights are upstream's, unchanged: [kobimusic/copista-28m](https://huggingface.co/kobimusic/copista-28m).

## Changes

- **Checkpoints load with `weights_only=True`** (`read.py` `load_model`, `detect.py` `Detector`). Both checkpoints
  load that way and hold no unsafe global, so a weights file is never unpickled as code.
- **One detector for the process** (`pipeline.py` `detections`). A page that had no cached detections built a new
  `Detector` (a checkpoint load) for itself; the process now keeps one, the one `featcache` already kept.
