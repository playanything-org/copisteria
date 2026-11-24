# Modified by Play Anything (https://github.com/playanything-org/copisteria): see README.md.
"""Find the printed staff at a head whose pitch could not be decoded."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

from .vocab import POS_MAX


@dataclass
class StaffLines:
    middle: float
    space: float
    slope: float
    strength: float

    def position(self, y: float) -> float:
        return 2 * (self.middle - y) / self.space


def _staff_lines(gray, x: float, y: float, space: float, radius: float,
                 slope: float | None = None) -> list[StaffLines]:
    """Score five evenly spaced ink ridges by sampling along their local slope."""
    top, bottom = max(0, round(y - radius)), min(gray.shape[0], round(y + radius))
    rows = np.arange(top, bottom)
    offsets = np.linspace(-4 * space, 4 * space, 25)
    offsets = offsets[(x + offsets >= 0) & (x + offsets < gray.shape[1])]
    if len(offsets) < 9:
        return []
    columns = np.broadcast_to(x + offsets, (len(rows), len(offsets)))
    slopes = np.linspace(-.7, .7, 41) if slope is None else slope + np.linspace(-.1, .1, 9)
    found = []
    for angle in slopes:
        values = map_coordinates(gray, [rows[:, None] + angle * offsets, columns],
                                 order=1, mode="nearest", output=np.float32)
        # A ridge must span the window. A stem or a beam filling a small part of it cannot set the row level.
        profile = -np.quantile(values, .65, axis=1)
        peaks, props = find_peaks(profile, prominence=14, distance=max(2, round(.3 * space)))
        for i, start in enumerate(peaks[:-4]):
            for end in peaks[i + 4:i + 9]:
                step = (end - start) / 4
                if not .52 * space <= step <= 1.55 * space:
                    continue
                expected = start + np.arange(5) * step
                nearest = np.argmin(abs(peaks - expected[:, None]), axis=1)
                lines = peaks[nearest]
                error = abs(lines - expected)
                if max(error) > max(1.25, .14 * step):
                    continue
                strength = float(np.mean(props["prominences"][nearest]) - 70 * np.mean(error) / step)
                if strength >= 40:
                    found.append(StaffLines(top + float(np.mean(lines)), float(step), float(angle), strength))
    best: list[StaffLines] = []
    for band in sorted(found, key=lambda b: -b.strength):
        # Five-line fits sharing at least three lines are alternatives for the same staff.
        if not any(abs(other.middle - band.middle) < 2.35 * min(other.space, band.space) for other in best):
            best.append(band)
    return best


def _traced_clef(gray, x: float, lines: StaffLines, clefs: list):
    """Follow the lines of the selected staff to its preceding clef, staying on the same ink."""
    if not clefs:
        return None
    end = min(c.cx for c in clefs)
    last_x, misses = x, 0
    while x >= end:
        step = 5 * lines.space
        for c in clefs:
            if x - step < c.cx <= x and abs(c.cy - lines.middle - lines.slope * (c.cx - last_x)) < 1.4 * lines.space:
                return c
        x -= step
        y = lines.middle + (x - last_x) * lines.slope
        found = _staff_lines(gray, x, y, lines.space, 3.5 * lines.space, lines.slope)
        close = [b for b in found if abs(b.middle - y) <= 1.15 * lines.space
                 and .79 * lines.space <= b.space <= 1.21 * lines.space]
        if close:
            lines = min(close, key=lambda b: abs(b.middle - y) / lines.space - b.strength / 330)
            last_x, misses = x, 0
        else:
            misses += 1
            if misses == 3:
                break
    return None


def _detected_position(head) -> int:
    try:
        return int(head.attrs.get("staff_position", 0))
    except (TypeError, ValueError):
        return 0


def _ledger_staff(gray, head, found: list[StaffLines], detected: int) -> StaffLines | None:
    """A very long stem can carry ledger heads across another staff's lines."""
    if abs(detected) < 12:
        return None
    x0, _, x1, _ = head.box
    for band in found:
        pos = band.position(head.cy)
        # At the detector's limit only the direction and a minimum distance survive.
        if pos * (1 if detected > 0 else -1) < abs(detected) - 3:
            continue
        if abs(detected) < 22 and abs(pos - detected) > 3:
            continue
        side = 1 if head.cy > band.middle else -1
        y0, y1 = sorted((round(head.cy), round(band.middle + side * 2 * band.space)))
        if y1 - y0 < 4 * band.space:
            continue
        left, right = max(0, round(x0 - 2)), min(gray.shape[1], round(x1 + 2))
        stroke = gray[max(0, y0):y1, left:right]
        if not stroke.size:
            continue
        paper = float(np.quantile(stroke, .9))
        if float((stroke < paper - 60).mean(axis=0).max()) > .85:
            return band
    return None


def _ledger_reaches_limit(gray, head, staff: StaffLines) -> bool:
    """Count ink between the outer staff line and the head before trusting an extrapolation."""
    side = 1 if head.cy > staff.middle else -1
    start = staff.middle + side * 2.55 * staff.space
    end = head.cy + side * .65 * staff.space
    lo, hi = sorted((round(start), round(end)))
    rows = np.arange(max(0, lo), min(gray.shape[0], hi + 1))
    x0, _, x1, _ = head.box
    columns = np.linspace(max(0, x0 - .15 * staff.space), min(gray.shape[1] - 1, x1 + .15 * staff.space), 9)
    ink = map_coordinates(gray, [rows[:, None] + staff.slope * (columns - head.cx),
                                 np.broadcast_to(columns, (len(rows), len(columns)))],
                          order=1, mode="nearest", output=np.float32)
    profile = -np.quantile(ink, .65, axis=1)
    peaks, _ = find_peaks(profile, prominence=40, distance=max(2, round(.45 * staff.space)))
    through_head = np.count_nonzero(side * (rows[peaks] - head.cy) <= .25 * staff.space)
    # Perspective and raster rounding can squeeze ten ledger lines into the
    # apparent distance of nine. Their ink still places the head at the limit.
    return through_head >= (POS_MAX - 4) // 2


def locate(gray, head, top: float, bottom: float, space: float, clefs: list):
    """Return the local position and geometry only when the staff can be traced to its clef.

    A clear reference staff is preferred for ledger notes. On a skewed page
    the measure-box band can cross another staff or belong to a facing page.
    """
    x, y = head.cx, head.cy
    reference = (top + bottom) / 2
    if gray is None:
        return None
    # A tilted measure box sometimes contains several spaces of blank paper.
    scale = min(space, head.box[3] - head.box[1])
    detected = _detected_position(head)
    radius = 21 if abs(detected) >= 22 else 15
    found = _staff_lines(gray, x, y, max(2.0, scale), radius * space)
    if not found:
        return None
    closest = min(found, key=lambda b: abs(b.middle - y) / b.space - b.strength / 110)
    assigned = [b for b in found if abs(b.middle - reference) < 1.25 * b.space
                and abs(b.position(y)) <= POS_MAX]
    if assigned and abs(closest.slope) < .10:
        closest = min(assigned, key=lambda b: abs(b.middle - reference))
    ledger = _ledger_staff(gray, head, found, detected)
    if ledger is not None:
        closest = ledger
    pos = round(closest.position(y))
    if abs(pos) >= POS_MAX:
        return None
    local = sorted((c for c in clefs if c.cx <= x), key=lambda c: -c.cx)
    clef = _traced_clef(gray, x, closest, local)
    if clef is None or abs(pos) > 4 and _ledger_reaches_limit(gray, head, closest):
        return None
    return pos, clef, closest.space, closest.slope
