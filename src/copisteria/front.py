# Modified by Play Anything (https://github.com/playanything-org/copisteria): see README.md.
"""The minimal front end: detector boxes -> symbols -> staves, bars and systems. No music in here.

Only geometry is decided here, and only the parts the writer cannot do without:

* **merge** -- boxes of one family that overlap (IoU >= MERGE_IOU) are one symbol; its class posterior is the
  boxes' confidence mass per class, its P(real) their noisy-or. Different families never merge: two readings of
  one glyph in different families stay two symbols, and the model decides which one is real.
* **staves** -- from the staff lines in the page's ink (five long thin runs at one spacing), checked against the
  staff space of the detector's ``measure`` boxes; a row of measure boxes no ink staff explains is a staff too. A
  scan's frame (the scanner bed around the sheet) is painted out first: it is not ink.
  Without the page image (or when the line finder fails on it), a staff is a chain of side-by-side, vertically
  overlapping measure boxes, and a gap between two of them that holds notes or rests becomes a bar of its own.
* **systems** -- two consecutive staves share a system when an ink run crosses the gap between them at the
  system's left (its opening line, brace or bracket) and are separate where that gap is white; in between, a group
  symbol spanning them or the ``links`` model decides.
* **bars** -- a system's bar lines are decided once for all its staves: where most staves show a bar line (a
  detection and a vertical stroke over the staff's full height, or a stroke through every staff); slivers and
  empty edge stretches fold away.
* **assignment** -- every other symbol goes to the staff it is nearest to (a note to the staff where its own
  staff-position reading fits), and to the bar its centre falls in.

Everything musical -- which readings are real, durations, pitches, voices, keys, meters -- is the model's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .vocab import CLASSES, CLS_ID, family

MERGE_IOU = 0.5
MEASURE_MIN_P = 0.25          # measure boxes below this do not build staves
MEASURE_MIN_W = 2.0           # nor boxes narrower than this many staff spaces
MEASURE_LOW_P = 0.08          # ... except where no confident staff is: then down to this
GRP_MIN_P = 0.3
ALIGN_TOL_SP = 2.0            # bar lines this close (in staff spaces) may be partners
ALIGN_FRAC = 0.7
ASSIGN_MAX_SP = 10.0          # farther than this from every staff: not part of the score
EMPTY_BAR_MAX_SP = 4.0        # an interior stretch narrower than this with no note or rest is a bar-line group


Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@dataclass
class Sym:
    i: int
    cls: str                       # the posterior's argmax
    post: dict                     # class id -> probability (sums to 1 over the classes the boxes read)
    p: float                       # P(real): noisy-or of the merged boxes' confidences
    n: int                         # boxes merged
    box: Box
    attrs: dict = field(default_factory=dict)
    attr_p: dict = field(default_factory=dict)
    attr_dist: dict = field(default_factory=dict)
    staff: int = -1                # page-global staff index
    bar: int = -1                  # bar index within its staff
    synthetic: bool = False        # a bar the front end made from a gap (no detector box)
    det: int = -1                  # the detection it was made from (its most confident box)
    cand: bool = False             # a sub-threshold candidate of the detector's

    @property
    def fam(self) -> str:
        return family(self.cls)

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    def to_dict(self) -> dict:
        return {"i": self.i, "cls": self.cls, "post": {str(k): round(v, 4) for k, v in self.post.items()},
                "p": round(self.p, 4), "n": self.n, "box": [round(v, 1) for v in self.box], "attrs": self.attrs,
                "attr_p": self.attr_p, "attr_dist": self.attr_dist, "staff": self.staff, "bar": self.bar,
                "synthetic": self.synthetic, "cand": self.cand}


@dataclass
class Staff:
    bars: list[int]                # symbol index of each bar's measure box, x order
    sp: float                      # staff space (px)
    y0: float
    y1: float
    x0: float
    x1: float
    system: int = -1
    pos: int = -1                  # index within its system (0 = top)
    line_x0: float | None = None   # where its lines start with near-white taken as paper (_line_starts), if found
    one_line: bool = False         # a one-line staff, taken on the percussion clef that stands on it (_one_line_staves)
    lined: bool = False            # found by its five lines in a gap between two staves (_gap_staves)

    @property
    def mid(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class System:
    staves: list[int]
    columns: list[list[int]] = field(default_factory=list)   # per staff (system order): bar index -> column


@dataclass
class Layout:
    syms: list[Sym]
    staves: list[Staff]
    systems: list[System]
    sp: float
    width: float
    height: float
    ink: bool = False                # staves and bars from the page's ink (else from the measure boxes)
    prior: dict | None = None        # what a caller's layout did to the reading (_with_layout), if one was given

    def bar_box(self, staff: int, bar: int) -> Box:
        return self.syms[self.staves[staff].bars[bar]].box

    def column(self, staff: int, bar: int) -> int:
        st = self.staves[staff]
        return self.systems[st.system].columns[st.pos][bar]


# ------------------------------------------------------------------------------------------------- merge
def merge(dets: list[dict], offset: int = 0) -> list[Sym]:
    """Boxes of one family with IoU >= MERGE_IOU -> one symbol, greedy from the most confident box. A symbol's
    ``det`` is its head box's index in ``dets`` plus ``offset``."""
    order = sorted(range(len(dets)), key=lambda k: -float(dets[k]["conf"]))
    groups: list[list[int]] = []
    heads: list[tuple[str, Box]] = []
    for k in order:
        d = dets[k]
        if d["cls"] not in CLS_ID:
            continue
        fam, box = family(d["cls"]), tuple(float(v) for v in d["xyxy"])
        for g, (gf, gb) in enumerate(heads):
            if gf == fam and iou(gb, box) >= MERGE_IOU:
                groups[g].append(k)
                break
        else:
            groups.append([k]); heads.append((fam, box))
    syms = []
    for g in groups:
        head = dets[g[0]]
        mass: dict[int, float] = {}
        miss = 1.0
        for k in g:
            c = float(dets[k]["conf"])
            mass[CLS_ID[dets[k]["cls"]]] = mass.get(CLS_ID[dets[k]["cls"]], 0.0) + c
            miss *= 1.0 - min(c, 0.999)
        tot = sum(mass.values())
        post = {c: m / tot for c, m in mass.items()}
        best = max(post, key=post.get)
        syms.append(Sym(i=len(syms), cls=CLASSES[best], post=post, p=1.0 - miss, n=len(g), box=tuple(float(v) for v in head["xyxy"]),
                        attrs=dict(head.get("attrs") or {}), attr_p=dict(head.get("attr_p") or {}),
                        attr_dist=dict(head.get("attr_dist") or {}), det=g[0] + offset))
    return syms


# ------------------------------------------------------------------------------------------------- staves
def _staves(syms: list[Sym]) -> list[Staff]:
    """Confident measure boxes make staves; where none sits, weak ones (down to MEASURE_LOW_P) may."""
    def wide(s):     # a bar is at least MEASURE_MIN_W staff spaces wide (its own height is four)
        return s.box[2] - s.box[0] >= MEASURE_MIN_W * (s.box[3] - s.box[1]) / 4

    staves = _rows([s for s in syms if s.fam == "measure" and s.p >= MEASURE_MIN_P and wide(s)])
    # weak boxes stay clear of the confident staves: staves sit at least ~6 staff spaces apart, centre to centre
    weak = [s for s in syms if s.fam == "measure" and MEASURE_LOW_P <= s.p < MEASURE_MIN_P and wide(s) and
            not any(abs(s.cy - st.mid) < 5 * st.sp for st in staves)]
    for st in _rows(weak):          # a weak staff must hold music of its own
        held = sum(1 for s in syms if s.fam in ("note", "rest", "clef") and s.p >= 0.3 and st.x0 <= s.cx <= st.x1
                   and st.y0 - 0.5 * st.sp <= s.cy <= st.y1 + 0.5 * st.sp)
        if held >= 3:
            staves.append(st)
    return _clear_rows(staves, syms)


def _two_staves(a: Staff, b: Staff, syms: list[Sym]) -> bool:
    """Whether two rows of measure boxes that overlap are two printed staves all the same: each has notes of its own.
    The detector says where on its staff a note stands (staff_position); a note is a row's own where it stands there
    to within a step and a half by that row's place, and not by the other's. Of one staff boxed twice, the notes
    are one row's (or neither's); of two staves printed close, whose boxes reach into each other, each has its own,
    one over the other: along the same stretch of the page. (A staff that bows or tilts across a photograph is boxed
    higher at one end than the other, and each row has its own notes there too, side by side.) Where the notes do
    not say (fewer than three a row), the rows are not shown to be two."""
    own: list[list[float]] = [[], []]
    low, high = min(a.y0, b.y0) - 2 * max(a.sp, b.sp), max(a.y1, b.y1) + 2 * max(a.sp, b.sp)
    for s in syms:
        if s.fam != "note" or s.p < 0.3 or not (min(a.x0, b.x0) <= s.cx <= max(a.x1, b.x1) and low <= s.cy <= high):
            continue
        try:
            pos = int(s.attrs.get("staff_position"))
        except (TypeError, ValueError):
            continue
        if abs(pos) >= 20 or float(s.attr_p.get("staff_position", 1.0)) < 0.5:
            continue
        off = [abs(((st.y0 + st.y1) / 2 - s.cy) / (st.sp / 2) - pos) for st in (a, b)]
        for k in (0, 1):
            if off[k] <= 1.5 < off[1 - k]:
                own[k].append(s.cx)
    if min(len(xs) for xs in own) < max(3, 0.25 * sum(len(xs) for xs in own)):
        return False
    over = min(max(own[0]), max(own[1])) - max(min(own[0]), min(own[1]))
    return over >= 0.5 * min(max(xs) - min(xs) for xs in own)


def _clear_rows(staves: list[Staff], syms: list[Sym]) -> list[Staff]:
    """The staves the measure boxes make, without those that stand on another. No two staves of a page overlap, so
    two that share more than DOUBLE_OVERLAP of the lower one's height are one printed staff read twice: on a
    photograph the detector boxes the bars of a tilted or blurred staff at two heights, part of a staff's height
    apart, and each set makes a row (_rows joins boxes that share half their height). The row its boxes are surer of
    stays, then the one that holds more of the page's music, and takes the other's bars where it has none of its
    own: they are that staff's bars, and what stands in them is found by their boxes (_band). Two rows each with
    notes of its own are two staves printed close, boxed loosely, and both stay (_two_staves). A page read from its
    staff lines has the same rule there (_without_doubles)."""
    def weight(st: Staff) -> tuple[float, int]:
        held = sum(1 for s in syms if s.fam in ("note", "rest", "clef") and s.p >= 0.3 and st.x0 <= s.cx <= st.x1
                   and st.y0 - 0.5 * st.sp <= s.cy <= st.y1 + 0.5 * st.sp)
        return sum(syms[i].p for i in st.bars), held

    def shared(st: Staff, o: Staff) -> float:
        return (min(st.y1, o.y1) - max(st.y0, o.y0)) / max(1.0, min(st.y1 - st.y0, o.y1 - o.y0))

    kept: list[Staff] = []
    for st in sorted(staves, key=weight, reverse=True):
        # the staff it shares most with: a row between two staves is the nearer one's
        on = max((o for o in kept if shared(st, o) > DOUBLE_OVERLAP), key=lambda o: shared(st, o), default=None)
        if on is None or _two_staves(st, on, syms):
            kept.append(st)
            continue
        bars = [syms[i] for i in on.bars]
        for i in st.bars:
            box = syms[i].box
            if all(min(box[2], q.box[2]) - max(box[0], q.box[0]) <= 0.5 * min(box[2] - box[0], q.box[2] - q.box[0])
                   for q in bars):
                bars.append(syms[i])
        bars.sort(key=lambda s: s.cx)
        on.bars, on.x0, on.x1 = [s.i for s in bars], bars[0].box[0], bars[-1].box[2]
    kept.sort(key=lambda st: st.mid)
    return kept


def _rows(bars: list[Sym]) -> list[Staff]:
    if not bars:
        return []
    parent = list(range(len(bars)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(bars)):
        A = bars[a].box
        for b in range(a + 1, len(bars)):
            B = bars[b].box
            ov = min(A[3], B[3]) - max(A[1], B[1])
            if ov < 0.5 * min(A[3] - A[1], B[3] - B[1]):
                continue
            # side by side (a missed bar between them is fine), not stacked readings of one bar
            xov = min(A[2], B[2]) - max(A[0], B[0])
            if xov < 0.5 * min(A[2] - A[0], B[2] - B[0]):
                parent[find(a)] = find(b)
    rows: dict[int, list[Sym]] = {}
    for a, s in enumerate(bars):
        rows.setdefault(find(a), []).append(s)
    staves = []
    for members in rows.values():
        members.sort(key=lambda s: s.cx)
        kept: list[Sym] = []
        for s in members:          # two boxes over most of one bar: the likelier one is the bar
            if kept:
                q = kept[-1]
                ov = min(q.box[2], s.box[2]) - max(q.box[0], s.box[0])
                if ov > 0.5 * min(q.box[2] - q.box[0], s.box[2] - s.box[0]):
                    if s.p > q.p:
                        kept[-1] = s
                    continue
            kept.append(s)
        y0 = sorted(s.box[1] for s in kept)[len(kept) // 2]
        y1 = sorted(s.box[3] for s in kept)[len(kept) // 2]
        staves.append(Staff(bars=[s.i for s in kept], sp=max(1.0, (y1 - y0) / 4), y0=y0, y1=y1,
                            x0=kept[0].box[0], x1=kept[-1].box[2]))
    staves.sort(key=lambda st: st.mid)
    return staves


LOOSE = ("note", "rest", "multiRest")    # what a bar holds: outside every bar box, a bar the detector missed


def _fill_gaps(syms: list[Sym], staves: list[Staff]) -> None:
    """Stretches of a staff with notes, rests or a multi-bar rest outside every bar box are bars the detector missed:
    a gap between two bars, or the run before the first / after the last. Detected bar lines inside a stretch split
    it. A part's opening multi-bar rest, boxed only faintly, was otherwise no bar: the staff began after it, without
    its clef, and its rests were not counted."""
    _assign(syms, staves)
    for k, st in enumerate(staves):
        boxes = [syms[i].box for i in st.bars]
        loose = [s for s in syms if s.staff == k and s.fam in LOOSE and s.p >= 0.25 and
                 not any(b[0] - 0.5 * st.sp <= s.cx <= b[2] + 0.5 * st.sp for b in boxes)]
        if not loose:
            continue
        spans = []                                  # (x0, x1) stretches between bars
        edges = [(-math.inf, boxes[0][0])] + [(a[2], b[0]) for a, b in zip(boxes, boxes[1:])] + \
                [(boxes[-1][2], math.inf)]
        for x0, x1 in edges:
            inside = [s for s in loose if x0 <= s.cx <= x1]
            if not inside:
                continue
            if x0 == -math.inf:
                x0 = min(s.box[0] for s in inside) - 1.5 * st.sp
            if x1 == math.inf:
                x1 = max(s.box[2] for s in inside) + 1.5 * st.sp
            cuts = sorted(s.cx for s in syms if s.staff == k and s.fam == "barLine" and s.p >= 0.5
                          and x0 + st.sp < s.cx < x1 - st.sp)
            if x1 - x0 < 1.5 * st.sp:                 # a sliver between two boxes: its notes join a neighbour
                continue
            xs = [x0, *cuts, x1]
            spans += [(a, b) for a, b in zip(xs, xs[1:]) if any(a <= s.cx <= b for s in inside)]
        new = []
        for x0, x1 in spans:
            _, y0, y1, _ = _band(syms, st, (x0 + x1) / 2)
            s = Sym(i=len(syms), cls="measure", post={CLS_ID["measure"]: 1.0}, p=0.0, n=0,
                    box=(x0, y0, x1, y1), synthetic=True)
            syms.append(s); new.append(s.i)
        if new:
            st.bars = sorted(st.bars + new, key=lambda i: syms[i].cx)
            st.x0, st.x1 = min(st.x0, syms[st.bars[0]].box[0]), max(st.x1, syms[st.bars[-1]].box[2])
    for s in syms:                                  # assigned again once the systems exist
        s.staff, s.bar = -1, -1


def _snap_bars(syms: list[Sym], staves: list[Staff]) -> None:
    """The boundary between two neighbouring bars sits on the bar line the detector saw there (measure boxes
    are loose at their edges); without one, overlapping boxes meet halfway."""
    lines = [s for s in syms if s.fam == "barLine" and s.p >= 0.4]
    for st in staves:
        for a, b in zip(st.bars, st.bars[1:]):
            A, B = syms[a], syms[b]
            lo, hi = min(A.box[2], B.box[0]) - 4 * st.sp, max(A.box[2], B.box[0]) + 4 * st.sp
            mid = (A.box[2] + B.box[0]) / 2
            near = [s for s in lines if lo <= s.cx <= hi and st.y0 - st.sp <= s.cy <= st.y1 + st.sp]
            if near:
                x = min(near, key=lambda s: abs(s.cx - mid)).cx
            elif A.box[2] > B.box[0]:
                x = mid
            else:
                continue
            if A.box[0] < x < B.box[2]:
                A.box = (A.box[0], A.box[1], x, A.box[3])
                B.box = (x, B.box[1], B.box[2], B.box[3])
        # a filled-in bar the snapping squeezed to a sliver is no bar: the next one takes its stretch
        keep = []
        for k, i in enumerate(st.bars):
            s = syms[i]
            if s.synthetic and s.box[2] - s.box[0] < 1.5 * st.sp and k + 1 < len(st.bars):
                n = syms[st.bars[k + 1]]
                n.box = (s.box[0], n.box[1], n.box[2], n.box[3])
                continue
            keep.append(i)
        st.bars = keep


def _aligned(syms: list[Sym], a: Staff, b: Staff) -> bool:
    """Bar lines of the two staves line up: over their common stretch, at least ALIGN_FRAC of each staff's
    detected internal bar lines have a partner on the other within ALIGN_TOL_SP, and the partners sit at one
    consistent offset (a warped page shifts a whole system's staves together; two systems whose bars merely
    happen to fall near each other scatter)."""
    sp = (a.sp + b.sp) / 2
    lo, hi = max(a.x0, b.x0) - sp, min(a.x1, b.x1) + sp

    def edges(st):
        return [syms[k].box[2] for k in st.bars[:-1] if not syms[k].synthetic and lo <= syms[k].box[2] <= hi]

    ea, eb = edges(a), edges(b)
    if not ea or not eb:
        return False
    tol = ALIGN_TOL_SP * sp
    diffs = [min(eb, key=lambda y: abs(x - y)) - x for x in ea]
    diffs = [d for d in diffs if abs(d) <= tol]
    hit2 = sum(1 for y in eb if any(abs(x - y) <= tol for x in ea))
    if len(diffs) < ALIGN_FRAC * len(ea) or hit2 < ALIGN_FRAC * len(eb):
        return False
    if len(diffs) == 1:
        return abs(diffs[0]) <= 0.5 * sp
    med = sorted(diffs)[len(diffs) // 2]
    return max(abs(d - med) for d in diffs) <= 0.6 * sp


def _braced(syms: list[Sym], a: Staff, b: Staff) -> bool:
    for s in syms:
        if s.fam != "grpSym" or s.p < GRP_MIN_P:
            continue
        x0, y0, x1, y1 = s.box
        if y0 <= a.mid + a.sp and y1 >= b.mid - b.sp and x1 >= min(a.x0, b.x0) - 6 * a.sp and \
                x0 <= min(a.x0, b.x0) + 3 * a.sp:
            return True
    return False


INK_LINK = 0.85               # share of the gap a vertical ink run must cover to join two staves
INK_SPLIT = 0.4               # below this (and no bar line through the gap): two systems
STAFF_PAPER_TOL = 0.04        # near-white this close to the paper (of the way to the ink) is paper where a staff's
                              # lines are taken to start (_line_starts): a JPEG's noise beside a brace is not a line
FRAME_PAPER = 0.5             # a row (column) is the sheet's once it holds half the paper of the page's middle lines
FRAME_MAX = 0.35              # a frame reaches at most this far in from its side (past it, that side has none)
FRAME_SLIVER = 0.01           # a white sliver this thin may run along the image's edge outside a frame (export padding)
FRAME_MARGIN = 0.005          # a frame's inner edge is ragged: the sheet is taken to start this far past it (its margin)


def _dark(gray):
    import numpy as np
    paper, ink = np.percentile(gray, 50), np.percentile(gray, 1)
    return gray < paper - 0.35 * max(1.0, paper - ink)


def _sheet_edges(paper) -> tuple[int, int]:
    """Where the sheet starts and ends along one dimension, from its paper profile (each line's share of paper): a
    frame is the run of lines from a side whose paper falls short of FRAME_PAPER of the middle lines'."""
    import numpy as np
    n = len(paper)
    edge = FRAME_PAPER * float(np.median(paper[n // 4: 3 * n // 4]))
    sliver = max(2, int(FRAME_SLIVER * n))
    margin = max(2, round(FRAME_MARGIN * n))

    def inward(line) -> int:
        k = 0
        while k < sliver and k < n and line[k] >= edge:
            k += 1
        if k == n or line[k] >= edge:
            return 0                                     # paper from the edge in: no frame on this side
        while k < n and line[k] < edge:
            k += 1
        return k + margin if k <= FRAME_MAX * n else 0
    a, b = inward(paper), n - inward(paper[::-1])
    return (a, b) if b > a else (0, n)


def _frame(dark) -> tuple[int, int, int, int]:
    """The sheet inside a scan's frame -- the scanner bed or the dark backing around the paper, a band of ink
    along the image's edges -- as (x0, y0, x1, y1); the whole image when there is none. Each dimension's paper
    profile is near nothing across a frame and high on the sheet, however dense its music; the profiles are taken
    twice, the second time over the sheet the first found, so one side's band does not count against the others'
    lines."""
    H, W = dark.shape
    x0, y0, x1, y1 = 0, 0, W, H
    for _ in range(2):
        cols = 1.0 - dark[y0:y1].mean(axis=0)
        rows = 1.0 - dark[:, x0:x1].mean(axis=1)
        (x0, x1), (y0, y1) = _sheet_edges(cols), _sheet_edges(rows)
    return x0, y0, x1, y1


def unframe(gray):
    """The page with a scan's frame painted the paper's tone. The frame is not ink: left in, it carries staff
    lines out to the image's edge and reads as one line through every gap between staves at the page's left."""
    import numpy as np
    x0, y0, x1, y1 = _frame(_dark(gray))
    H, W = gray.shape
    if (x0, y0, x1, y1) == (0, 0, W, H) or x1 <= x0 or y1 <= y0:
        return gray
    out = np.full_like(gray, np.median(gray[y0:y1, x0:x1]))
    out[y0:y1, x0:x1] = gray[y0:y1, x0:x1]
    return out


def ink_mask(gray):
    """Ink pixels of a page: darker than 35 % of the way from its paper (median) to its ink (1st percentile), so a
    thin anti-aliased line on a small render counts as well as a thick stroke on a 1-bit scan. A scan's frame is
    not ink (unframe)."""
    return _dark(unframe(gray))


SHADED = 0.12                 # a page is shaded once its paper's tone varies by this share of it (_shading)


def _shading(gray):
    """(the page's paper with the ink filled in, at the page's size; how much its tone varies over the page: the
    spread between its 5th and 95th percentiles as a share of its median). A render or a flat scan varies by a few
    hundredths; a photograph under a lamp, or a book's page curving into its gutter, by several tenths."""
    import numpy as np
    from scipy import ndimage
    g = gray.astype(np.float32)
    k = max(15, int(min(g.shape) / 30)) | 1
    small = g[::4, ::4]
    bg = ndimage.grey_closing(small, size=(max(3, k // 4), max(3, k // 4)))      # the paper, ink filled in
    bg = ndimage.uniform_filter(bg, size=max(3, k // 4))
    lo, mid, hi = (float(v) for v in np.percentile(bg, (5, 50, 95)))
    return np.kron(bg, np.ones((4, 4), np.float32))[:g.shape[0], :g.shape[1]], (hi - lo) / max(1.0, mid)


def flat_page(gray):
    """The page with its paper's large-scale shading (a vignette, a shadow, a tinted or textured sheet) divided
    out: the paper white all over, the ink as dark against it as it was."""
    import numpy as np
    bg, _ = _shading(gray)
    return np.clip(gray.astype(np.float32) / np.maximum(bg, 1.0) * 255.0, 0, 255)


def page_mask(gray):
    """The page's ink for what is read off it between and across the staves (the line that opens a system, the bar
    lines' strokes). ink_mask sets one threshold for the whole page, from its median paper: under a photograph's
    uneven light the paper in a shadow falls below it and is ink all over, and a thin line in a bright patch rises
    above it and is paper. So a shaded page (SHADED) is masked with its shading divided out; any other page as it
    is, by the same threshold as before."""
    bg, spread = _shading(gray)
    if spread < SHADED:
        return ink_mask(gray)
    import numpy as np
    return ink_mask(np.clip(gray.astype(np.float32) / np.maximum(bg, 1.0) * 255.0, 0, 255))


def clean_page(gray):
    """The page as black ink on white for the staff-line finder: the paper's large-scale shading (a vignette, a
    shadow, a tinted or textured sheet) divided out, then the page-relative ink threshold (ink_mask). The line
    finder's own threshold takes anything short of pure white for ink, which on a scan's paper is everything."""
    import numpy as np
    return np.where(ink_mask(flat_page(gray)), 0, 255).astype(np.uint8)


def _ink_run(mask, x: float, y0: float, y1: float, reach: int = 2) -> float:
    """Share of the rows between y0 and y1 that hold ink within ``reach`` px of column x."""
    H, W = mask.shape
    a, b = max(0, int(round(y0))), min(H, int(round(y1)))
    xa, xb = max(0, int(round(x)) - reach), min(W, int(round(x)) + reach + 1)
    if b - a < 2 or xb <= xa:
        return 1.0 if b - a < 2 else 0.0
    return float(mask[a:b, xa:xb].any(axis=1).mean())


def _left_span(a: Staff, b: Staff, page_left: float, lines: bool = False) -> tuple[int, int]:
    """The columns where the line, brace or bracket that opens a system stands, between staves a and b: from the
    page's leftmost staff start (less 3 staff spaces) to a staff space past the earlier of their starts. ``lines``:
    the starts of their lines with near-white taken as paper (line_x0) where found, past a brace that the noise
    beside it joined to the staff (_line_starts)."""
    sp = (a.sp + b.sp) / 2
    start = (lambda st: max(st.x0, st.line_x0) if lines and st.line_x0 is not None else st.x0)    # noqa: E731
    return int(min(page_left, a.x0, b.x0) - 3 * sp), int(min(start(a), start(b)) + sp)


def _ink_links(gray, syms: list[Sym], a: Staff, b: Staff, page_left: float) -> tuple[float, float, float]:
    """(left, bars, margin): how much of the gap between two staves a vertical ink run crosses at their left end (the
    system's opening line, brace or bracket -- the measure boxes start after the clef, so from the page's leftmost
    staff start on), the share of a's detected bar lines whose ink runs on into b, and how much of the gap ink crosses
    there that is no such stroke (_meets_staves): the edge of a binder, a folder or the sheet, a band of shadow."""
    sp = (a.sp + b.sp) / 2
    y0, y1 = a.y1 + 1, b.y0 - 1
    lo, hi = _left_span(a, b, page_left)
    left = margin = 0.0
    for x in range(max(0, lo), hi + 1, 2):
        run = _ink_run(gray, x, y0, y1)
        # a run that would say something (INK_SPLIT and up) counts only where the staves' lines begin at it
        if run < INK_SPLIT or run <= left or _meets_staves(gray, x, a, b):
            left = max(left, run)
        else:
            margin = max(margin, run)
    lines = [s for s in syms if s.fam == "barLine" and s.p >= 0.5 and s.staff < 0 and
             a.y0 - sp <= s.cy <= a.y1 + sp and max(a.x0, b.x0) <= s.cx <= min(a.x1, b.x1)]
    bars = (sum(_ink_run(gray, s.cx, y0, y1) >= INK_LINK for s in lines) / len(lines)) if lines else 0.0
    return left, bars, margin


MEETS_REACH = 3.0             # staff spaces past a stroke within which its staves' lines begin, if it opens their system
MEETS_WIDE = 1.5              # a stroke at most this many staff spaces wide across the gap (a brace's, a bracket's)
MEETS_ROW = 0.5               # a row is a staff line's where it is ink over this share of those columns
MEETS_LINES = 3               # a staff (of five lines) shows this many of its lines there; a one-line staff its one
MEETS_OFF = 2.0               # staff spaces a staff's lines may lie off its box at the page's left (a tilted page)


def _line_rows(mask, r0: int, r1: int, c0: int, c1: int) -> int:
    """How many separate horizontal lines cross columns c0..c1 between rows r0 and r1: runs of rows that hold ink over
    MEETS_ROW of those columns. Words, a clef, a band of ink across them make few such rows, or one run of them."""
    rows = mask[r0:r1, c0:c1].mean(axis=1) >= MEETS_ROW
    return int(rows[0]) + int((rows[1:] & ~rows[:-1]).sum()) if len(rows) else 0


def _meets_staves(mask, x: int, a: Staff, b: Staff) -> bool:
    """Whether the vertical ink at column x, between staves a and b, is a stroke their lines begin at: no wider than
    a brace (MEETS_WIDE), and within MEETS_REACH staff spaces past it each staff's lines run on to the right. A
    system's opening line, bracket or brace stands at its staves' start; the edge of the sheet, of a binder or a
    folder the page lies in, or a shadow stands in the margin, as a dark band or with paper between it and the
    staves, and in a photograph taken by hand it runs past every gap of the page. Taken for a system's line it
    joined every staff of a part into one system of many staves."""
    H, W = mask.shape
    sp = (a.sp + b.sp) / 2
    y0, y1 = a.y1 + 1, b.y0 - 1
    inked = lambda c: _ink_run(mask, c, y0, y1, reach=0) >= 0.5                    # noqa: E731
    # x was found with a reach of two columns: the stroke is the run of inked columns nearest it
    near = [c for c in range(max(0, x - 2), min(W, x + 3)) if inked(c)]
    if near:
        left = right = near[0]
        while left > 0 and inked(left - 1):
            left -= 1
            if right - left + 1 > MEETS_WIDE * sp:
                return False                          # a dark band (the background past the sheet), not a stroke
        while right + 1 < W and inked(right + 1):
            right += 1
            if right - left + 1 > MEETS_WIDE * sp:
                return False
    else:
        right = min(W - 1, x + 2)
    c0, c1 = right + 1, min(W, int(round(right + 1 + MEETS_REACH * sp)))
    if c1 - c0 < 2:
        return False
    for st in (a, b):
        # a staff found from its measure boxes is placed roughly: on a tilted page its lines at the left lie a staff
        # space or two off its box; a one-line staff's rows are two staff spaces either side of its line
        reach = (0.5 if st.one_line else 2.0) * st.sp + (0.0 if st.line_x0 is not None else MEETS_OFF * st.sp)
        r0, r1 = max(0, int(round(st.mid - reach))), min(H, int(round(st.mid + reach)) + 1)
        if _line_rows(mask, r0, r1, c0, c1) < (1 if st.one_line else MEETS_LINES):
            return False
    return True


PALE = 0.12                   # a pale line is at least this far from the paper toward the ink (_pale_line)
PALE_BACK = 0.5               # and beside it the page is this much of its depth nearer its paper, or as much darker


def _pale_line(levels, paper: float, dark: float, a: Staff, b: Staff, page_left: float) -> float:
    """Share of the gap between two staves that a pale vertical line crosses where a system opens (_pale_at)."""
    return _pale_at(levels, paper, dark, a, b, page_left)[0]


def _pale_at(levels, paper: float, dark: float, a: Staff, b: Staff, page_left: float,
             staves: list[Staff] | None = None) -> tuple[float, int]:
    """(share, x): the share of the gap between two staves that a pale vertical line crosses where a system opens
    (_left_span, up to the staves' line starts), and the line's column. ``staves``: the page's, where a line that
    goes on past all of them is not to count (_runs_on). On a small render the opening line is one
    anti-aliased pixel, grey 200-210 on white, which the ink mask does not count; only a line counts here, with
    paper a staff space to either side and the page changing within a quarter of one, so a grey patch (a stamp, its
    edge, a shadow, a shadow's edge) is not one. ``paper``, ``dark``: the page's median and 1st-percentile levels.
    After issue #1's suggested _faint_line."""
    H, W = levels.shape
    ya, yb = max(0, int(round(a.y1 + 1))), min(H, int(round(b.y0 - 1)))
    if yb - ya < 2:
        return 0.0, -1
    lo, hi = _left_span(a, b, page_left, lines=True)
    pale = levels[ya:yb] < paper - PALE * max(1.0, paper - dark)
    away = max(3, int(round((a.sp + b.sp) / 2)))
    # a line is narrow: within a quarter of a staff space to either side the page is half way back to the paper it
    # has a staff space away (the lighter side's: a shaded sheet is darker than the page's paper on both, a shadow's
    # edge on one), or runs into darker ink (the line itself, where this is its pale shoulder; a bracket beside it).
    # Beside a shadow's edge the page stays as pale as the edge, neither paper nor ink
    near = max(2, int(round(away / 4)))
    tone = levels[ya:yb].mean(axis=0)
    best, at = 0.0, -1
    for x in range(max(away + 1, lo), min(W - away - 1, hi) + 1):
        run = float(pale[:, x - 1:x + 2].any(axis=1).mean())
        if run > best and all(float(pale[:, c - 1:c + 1].any(axis=1).mean()) < 0.5 for c in (x - away, x + away)):
            core = float(tone[x - 1:x + 2].min())
            step = PALE_BACK * (min(paper, max(float(tone[x - away]), float(tone[x + away]))) - core)
            sides = (tone[x - near - 2:x - 1], tone[x + 2:x + near + 3])
            if step > 0 and all(float(side.max()) >= core + step or float(side.min()) <= core - step for side in sides) \
                    and not (staves and _runs_on(levels, paper, dark, x, staves)):
                best, at = run, x
    return best, at


RUNS_ON = (1.5, 4.0)          # staff spaces past the page's outer staves where a system's opening line is not (_runs_on)
RUNS_ON_SHARE = 0.5           # of those rows pale: a line that is the page's. A system's own line has none of them


def _runs_on(levels, paper: float, dark: float, x: int, staves: list[Staff]) -> bool:
    """Whether the pale line at column x goes on past the page's first staff and past its last: a system's opening
    line ends with its system (a bracket's hook reaches a staff space further), so a line that is there above every
    staff of the page and below every staff is the page's, not a system's -- the edge of a binding's shadow, of the
    sheet, of a frame taken off (unframe), which is pale for the page's whole height. A page whose staves leave no
    margin to look at on a side says nothing there, and then nothing is said."""
    H, W = levels.shape
    top, bottom = min(staves, key=lambda st: st.y0), max(staves, key=lambda st: st.y1)
    near, far = RUNS_ON
    bands = ((int(round(top.y0 - far * top.sp)), int(round(top.y0 - near * top.sp))),
             (int(round(bottom.y1 + near * bottom.sp)), int(round(bottom.y1 + far * bottom.sp))))
    limit = paper - PALE * max(1.0, paper - dark)
    for y0, y1 in bands:
        if y0 < 0 or y1 > H or y1 - y0 < 2:
            return False
        if float((levels[y0:y1, max(0, x - 1):x + 2] < limit).any(axis=1).mean()) < RUNS_ON_SHARE:
            return False
    return True


def _opening_line(mask, a: Staff, b: Staff, page_left: float) -> float:
    """How much of the gap between two staves an ink run crosses in the columns their line starts add to the
    search (_left_span with ``lines``): the opening line, where a brace that touches a staff carried the staff's
    start out onto the brace and the search stopped short of it."""
    _, hi = _left_span(a, b, page_left)
    _, hi2 = _left_span(a, b, page_left, lines=True)
    return max((_ink_run(mask, x, a.y1 + 1, b.y0 - 1) for x in range(hi + 1, hi2 + 1, 2)), default=0.0)


def _links(syms: list[Sym], staves: list[Staff], gray=None, levels=None) -> tuple[list[bool], list[str]]:
    """Per pair of neighbouring staves (k - 1, k): whether one system holds both, and what said so -- "line" (ink
    through the gap between them), "white" (none) or "unclear" (the links model, a brace, bar lines that line up).
    ``gray``: the page's ink mask; ``levels``: its grey levels (for a line too pale for the mask)."""
    from . import links
    n = len(staves)
    if n < 2:
        return [], []
    w = links.load()
    if w is not None:
        gaps = sorted((staves[k + 1].y0 - staves[k].y1) / staves[k].sp for k in range(n - 1))
        gm = gaps[len(gaps) // 2]
        ps = [links.prob(w, links.pair_features(syms, staves[k - 1], staves[k], gm, n,
                                                _braced(syms, staves[k - 1], staves[k])))
              for k in range(1, n)]
        decided = list(links.decode(ps))
    else:
        decided = [_braced(syms, staves[k - 1], staves[k]) or _aligned(syms, staves[k - 1], staves[k])
                   for k in range(1, n)]
    said = ["unclear"] * (n - 1)
    if gray is not None:
        # the ink decides where it is clear: a line through the gap joins -- looked for up to where the staves' lines
        # start (past a brace the paper's noise joined to them), and in the grey levels where the mask sees none (a
        # small render's one-pixel opening line is too pale for it) --, a white gap separates; a brace or bracket
        # the detector saw over both staves joins them whatever the ink
        import numpy as np
        page_left = min(st.x0 for st in staves)
        tone: list = []                                # the page's paper and dark levels, once needed
        for k in range(1, n):
            a, b = staves[k - 1], staves[k]
            left, bars, margin = _ink_links(gray, syms, a, b, page_left)
            line = 0.0
            if left < INK_LINK and bars < 0.5:
                line = _opening_line(gray, a, b, page_left)
                if line < INK_LINK and levels is not None:
                    tone = tone or [float(np.percentile(levels, 50)), float(np.percentile(levels, 1))]
                    line = max(line, _pale_at(levels, *tone, a, b, page_left, staves)[0])
            if left >= INK_LINK or line >= INK_LINK or bars >= 0.5:
                decided[k - 1], said[k - 1] = True, "line"
            elif left < INK_SPLIT and bars < 0.1 and margin >= INK_SPLIT and _braced(syms, a, b):
                decided[k - 1] = True    # a brace or bracket the detector saw, its stroke not told from the margin's ink
            elif left < INK_SPLIT and bars < 0.1:
                decided[k - 1], said[k - 1] = False, "white"     # white between them: a brace box claiming otherwise is wrong
            elif _braced(syms, a, b):
                decided[k - 1] = True
    return [bool(v) for v in decided], said


def _systems(syms: list[Sym], staves: list[Staff], gray=None, ink: bool = False, levels=None,
             linked: list[bool] | None = None) -> list[System]:
    """``gray``: the page's ink mask; ``levels``: its grey levels. ``linked``: per pair of neighbouring staves
    whether one system holds both, where the caller has decided it (else _links)."""
    if linked is None:
        linked, _ = _links(syms, staves, gray, levels)
    systems: list[System] = []
    for k in range(len(staves)):
        if k and linked[k - 1]:
            systems[-1].staves.append(k)
        else:
            systems.append(System(staves=[k]))
    for n, sy in enumerate(systems):
        for pos, k in enumerate(sy.staves):
            staves[k].system, staves[k].pos = n, pos
        ref = max(sy.staves, key=lambda k: (sum(not syms[i].synthetic for i in staves[k].bars),
                                            -sum(syms[i].synthetic for i in staves[k].bars)))
        if ink:
            _segment(syms, staves, sy, gray)
        elif gray is not None and len(sy.staves) > 1:
            _joint_bars(syms, staves, sy, gray)
        else:
            _reconcile(syms, staves, sy, ref)
        # bar columns: every bar maps to the reference staff's bar it overlaps most in x
        rb = [syms[i].box for i in staves[ref].bars]
        cols = []
        for k in sy.staves:
            row = []
            for i in staves[k].bars:
                b = syms[i].box
                ov = [min(b[2], r[2]) - max(b[0], r[0]) for r in rb]
                row.append(max(range(len(rb)), key=lambda j: ov[j]))
            cols.append(row)
        sy.columns = cols
    return systems


def _joint_bars(syms: list[Sym], staves: list[Staff], sy: System, gray) -> None:
    """A system's bar lines, decided once for all its staves: a candidate x (a measure box's edge or a detected
    bar line, on any staff) is a bar line where most staves show one there -- a detection, or a vertical ink
    stroke over the staff's whole height. Every staff then gets exactly those bars; a detected measure box that
    matches one keeps its identity, the others are filled in."""
    S = [staves[k] for k in sy.staves]
    sp = sum(st.sp for st in S) / len(S)
    left = min(syms[st.bars[0]].box[0] for st in S)
    right = max(syms[st.bars[-1]].box[2] for st in S)
    cand = []                                     # (x, staff index in S)
    for j, st in enumerate(S):
        for i in st.bars[:-1]:
            if not syms[i].synthetic:
                cand.append((syms[i].box[2], j))
        for b in syms:
            if b.fam == "barLine" and b.p >= 0.4 and st.y0 - sp <= b.cy <= st.y1 + sp:
                cand.append((b.cx, j))
    cand = sorted((x, j) for x, j in cand if left + 3 * sp < x < right - 1.5 * sp)
    clusters: list[list] = []
    for x, j in cand:
        if clusters and x - clusters[-1][-1][0] <= sp:
            clusters[-1].append((x, j))
        else:
            clusters.append([(x, j)])
    cuts = []
    for cl in clusters:
        x = sorted(v for v, _ in cl)[len(cl) // 2]
        seen = {j for _, j in cl}
        ev = 0
        for j, st in enumerate(S):
            if j in seen or _ink_run(gray, x, st.y0 + 1, st.y1 - 1, reach=1) >= 0.9:
                ev += 1
        if ev >= max(2, 0.75 * len(S)):
            cuts.append(x)
    xs = [left, *cuts, right]
    for st in S:
        old = [syms[i] for i in st.bars]
        new = []
        for a, b in zip(xs, xs[1:]):
            match = [o for o in old if not o.synthetic and abs(o.box[0] - a) < 1.5 * sp and abs(o.box[2] - b) < 1.5 * sp]
            if match:
                o = match[0]
                o.box = (a, o.box[1], b, o.box[3])
                new.append(o.i)
            else:
                band = [o for o in old if o.box[0] < b and o.box[2] > a] or old
                yy0 = sorted(o.box[1] for o in band)[len(band) // 2]
                yy1 = sorted(o.box[3] for o in band)[len(band) // 2]
                t = Sym(i=len(syms), cls="measure", post={CLS_ID["measure"]: 1.0}, p=0.0, n=0,
                        box=(a, yy0, b, yy1), synthetic=True)
                syms.append(t)
                new.append(t.i)
        st.bars = new
        st.x0, st.x1 = left, right


def _reconcile(syms: list[Sym], staves: list[Staff], sy: System, ref: int) -> None:
    """In a system, a staff's filled-in bars take the bar lines of the staff that has the most detected bars:
    its bars at that x are better evidence than lone bar-line boxes (stems read as bar lines split a gap)."""
    rb = [syms[i].box for i in staves[ref].bars]
    for k in sy.staves:
        st = staves[k]
        if k == ref or not any(syms[i].synthetic for i in st.bars):
            continue
        new = []
        for i in st.bars:
            s = syms[i]
            if not s.synthetic:
                new.append(i)
                continue
            x0, y0, x1, y1 = s.box
            cuts = [x0] + [r[2] for r in rb if x0 + st.sp < r[2] < x1 - st.sp] + [x1]
            # the stretch keeps its ends; inside it, the reference's bar lines
            first = True
            for a, b in zip(cuts, cuts[1:]):
                if first:
                    s.box = (a, y0, b, y1)
                    new.append(i)
                    first = False
                else:
                    t = Sym(i=len(syms), cls="measure", post={CLS_ID["measure"]: 1.0}, p=0.0, n=0,
                            box=(a, y0, b, y1), synthetic=True)
                    syms.append(t)
                    new.append(t.i)
        # filled bars that start on a reference bar's start take its left edge (a missed first bar)
        st.bars = sorted(new, key=lambda i: syms[i].cx)
    # the merged-away splits: synthetic bars inside one reference bar fold together
    for k in sy.staves:
        st = staves[k]
        if k == ref:
            continue
        keep = []
        for i in st.bars:
            s = syms[i]
            if keep and (s.synthetic or syms[keep[-1]].synthetic):
                prev = syms[keep[-1]]
                mid_prev, mid = (prev.box[0] + prev.box[2]) / 2, (s.box[0] + s.box[2]) / 2
                same = [j for j, r in enumerate(rb) if r[0] <= mid_prev <= r[2] and r[0] <= mid <= r[2]]
                if same:                       # one reference bar: one bar (the detected box, widened)
                    a, b = (prev, s) if not prev.synthetic or s.synthetic else (s, prev)
                    a.box = (min(prev.box[0], s.box[0]), a.box[1], max(prev.box[2], s.box[2]), a.box[3])
                    keep[-1] = a.i
                    continue
            keep.append(i)
        st.bars = keep


# ------------------------------------------------------------------------------------------------- ink path
def _ink_staves(gray, syms: list[Sym], spare: list | None = None) -> list[Staff] | None:
    """Staves from the staff lines in the ink: the page as it is, and the page cleaned (clean_page) -- a scan's
    tinted, shaded or grainy paper reads as ink to the line finder -- whichever set the detector's confident
    symbols confirm better (the raw page unless the cleaned one is clearly better). ``spare`` is given the staves
    that set saw and did not take (_spare_staves), likeliest first."""
    mine: list = []
    theirs: list = []
    staves = _ink_staves_from(gray, gray, syms, mine)
    other = _ink_staves_from(clean_page(gray), gray, syms, theirs)
    if other is not None and (staves is None or _staves_score(other, syms) > 1.02 * _staves_score(staves, syms)):
        staves, mine = other, theirs
    if spare is not None and staves is not None:
        spare.extend(mine)
    if staves is not None:
        _line_starts(gray, staves)
    return staves


def _line_starts(gray, staves: list[Staff]) -> None:
    """Each staff's line_x0: where the line finder starts its lines when near-white is paper (STAFF_PAPER_TOL).
    Its own threshold is 254 on a page with pure white, so a JPEG's or a screen capture's faint noise beside a brace
    (247-253 on 255 paper) is ink to it, joins the brace to the ends of the lines, and the staff starts on the brace,
    left of the line that opens the system. Only that line's search uses it (_left_span); the staff's x0 stays the
    finder's (the bar lines are cut from it)."""
    from PIL import Image

    from .vision.page.staves import find_staves
    found = find_staves(Image.fromarray(gray), paper_tol=STAFF_PAPER_TOL).staves
    for st in staves:
        near = [f for f in found if abs((f.top + f.bottom) / 2 - st.mid) < st.sp]
        if near:
            st.line_x0 = float(near[0].x0)


def _staves_score(staves: list[Staff], syms: list[Sym]) -> float:
    """How well the detector confirms a set of staves: the confident measure boxes and notes that sit on one of
    them (a staff the other set lacks shows here as the symbols it holds)."""
    held = [s for s in syms if (s.fam == "measure" or s.fam in ("note", "rest")) and s.p >= 0.5]
    return sum(1 for b in held if any(st.y0 - 0.5 * st.sp <= b.cy <= st.y1 + 0.5 * st.sp and
                                      st.x0 - st.sp <= b.cx <= st.x1 + st.sp for st in staves))


DOUBLE_OVERLAP = 0.1          # two staves sharing more than this of the lower one's height are one staff found twice


def _without_doubles(found: list, syms: list[Sym]) -> list:
    """The line finder's staves without those that double another. On a scan that bows or tilts, a staff line is two
    runs of rows, one for each end of it, and the finder groups each set of five into a staff: the same printed staff
    twice, about half a space apart. No two staves of a page overlap, so of two that share DOUBLE_OVERLAP of the
    lower one's height one goes: the one the detector's measure boxes sit on less well (a measure box spans its
    staff's five lines), else the one whose lines score lower. A doubled staff left in gives its system a staff the
    page does not print, and the page's parts are planned on its fullest system."""
    boxes = [s for s in syms if s.fam == "measure" and s.p >= MEASURE_LOW_P]

    def held(st) -> float:
        return sum(max(0.0, min(s.box[3], st.bottom) - max(s.box[1], st.top)) / max(1.0, s.box[3] - s.box[1])
                   for s in boxes if s.box[2] > st.x0 and s.box[0] < st.x1)

    def weight(st) -> tuple[float, float]:
        return held(st), sum(line.score for line in st.lines)

    out: list = []
    for st in sorted(found, key=lambda st: st.top):
        prev = out[-1] if out else None
        if prev is not None and min(prev.bottom, st.bottom) - max(prev.top, st.top) > \
                DOUBLE_OVERLAP * min(prev.bottom - prev.top, st.bottom - st.top):
            if weight(st) > weight(prev):
                out[-1] = st
            continue
        out.append(st)
    return out


ONE_LINE_REACH = 1.0          # a percussion clef stands within this many staff spaces of the one line of its staff
ONE_LINE_WIDTH = 0.6          # ... and that line runs at least this share of the width of the page's staves
ONE_LINE_APART = 4.0          # ... and no nearer than this many staff spaces to another staff's lines


def _one_line_staves(lines: list, staves: list[Staff], syms: list[Sym]) -> list[Staff]:
    """One-line staves (unpitched percussion: a triangle, a tambourine, a bass drum): each line of the finder's that
    belongs to no staff, runs as wide as the page's staves and has a percussion clef standing on it. The line finder
    knows five-line staves only, and the detector's measure boxes on a single line are a few rows tall and unsure, so
    such a staff was not there: its system read a staff short, and what stood on it went to the staff above or below
    or was dropped. It is given a five-line staff's height about its line, the page's staff space, and its bars with
    its system's (_segment)."""
    if not staves:
        return []
    sps = sorted(st.sp for st in staves)
    sp = sps[len(sps) // 2]
    widths = sorted(st.x1 - st.x0 for st in staves)
    width = widths[len(widths) // 2]
    out: list[Staff] = []
    for c in syms:
        if c.cls != "clef_perc" or c.p < 0.5 or any(st.y0 - st.sp <= c.cy <= st.y1 + st.sp for st in staves):
            continue                                  # none, or on a five-line staff (a drum kit's)
        near = [ln for ln in lines if abs(ln.y - c.cy) <= ONE_LINE_REACH * sp and ln.x1 - ln.x0 >= ONE_LINE_WIDTH * width
                and ln.x0 <= c.box[2] + 2 * sp]
        if not near:
            continue
        line = min(near, key=lambda ln: abs(ln.y - c.cy))
        if any(o.y0 - ONE_LINE_APART * sp < line.y < o.y1 + ONE_LINE_APART * sp for o in staves + out):
            continue
        out.append(Staff(bars=[], sp=sp, y0=line.y - 2 * sp, y1=line.y + 2 * sp, x0=float(line.x0), x1=float(line.x1),
                         one_line=True))
    return out


def _spare_staves(m, staves: list[Staff], syms: list[Sym], claimed: set) -> list[Staff]:
    """Staves the page may hold that were not taken, likeliest first: rows of measure boxes too unsure to make a
    staff of their own, then lines of the finder's that belong to no staff and run as wide as the page's staves (a
    one-line staff whose clef was not read). Each stands clear of the staves taken. They are staves only for a
    caller that knows the page holds more staves than were found (_with_layout)."""
    if not staves:
        return []
    sps = sorted(st.sp for st in staves)
    sp = sps[len(sps) // 2]
    widths = sorted(st.x1 - st.x0 for st in staves)
    width = widths[len(widths) // 2]

    def clear(y: float, of: list) -> bool:
        return not any(o.y0 - ONE_LINE_APART * sp < y < o.y1 + ONE_LINE_APART * sp for o in of)

    out: list[tuple[float, Staff]] = []
    weak = [s for s in syms if s.fam == "measure" and s.p >= MEASURE_LOW_P and s.i not in claimed and
            s.box[2] - s.box[0] >= MEASURE_MIN_W * (s.box[3] - s.box[1]) / 4]
    for st in _rows(weak):
        if clear(st.mid, staves) and st.x1 - st.x0 >= ONE_LINE_WIDTH * width:
            out.append((1.0 + sum(syms[i].p for i in st.bars) / len(st.bars), st))
    staffed = {id(line) for st in m.staves for line in st.lines}
    for line in m.lines:
        if id(line) in staffed or line.x1 - line.x0 < ONE_LINE_WIDTH * width or not clear(line.y, staves):
            continue
        out.append((line.score * (line.x1 - line.x0) / max(1.0, width),
                    Staff(bars=[], sp=sp, y0=line.y - 2 * sp, y1=line.y + 2 * sp, x0=float(line.x0), x1=float(line.x1))))
    spare: list[Staff] = []
    for _, st in sorted(out, key=lambda t: -t[0]):
        if clear(st.mid, spare):
            spare.append(st)
    return spare


def _ink_staves_from(page, gray, syms: list[Sym], spare: list | None = None) -> list[Staff] | None:
    """Staves from the staff lines in ``page`` (vision.page.staves: five long thin runs at one spacing), each
    holding the detected measure boxes that sit on it. A row of measure boxes no ink staff explains (a staff the
    line finder missed) is a staff of its own, as on the box path."""
    from PIL import Image

    from .vision.page.staves import find_staves
    m = find_staves(Image.fromarray(page))
    if not m.staves or m.spacing <= 0:
        return None
    # the measure boxes' own staff space checks the line finder (a skewed or curved page gives it false staves)
    hs = sorted(s.box[3] - s.box[1] for s in syms if s.fam == "measure" and s.p >= 0.5)
    box_sp = hs[len(hs) // 2] / 4 if hs else None
    found = [st for st in m.staves if box_sp is None or 0.6 * box_sp <= st.spacing <= 1.6 * box_sp]
    found = _without_doubles(found, syms)
    if not found:
        return None
    if hs and len(found) < 0.8 * len(_rows([s for s in syms if s.fam == "measure" and s.p >= MEASURE_MIN_P])):
        return None                               # the lines were found on too few staves: the box path
    staves = [Staff(bars=[], sp=max(1.0, st.spacing), y0=st.top, y1=st.bottom, x0=float(st.x0), x1=float(st.x1))
              for st in found]
    claimed = set()
    for s in syms:
        if s.fam != "measure" or s.p < MEASURE_LOW_P:
            continue
        h = s.box[3] - s.box[1]
        ov = [min(s.box[3], st.y1) - max(s.box[1], st.y0) for st in staves]
        k = max(range(len(staves)), key=lambda j: ov[j])
        if ov[k] >= 0.5 * min(h, staves[k].y1 - staves[k].y0):
            staves[k].bars.append(s.i)
            claimed.add(s.i)
    rest = [s for s in syms if s.fam == "measure" and s.p >= 0.5 and s.i not in claimed and
            s.box[2] - s.box[0] >= MEASURE_MIN_W * (s.box[3] - s.box[1]) / 4]
    for st in _rows(rest):
        if not any(abs(st.mid - o.mid) < 6 * o.sp for o in staves):
            staves.append(st)
    # a "staff" with no measure box and no note on it is lines that are not one (a ruled line, a border, a scratch)
    notes = [s for s in syms if s.fam in ("note", "rest") and s.p >= 0.5]
    held = [st for st in staves if st.bars or
            sum(1 for n in notes if st.y0 - st.sp <= n.cy <= st.y1 + st.sp and st.x0 <= n.cx <= st.x1) >= 2]
    bare = [st for st in staves if not any(st is o for o in held)]
    staves = held
    if not staves:
        return None
    staves += _one_line_staves(m.lines, staves, syms)
    if spare is not None:
        spare.extend(_spare_staves(m, staves, syms, claimed))
        # the finder's own staves that hold nothing: staves of rests, where the caller counts them (_with_layout);
        # as wide as the page's staves, so that an ossia or a ruled box is not one
        widths = sorted(st.x1 - st.x0 for st in staves)
        for st in bare:
            if st.x1 - st.x0 >= ONE_LINE_WIDTH * widths[len(widths) // 2]:
                st.lined = True
                spare.append(st)
    for st in staves + (spare or []):
        st.bars.sort(key=lambda i: syms[i].cx)
    staves.sort(key=lambda st: st.mid)
    return staves


def _segment(syms: list[Sym], staves: list[Staff], sy: System, gray) -> None:
    """A system's bars from the evidence of all its staves at once: a candidate x (a measure box's edge or a
    detected bar line) is a bar line where a vertical stroke crosses the staff's full height -- on most staves of a
    multi-staff system, with a detection behind it; on a single staff, a detection and the stroke (or a firm
    bar-line detection alone). Every staff gets exactly those bars, from the staff lines' left end to their right
    end; a detected measure box that matches a bar keeps its identity, the others are filled in."""
    S = [staves[k] for k in sy.staves]
    n = len(S)
    sp = sum(st.sp for st in S) / n
    left, right = min(st.x0 for st in S), max(st.x1 for st in S)
    cand = []                                       # (x, staff index, kind, p)
    for j, st in enumerate(S):
        for i in st.bars:
            b = syms[i]
            if b.p >= 0.6 and not b.synthetic:
                cand += [(b.box[0], j, "edge", b.p), (b.box[2], j, "edge", b.p)]
        for b in syms:
            if b.fam == "barLine" and b.p >= 0.3 and st.y0 - sp <= b.cy <= st.y1 + sp:
                cand.append((b.cx, j, "line", b.p))
    if n >= 2:
        # a column of ink over the full height of every staff at once is a bar line, detected or not
        import numpy as np
        xa, xb = int(left + 2 * sp), int(right - 1.5 * sp)
        if xb > xa:
            rows = [gray[int(st.y0) + 1:int(st.y1), xa:xb].mean(axis=0) for st in S]
            if n == 2:                  # two hands' stems line up; a grand staff's bar line also crosses the gap
                rows.append(gray[int(S[0].y1) + 1:int(S[1].y0), xa:xb].mean(axis=0))
            cover = np.min(rows, axis=0)
            run = None
            for x, v in enumerate(cover >= 0.97):
                if v and run is None:
                    run = x
                if (not v or x == len(cover) - 1) and run is not None:
                    cand.append((xa + (run + x - 1) / 2, -1, "ink", 1.0))
                    run = None
    cand = sorted(c for c in cand if left + 2 * sp < c[0] < right - 1.5 * sp)
    clusters: list[list] = []
    for c in cand:
        if clusters and c[0] - clusters[-1][-1][0] <= sp:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    cuts = []
    for cl in clusters:
        lines = [c for c in cl if c[2] == "line"]
        x = sorted(c[0] for c in (lines or cl))[len(lines or cl) // 2]
        # the stroke anywhere across the candidates' own span: a double bar line's centre is the gap between its
        # two strokes
        span = [c[0] for c in (lines or cl)]
        xr = range(int(min(span) - 0.5 * sp), int(max(span) + 0.5 * sp) + 1, 2)
        ink = [max(_ink_run(gray, xx, st.y0 + 1, st.y1 - 1, reach=2) for xx in xr) >= 0.9 for st in S]
        with_line = {c[1] for c in lines}
        firm = max((c[3] for c in lines), default=0.0)
        if n >= 2:
            # stems of notes struck together line up across the staves too: a bar line needs detections on at
            # least half the staves and the stroke on most -- or the stroke through every staff's full height
            ok = (len(with_line) >= max(1, n / 2) and sum(ink) >= 0.75 * n) or len(with_line) >= 0.75 * n or \
                any(c[2] == "ink" for c in cl)
        else:
            edges = [c for c in cl if c[2] == "edge"]
            ok = (ink[0] and (firm >= 0.5 or len(edges) >= 2)) or firm >= 0.85
        if ok:
            cuts.append(x)
    cuts = sorted(cuts + _FORCED.get(sy.staves[0], []))     # bar lines the readings showed were missed (revise.py)
    xs = [left, *cuts, right]
    # no bar is narrower than MEASURE_MIN_W staff spaces: a sliver (a double bar line's second stroke, the staff
    # lines running on past the last bar line) joins its neighbour
    while len(xs) > 2:
        w = [b - a for a, b in zip(xs, xs[1:])]
        k = min(range(len(w)), key=lambda j: w[j])
        if w[k] >= MEASURE_MIN_W * sp:
            break
        del xs[k + 1 if k == 0 else k]
    # a first or last stretch with no note or rest in it is no bar: the staff lines' run into a margin, or the
    # cautionary key / clef after a system's last bar line
    music = [m for m in syms if m.fam in ("note", "rest", "mRest", "multiRest", "mRpt", "multiRpt") and m.p >= 0.4
             and S[0].y0 - 4 * sp <= m.cy <= S[-1].y1 + 4 * sp]

    def empty(a, b):
        return not any(a <= m.cx <= b for m in music)

    while len(xs) > 2 and empty(xs[-2], xs[-1]):
        del xs[-2]
    while len(xs) > 2 and empty(xs[0], xs[1]):
        del xs[1]
    # inside the system, a narrow stretch with nothing in it is a bar-line group (a repeat sign, a thick double
    # bar), not a bar: its two cuts become one, in its middle
    k = 1
    while k < len(xs) - 2:
        a, b = xs[k], xs[k + 1]
        if b - a < EMPTY_BAR_MAX_SP * sp and empty(a, b):
            xs[k:k + 2] = [(a + b) / 2]
        else:
            k += 1
    for st in S:
        old = [syms[i] for i in st.bars]
        new = []
        for a, b in zip(xs, xs[1:]):
            match = [o for o in old if not o.synthetic and abs(o.box[0] - a) < 1.5 * sp and abs(o.box[2] - b) < 1.5 * sp]
            if match:
                o = max(match, key=lambda o: o.p)
                o.box = (a, o.box[1], b, o.box[3])
                new.append(o.i)
            else:
                t = Sym(i=len(syms), cls="measure", post={CLS_ID["measure"]: 1.0}, p=0.0, n=0,
                        box=(a, st.y0, b, st.y1), synthetic=True)
                syms.append(t)
                new.append(t.i)
        st.bars = new
        st.x0, st.x1 = left, right


# ------------------------------------------------------------------------------------------------- assignment
def _band(syms: list[Sym], st: Staff, x: float) -> tuple[int, float, float, float]:
    """(bar index, top, bottom, staff space) of staff ``st`` at x."""
    best, bd = 0, math.inf
    for j, i in enumerate(st.bars):
        b = syms[i].box
        d = 0.0 if b[0] <= x <= b[2] else min(abs(x - b[0]), abs(x - b[2]))
        if d < bd:
            best, bd = j, d
    b = syms[st.bars[best]].box
    if syms[st.bars[best]].synthetic:
        return best, st.y0, st.y1, st.sp
    return best, b[1], b[3], max(1.0, (b[3] - b[1]) / 4)


def _assign(syms: list[Sym], staves: list[Staff], only: list[Sym] | None = None) -> None:
    left = min((st.x0 for st in staves), default=0.0)
    right = max((st.x1 for st in staves), default=0.0)
    for k, st in enumerate(staves):
        for j, i in enumerate(st.bars):
            syms[i].staff, syms[i].bar = k, j
    for s in (syms if only is None else only):
        if s.staff >= 0 or s.fam == "measure":
            continue
        x = s.box[2] if s.cls == "barLine_repeat_start" else s.cx      # it opens the bar on its right
        best, bc, bbar = -1, math.inf, -1
        pos = None
        if s.fam in ("note", "rest") and s.attrs.get("staff_position") not in (None, "<na>"):
            try:
                pos = int(s.attrs["staff_position"])
            except ValueError:
                pos = None
            # a saturated (the head's last value) or unsure reading says nothing about the staff
            if pos is not None and (abs(pos) >= 20 or float(s.attr_p.get("staff_position", 1.0)) < 0.5):
                pos = None
        for k, st in enumerate(staves):
            # any staff across the page's width: a staff whose first bars the detector missed starts late
            if not (left - 4 * st.sp <= x <= right + 4 * st.sp):
                continue
            j, y0, y1, sp = _band(syms, st, x)
            d = max(0.0, y0 - s.cy, s.cy - y1) / sp
            if pos is not None and s.fam == "note":
                implied = ((y0 + y1) / 2 - s.cy) / (sp / 2)
                d = min(d + 1.0, abs(implied - pos) / 2) if d > 0 else abs(implied - pos) / 2
            if d < bc:
                best, bc, bbar = k, d, j
        if best >= 0 and bc <= ASSIGN_MAX_SP:
            s.staff, s.bar = best, bbar


DROP_SHARE = 0.34             # a staff with less than this share of the page's median staff's bars may go (_with_layout)
SPARE_HOLDS = 2               # a staff the finder left joins only if this many sure symbols stand on it
GAP_CLEAR = 1.5               # a staff in a gap between two staves stands this many staff spaces clear of both
GAP_LINE = 0.3                # a row of the gap is a staff line where it holds ink over this share of the staves' width
GAP_EVEN = 0.2                # five lines are a staff's where their distances are within this share of their mean
LINED_SPACING = 0.15          # a staff known by its lines alone is ruled like the page's staves to within this share


def _lined_staves(gray, staves: list[Staff], syms: list[Sym]) -> list[Staff]:
    """On a page whose staves are the rows of its measure boxes (the line finder found too few to go by): the staves
    the finder did find that stand clear of every one of those and are as wide as a staff. The boxes miss a staff
    of rests, which the finder sees as well as any other. They are staves only for a caller that knows the page holds more than were found."""
    from PIL import Image

    from .vision.page.staves import find_staves
    sps = sorted(st.sp for st in staves)
    sp = sps[len(sps) // 2]
    widths = sorted(st.x1 - st.x0 for st in staves)
    width = widths[len(widths) // 2]
    out: list[Staff] = []
    for page in (gray, clean_page(gray)):
        m = find_staves(Image.fromarray(page))
        for st in _without_doubles([f for f in m.staves if 0.6 * sp <= f.spacing <= 1.6 * sp], syms):
            mid = (st.top + st.bottom) / 2
            if st.x1 - st.x0 >= ONE_LINE_WIDTH * width and \
                    not any(o.y0 - GAP_CLEAR * sp < mid < o.y1 + GAP_CLEAR * sp for o in staves + out):
                out.append(Staff(bars=[], sp=max(1.0, st.spacing), y0=st.top, y1=st.bottom, x0=float(st.x0),
                                 x1=float(st.x1), lined=True))
    return out


def _borrow_bars(syms: list[Sym], staves: list[Staff], before: list[Staff]) -> None:
    """A staff taken by its lines among staves made of measure boxes has no boxes: it is given its nearest neighbour's
    bars, at its own height (the staves of a system share their bar lines, and a staff taken so is one of rests)."""
    old = [st for st in staves if any(st is o for o in before) and st.bars]
    for st in staves:
        if st.bars or not st.lined or not old:
            continue
        near = min(old, key=lambda o: abs(o.mid - st.mid))
        new = []
        for i in near.bars:
            box = syms[i].box
            s = Sym(i=len(syms), cls="measure", post={CLS_ID["measure"]: 1.0}, p=0.0, n=0,
                    box=(box[0], st.y0, box[2], st.y1), synthetic=True)
            syms.append(s)
            new.append(s.i)
        st.bars, st.x0, st.x1 = new, near.x0, near.x1


def _gap_staves(mask, staves: list[Staff]) -> list[Staff]:
    """Staves the line finder did not find, in the gaps between those it did: five rows of ink, evenly a staff space
    apart, clear of the staves above and below, and as wide as the page's staves by their own ink (an ossia, a ruled
    box or a line of text is not). The finder wants its lines long, thin and sharp; on a blurred scan a staff of
    rests (nothing on it but its lines, and those grey) is not one to it, and the page reads a staff short. They
    are staves only for a caller that knows the page holds more than were found (_with_layout takes them after the
    finder's own spare ones)."""
    import numpy as np
    out: list[Staff] = []
    if len(staves) < 2:
        return out
    sps = sorted(st.sp for st in staves)
    sp = sps[len(sps) // 2]
    widths = sorted(st.x1 - st.x0 for st in staves)
    width = widths[len(widths) // 2]
    order = sorted(staves, key=lambda st: st.mid)
    for a, b in zip(order, order[1:]):
        x0, x1 = int(max(a.x0, b.x0) + 2 * sp), int(min(a.x1, b.x1) - 2 * sp)
        ya, yb = max(0, int(a.y1 + GAP_CLEAR * sp)), min(mask.shape[0], int(b.y0 - GAP_CLEAR * sp))
        if yb - ya < 4 * sp or x1 - x0 < 10 * sp:
            continue
        rows = mask[ya:yb, max(0, x0):x1].mean(axis=1) >= GAP_LINE
        runs, start = [], None                      # (middle row, thickness) of each run of line rows
        for y, on in enumerate([*rows, False]):
            if on and start is None:
                start = y
            elif not on and start is not None:
                runs.append(((start + y - 1) / 2, y - start))
                start = None
        runs = [r for r in runs if r[1] <= 0.6 * sp]
        for k in range(len(runs) - 4):
            ys = [r[0] for r in runs[k:k + 5]]
            steps = [q - p for p, q in zip(ys, ys[1:])]
            mean = sum(steps) / 4
            if not (0.7 * sp <= mean <= 1.4 * sp and all(abs(step - mean) <= GAP_EVEN * mean for step in steps)):
                continue
            # where its lines run: the columns most of the five hold ink in, between the first and the last of them
            cols = mask[[ya + int(round(y)) for y in ys]].sum(axis=0) >= 3
            reach = max(2, int(2 * sp))               # an end of the lines: from there on the lines are there
            dense = np.convolve(cols, np.ones(reach), "valid") >= 0.8 * reach
            inked = np.flatnonzero(dense)
            if not len(inked):
                continue
            on = np.flatnonzero(cols[inked[0]:inked[-1] + reach]) + inked[0]
            left, right = float(on[0]), float(on[-1])
            if right - left >= ONE_LINE_WIDTH * width and cols[int(left):int(right) + 1].mean() >= 0.5:
                out.append(Staff(bars=[], sp=max(1.0, mean), y0=ya + ys[0], y1=ya + ys[4], x0=left, x1=right, lined=True))
                break
    return out


def _staff_bars(st: Staff, width: float) -> float:
    """How much of a staff a staff is: the measure boxes on it, scaled by its width against the page's staves (an
    inset ossia runs a bar or two; a line of text read as a staff has a box or none). A staff of rests has as many
    boxes as its neighbours, so what it holds does not count: a silent staff is a staff."""
    return math.inf if st.one_line else len(st.bars) * min(1.0, (st.x1 - st.x0) / max(1.0, width))


def _on_staff(syms: list[Sym], st: Staff) -> int:
    """The symbols the detector is sure of that stand on a staff: notes, rests, bar rests and clefs."""
    return sum(1 for s in syms if s.fam in ("note", "rest", "mRest", "multiRest", "clef") and s.p >= 0.5 and
               st.x0 <= s.cx <= st.x1 and st.y0 - st.sp <= s.cy <= st.y1 + st.sp)


def _apart(syms: list[Sym], a: Staff, b: Staff) -> bool:
    """Two staves whose bar lines plainly do not line up: each has at least two of its own over their common
    stretch, and fewer than half of either's have a partner on the other. Staves of one system share their bar
    lines; with fewer to go by, nothing is said."""
    sp = (a.sp + b.sp) / 2
    lo, hi = max(a.x0, b.x0) - sp, min(a.x1, b.x1) + sp

    def edges(st):
        return [syms[k].box[2] for k in st.bars[:-1] if not syms[k].synthetic and lo <= syms[k].box[2] <= hi]

    ea, eb = edges(a), edges(b)
    if len(ea) < 2 or len(eb) < 2:
        return False
    tol = ALIGN_TOL_SP * sp
    hit = lambda xs, ys: sum(1 for x in xs if any(abs(x - y) <= tol for y in ys)) / len(xs)      # noqa: E731
    return hit(ea, eb) < 0.5 and hit(eb, ea) < 0.5


LINED_UP = 0.75                # two staves share their bar lines where this share of the fewer one's have a partner
TELLING = 3                    # a staff says whose bar lines it shares once it shows this many
AROUND = 3                     # staves on each side of a join that speak for it
AROUND_TOL_SP = 4.0            # two staves' bar lines are one where they stand this near (staff spaces), and within
AROUND_TOL_BAR = 0.1           # this share of the narrowest bar between them


def _lined_up(syms: list[Sym], a: Staff, b: Staff) -> bool | None:
    """Whether two staves share their bar lines: of those of the one that shows fewer over their common stretch,
    LINED_UP have a partner on the other. None where either shows fewer than TELLING: two lines coincide by chance
    too easily to say anything.

    A partner stands within AROUND_TOL_SP staff spaces, wider than _apart allows: these are the right edges of
    measure boxes, which on some scans end three or four staff spaces apart on staves of one system, and what
    tells one system from two is how many of them agree, not how exactly (three unrelated lines each finding a
    partner that near happens once in some hundred pairs). On a page of narrow bars the reach is a tenth of the
    narrowest bar, so that lines a bar apart are never partners."""
    sp = (a.sp + b.sp) / 2
    lo, hi = max(a.x0, b.x0) - sp, min(a.x1, b.x1) + sp
    ea, eb = (sorted(syms[k].box[2] for k in st.bars[:-1] if not syms[k].synthetic and lo <= syms[k].box[2] <= hi) for st in (a, b))
    if len(ea) < TELLING or len(eb) < TELLING:
        return None
    few, many = (ea, eb) if len(ea) <= len(eb) else (eb, ea)
    tol = min(AROUND_TOL_SP * sp, AROUND_TOL_BAR * min(y - x for x, y in zip(many, many[1:])))
    return sum(1 for x in few if any(abs(x - y) <= tol for y in many)) >= LINED_UP * len(few)


def _joined_around(syms: list[Sym], upper: list[Staff], lower: list[Staff]) -> bool:
    """Whether the staves on the two sides of a join share bar lines although the two at the join do not: of the
    pairs of one staff above it and one below it that have something to say, more than half line up. Only the AROUND
    staves nearest the join on each side speak (``upper`` and ``lower`` run to the join and from it): staves further
    off say nothing of it, and where a layout states one system over three, the outer two, alike, would vouch for
    both joins with the one between them. Where a staff
    shows little (a staff of rests in a band score: three measure boxes for six bars) the detector's boxes end two
    or three staff spaces off, and that one staff lines up with neither neighbour; a system is not parted for it
    where the staves around it are plainly one. Staves that are two systems each have bar lines of their own, and
    hardly a pair across them lines up. Where no pair has enough to say because one side shows too little (one
    staff of rests under a system), that side is held to every bar line it shows: each has to stand where a staff
    of the other side that shows enough has one. Where neither side shows enough, nothing is said."""
    upper, lower = upper[-AROUND:], lower[:AROUND]
    said = [s for s in (_lined_up(syms, u, v) for u in upper for v in lower) if s is not None]
    if said:
        return 2 * sum(said) > len(said)
    both = upper + lower
    sp = sum(st.sp for st in both) / len(both)
    lo, hi = max(st.x0 for st in both) - sp, min(st.x1 for st in both) + sp
    shown = lambda st: [syms[k].box[2] for k in st.bars[:-1] if not syms[k].synthetic and lo <= syms[k].box[2] <= hi]   # noqa: E731
    telling = [[x for st in side if len(shown(st)) >= TELLING for x in shown(st)] for side in (upper, lower)]
    if bool(telling[0]) == bool(telling[1]):
        return False                              # neither side has a staff that says enough
    few = [x for st in (lower if telling[0] else upper) for x in shown(st)]
    many = telling[0] or telling[1]
    return len(few) >= 2 and all(any(abs(x - y) <= ALIGN_TOL_SP * sp for y in many) for x in few)


def _with_layout(layout: list, syms: list[Sym], staves: list[Staff], spare: list[Staff], links_of,
                 add: bool, most: int | None = None) -> tuple[dict, list[Staff], list[bool]]:
    """The page read with a layout its caller knows: the staves of each of its systems, counted top to bottom (from
    a reading of the whole score at a glance, where each page is a thumbnail: enough to count staves, and it sees
    what one page's ink does not say). ``links_of``: staves -> (pairs joined, what said so), _links for this page.
    Returns (what it did, the staves, the pairs joined).

    The layout decides how the staves found are grouped into systems, once they are as many as it counts. Where
    more were found, the weakest go -- those with less than DROP_SHARE of the page's median staff both in bars and
    in what stands on them (an inset ossia, a line of text read as a staff, a second reading of one staff): never
    a staff for being silent, nor one that holds music the detector found few bars for; where fewer, the likeliest
    of the staves the finder saw and did not take join (``spare``, when ``add``), each only with SPARE_HOLDS sure
    symbols on it. The layout is evidence too, and may be wrong: it joins, it does not part. The front end joins
    two staves on what the page shows -- a line through the gap, bar lines running on, a brace -- and misses a join
    where that ink is faint; a count taken from a thumbnail sees neither a dashed connector nor that the system
    below is an alternative printed under the one above. So a layout is not used where it would part two staves
    the page joins; where it cannot be met as above; where it counts a system of more staves than the score has
    (``most``, where the caller knows the score's instruments); or where a join it adds does not fit the page: two
    staves whose bar lines plainly do not line up (unless staves of the system the layout forms line up across
    them, _joined_around), or a gap as wide as one it leaves between two systems (staves of one system stand
    closer than systems do). The page is then read as without it, and ``why`` says so. The
    notes are read as ever; the layout says only which staves there are and which are played together."""
    linked, said = links_of(staves)
    sizes, run = [], 0
    for k in range(len(staves)):
        if k and not linked[k - 1]:
            sizes.append(run)
            run = 0
        run += 1
    if staves:
        sizes.append(run)
    try:
        target = [int(n) for n in layout]
    except (TypeError, ValueError):
        target = []
    prior = {"target": target, "own": sizes, "said": list(said), "used": False, "dropped": 0, "added": 0}

    def unused(why: str):
        prior.update(dropped=0, added=0, why=why)
        return prior, staves, linked

    total = sum(target)
    if not target or any(n < 1 for n in target):
        return unused("the layout counts no staves")
    if most is not None and max(target) > most:
        return unused(f"the layout counts a system of {max(target)} staves and the score has {most}")
    fitted = sorted(staves, key=lambda st: st.mid)
    if len(fitted) > total:
        widths = sorted(st.x1 - st.x0 for st in fitted)
        bars = {id(st): _staff_bars(st, widths[len(widths) // 2]) for st in fitted}
        holds = {id(st): _on_staff(syms, st) for st in fitted}
        few_bars = DROP_SHARE * sorted(bars.values())[len(fitted) // 2]
        holds_little = DROP_SHARE * sorted(holds.values())[len(fitted) // 2]
        for st in sorted(fitted, key=lambda st: (bars[id(st)], holds[id(st)])):
            if len(fitted) == total:
                break
            if bars[id(st)] < few_bars and holds[id(st)] <= holds_little:
                fitted = [o for o in fitted if o is not st]
        prior["dropped"] = len(staves) - len(fitted)
    elif len(fitted) < total and add:
        for st in spare:
            if len(fitted) == total:
                break
            # five even lines across the page are a staff whatever stands on it: a part that rests is printed so
            if st.lined or _on_staff(syms, st) >= SPARE_HOLDS:
                fitted.append(st)
        prior["added"] = len(fitted) - len(staves)
        fitted.sort(key=lambda st: st.mid)
    if len(fitted) != total:
        return unused(f"{len(staves)} staves found and {total} counted")
    ends, at = set(), 0
    for n in target[:-1]:
        at += n
        ends.add(at)
    forced = [k not in ends for k in range(1, total)]
    joined, _ = links_of(fitted) if prior["dropped"] or prior["added"] else (linked, said)
    gap = lambda k: fitted[k].y0 - fitted[k - 1].y1                        # noqa: E731
    between = min((gap(k) for k in ends), default=math.inf)
    bounds = [0, *sorted(ends), total]
    above = lambda k: fitted[max(b for b in bounds if b < k):k]            # noqa: E731  (the layout's system, to the join)
    below = lambda k: fitted[k:min(b for b in bounds if b > k)]            # noqa: E731
    for k in range(1, total):
        if joined[k - 1] and not forced[k - 1]:
            return unused("the layout parts staves the page joins")
        if forced[k - 1] and not joined[k - 1]:
            if _apart(syms, fitted[k - 1], fitted[k]) and not _joined_around(syms, above(k), below(k)):
                return unused("the layout joins staves whose bar lines do not line up")
            if gap(k) >= between:
                return unused("the layout joins staves as far apart as two of its systems")
    prior["used"] = True
    prior["joined"] = [k for k in range(1, total) if forced[k - 1] and not joined[k - 1]]
    return prior, fitted, forced


def _loses_bars(systems: list[System], shown: list[int], joined: list[int]) -> bool:
    """Whether a system the layout formed by joining staves has fewer than two thirds of the bars its staves show by
    themselves (``shown``: per staff, its measure boxes; ``joined``: the pairs (k - 1, k) the layout joined). The bars
    of a system are cut where its staves agree on a bar line; staves that are not one system do not agree, and nor
    do the staves of one on a page photographed askew. Either way the join does not fit what the page shows."""
    for sy in systems:
        if not any(k in sy.staves and k - 1 in sy.staves for k in joined):
            continue
        bars = 1 + max((max(c) for c in sy.columns if c), default=-1)
        if 3 * bars < 2 * max(shown[k] for k in sy.staves):
            return True
    return False


CAND_SKIP = ("measure", "barLine")         # structure is decided before candidates join


_FORCED: dict = {}            # system (its top staff's index) -> x of bar lines to add (build's ``cuts``)


def build(dets: list[dict], width: float, height: float, image=None, cands: list[dict] | None = None,
          cuts: dict | None = None, layout: list | None = None, most: int | None = None) -> Layout:
    """``image``: the page (PIL image or grey array); with it, the ink between staves decides the systems.
    ``layout``: the staves of each of the page's systems, top to bottom, where the caller knows them
    (_with_layout), and ``most``: the most staves a system of the score can have; ``Layout.prior`` then says what
    the layout did.
    ``cands``: the detector's sub-threshold candidates; once the structure stands, the ones no symbol of their
    family already covers join as symbols flagged ``cand`` (the model decides whether they are real).
    ``cuts``: bar lines to add, per system (its top staff's index) -- the ones a first reading showed were
    missed (revise.py)."""
    import numpy as np
    _FORCED.clear()
    _FORCED.update(cuts or {})
    gray = None
    if image is not None:
        gray = unframe(np.asarray(image.convert("L") if hasattr(image, "convert") else image))
    syms = merge(dets)
    spare: list = []
    staves = _ink_staves(gray, syms, spare) if gray is not None else None
    mask = page_mask(gray) if gray is not None else None
    ink_path = bool(staves)
    if not staves:
        staves = _staves(syms)
        _fill_gaps(syms, staves)
        _snap_bars(syms, staves)
    linked, prior = None, None
    if layout is not None:
        if mask is not None and staves:
            if not ink_path:
                spare.extend(_lined_staves(gray, staves, syms))
            # after the finder's own, likeliest first; a gap is between staves, not between a staff and a stray line
            lined = [st for st in spare if st.lined]
            spare.extend(st for st in _gap_staves(mask, staves + lined)
                         if not any(o.y0 - GAP_CLEAR * st.sp < st.mid < o.y1 + GAP_CLEAR * st.sp for o in lined))
            # of the staves known by their lines alone, those ruled like the page's staves, the likest first: a
            # tablature's six lines or a ruled block are spaced otherwise, and are not taken for a staff of rests
            sps = sorted(st.sp for st in staves)
            sp = sps[len(sps) // 2]
            # (the staff space of staves made of measure boxes is a quarter of a box's height, and only roughly the
            # lines': there the likeness asked is half as close)
            near = LINED_SPACING * (1 if ink_path else 2) * sp
            lined = sorted((st for st in spare if st.lined and abs(st.sp - sp) <= near), key=lambda st: abs(st.sp - sp))
            spare[:] = ([st for st in spare if not st.lined] if ink_path else []) + lined
        before = list(staves)
        prior, staves, linked = _with_layout(layout, syms, staves, spare, lambda sts: _links(syms, sts, mask, gray),
                                             add=ink_path or bool(spare), most=most)
        if not ink_path:
            _borrow_bars(syms, staves, before)
    shown = [sum(not syms[i].synthetic for i in st.bars) for st in staves]
    systems = _systems(syms, staves, mask, ink=ink_path, levels=gray, linked=linked)
    if prior is not None and prior.get("joined") and _loses_bars(systems, shown, prior["joined"]):
        L = build(dets, width, height, image=image, cands=cands, cuts=cuts)           # the page as read without it
        L.prior = {**prior, "used": False, "dropped": 0, "added": 0, "joined": [],
                   "why": "the layout joins staves into a system with fewer bars than they show"}
        return L
    _assign(syms, staves)
    if cands:
        new = []
        for c in merge(cands, offset=len(dets)):
            if c.fam in CAND_SKIP or any(o.fam == c.fam and iou(o.box, c.box) >= 0.3 for o in syms):
                continue
            c.i, c.cand = len(syms), True
            syms.append(c)
            new.append(c)
        _assign(syms, staves, only=new)
    sps = sorted(st.sp for st in staves)
    sp = sps[len(sps) // 2] if sps else 10.0
    return Layout(syms=syms, staves=staves, systems=systems, sp=sp, width=width, height=height, ink=ink_path,
                  prior=prior)
