# Added by Play Anything (https://github.com/playanything-org/copisteria): see README.md.
"""Parts planned on a roster the caller knows, instead of on the page's fullest system.

The writer plans a page's parts on its fullest system and reads every other system against it (write._plan_parts,
write._align_systems). A page says less than the score does: a page of piano alone is one part to it, the song's
voice nowhere; a system misread with a staff too many is its fullest. A caller that knows the score's instruments
(read once from its opening page) gives them as a roster -- the number of staves of each, top to bottom -- and the
page's parts are then the roster's, the same on every page of the score.

The roster is evidence, not authority: it is used only on a page all of whose systems it explains. Every system
has to be the roster or an ordered part of it, a brace has to join the two staves of one of its two-staff
instruments, and no system may show more staves than it has. A page it does not explain (another piece, an
ensemble that changed, a roster read wrong) is planned as before, and says why.

Nor is a system placed by a guess. A system of fewer staves than the roster is some of its instruments, and which
ones only its braces and opening clefs say: two voices over a piano are the roster's second and third voices as
readily as its first and second, and written onto the first two they leave the players who sang with nothing and
give their music to players who rested. Where the evidence fits a system to the roster more than one way the page is
not explained, and is planned as before, each system on the parts the page itself shows.

A caller that read the score's print can say more. ``clefs``: the clef each of the roster's staves opens with, for a
page that shows no whole system of its own to read them from. ``among``: for each staff of each of the page's
systems, the roster's instruments it may be, as the label printed by the staff names them ("T." and "B." over the
piano are the tenor and the bass, where braces and clefs could not say which two of four voices sing). Whose label a
label is, is the caller's to decide: this module knows no names. A staff stands only at an instrument it may be, a
whole system one of whose staves may not be the instrument at its place is not this roster's, and the rule above
stands for what remains: one fit, or none. What it says of the staves is of the systems it saw: where those are not
the systems found here (another number of them, or of staves in one), nothing it sent explains the page.
"""
from __future__ import annotations

import math
from collections import Counter

from .read import Reading

EPS = 1e-9             # two fits of a system to the roster score the same


def _stated_clef_says(staff: str, place: str) -> float:
    """What a staff's opening clef says of a place whose clef the caller read in print: the same clef, another, or
    nothing to tell. A clef with and without its octave mark is one clef here: a tenor's staff is read either way, by
    this reader and by whoever read the print."""
    if "other" in (staff, place):
        return 0.0
    return 1.0 if staff.split("_")[0] == place.split("_")[0] else -1.0


def plan(rd: Reading, counts: list[int], clefs: list | None = None,
         among: list | None = None) -> tuple[tuple[list, list[dict[int, int]]] | None, str | None]:
    """((plans, place), None) for a page the roster explains: the roster's parts, and per system place -> staff
    as write._align_systems gives it; else (None, why not). ``clefs``: per staff of the roster its opening clef
    (write.MID_LINE's names), or None. ``among``: per system of the page, top to bottom, per staff the indices of the
    roster's instruments it may be, or None for any. ``clefs`` is used as far as it is well formed. ``among`` has a
    row for each system and an entry for each staff of each, or the caller's systems are not these and the page is not
    explained."""
    from . import write as W
    L = rd.layout
    if not L.systems:
        return None, "the page has no system"
    if not counts or any(not isinstance(c, int) or c < 1 for c in counts):
        return None, "no roster"
    K = sum(counts)
    widest = max(len(sy.staves) for sy in L.systems)
    if widest > K:
        return None, f"a system of {widest} staves, and the roster has {K}"
    plans, at = [], 0
    for c in counts:
        plans.append(W.PartPlan(staves=list(range(at, at + c))))
        at += c
    pair = {pl.staves[0] for pl in plans if len(pl.staves) == 2}        # first places of the two-staff parts
    part_of = {p: n for n, pl in enumerate(plans) for p in pl.staves}
    stated = [x if isinstance(x, str) and x in W.MID_LINE else None for x in clefs] \
        if isinstance(clefs, list) and len(clefs) == K else [None] * K
    index = lambda x: isinstance(x, int) and not isinstance(x, bool) and 0 <= x < len(counts)      # noqa: E731
    # per system, per staff: its instruments, or None. A caller that says it at all read the page's systems, and a
    # row that does not fit its system (another number of staves, or of systems) says they are not the ones found
    # here: then what it read is of other systems, the roster it sent with it too, and the page is not explained (a
    # part's page read as one system of seven staves stood on seven of the roster's instruments, by the clefs alone)
    may: list = [None] * len(L.systems)
    if among is not None:
        if not (isinstance(among, list) and len(among) == len(L.systems) and all(
                isinstance(row, list) and len(row) == len(sy.staves) and
                all(x is None or isinstance(x, list) and x and all(index(y) for y in x) for x in row)
                for row, sy in zip(among, L.systems))):
            return None, "the caller's systems are not the ones found"
        may = [[None if x is None else frozenset(x) for x in row] for row in among]

    def allowed(n: int, i: int, p: int) -> bool:
        """Whether staff i of system n may stand at place p."""
        return may[n] is None or may[n][i] is None or part_of[p] in may[n][i]
    braces = [s for s in L.syms if W._real(rd, s.i) and W._cls(rd, s.i) == "grpSym_brace"]
    own = [W._braced_pairs(L, sy, braces) for sy in L.systems]
    opens = [[W._opening_clef(rd, k) for k in sy.staves] for sy in L.systems]     # as read, per system and staff
    # what stands at each place: the opening clef most of the page's whole systems show there; where the page shows
    # none there, the one the caller read in print
    whole = [n for n, sy in enumerate(L.systems) if len(sy.staves) == K]
    t_clef: list = []
    for p in range(K):
        seen = Counter(opens[n][p] for n in whole if opens[n][p])
        t_clef.append(seen.most_common(1)[0][0] if seen else None)
    told = [t_clef[p] is None and stated[p] is not None for p in range(K)]
    t_clef = [stated[p] if told[p] else t_clef[p] for p in range(K)]
    # (where the caller stated clefs and left one out, that staff's clef is unknown rather than untold: a staff may as
    # well be that instrument's as one whose clef it shows)
    open_place = [any(told) and t_clef[p] is None for p in range(K)]
    place = []
    for n, sy in enumerate(L.systems):
        size = len(sy.staves)
        if size == K:
            places = list(range(K))
            if not all(allowed(n, i, i) for i in range(K)):
                return None, "a staff of a whole system stands where it may not be the roster's instrument"
        else:
            if own[n] and not pair:
                return None, "a system braces two staves, and the roster has no two-staff part"

            def unary(i: int, p: int) -> float:
                # what the staff says of its place: whether it may stand there at all, and its opening clef against
                # the one the page's whole systems show there (or the caller read there)
                if not allowed(n, i, p):
                    return -math.inf
                if opens[n][i] and open_place[p]:
                    return W.ALIGN_CLEF
                if opens[n][i] and t_clef[p]:
                    if told[p]:
                        return W.ALIGN_CLEF * _stated_clef_says(opens[n][i], t_clef[p])
                    return W.ALIGN_CLEF if opens[n][i] == t_clef[p] else -W.ALIGN_CLEF
                return 0.0

            def link(i: int, q: int, p: int) -> float:
                """Staff i - 1 at place q, staff i at place p."""
                v = 0.0
                if i - 1 in own[n] and p != q + 1:        # a brace's two staves placed apart
                    v -= W.ALIGN_BRACE
                elif i - 1 in own[n] and q in pair:       # ... or on a two-staff part
                    v += W.ALIGN_BRACE
                if q in pair and p != q + 1:              # a two-staff part's lower staff hidden
                    v -= W.ALIGN_BRACE
                if p - 1 in pair and q != p - 1:          # its upper staff hidden
                    v -= W.ALIGN_BRACE
                return v

            # the best fit by evidence alone, and in how many ways it is reached: with nothing to tell two fits
            # apart there is no fit (write._align_systems takes the top one; a roster's page may not)
            best = [[-math.inf] * K for _ in range(size)]
            ways = [[0] * K for _ in range(size)]
            back = [[-1] * K for _ in range(size)]
            for p in range(K - size + 1):
                best[0][p], ways[0][p] = unary(0, p) - (W.ALIGN_BRACE if p - 1 in pair else 0.0), 1
            for i in range(1, size):
                for p in range(i, K - size + i + 1):
                    for q in range(i - 1, p):
                        v = best[i - 1][q] + link(i, q, p)
                        if v > best[i][p] + EPS:
                            best[i][p], ways[i][p], back[i][p] = v, ways[i - 1][q], q
                        elif abs(v - best[i][p]) <= EPS:
                            ways[i][p] += ways[i - 1][q]
                    best[i][p] += unary(i, p)
            ends = {p: best[size - 1][p] - (W.ALIGN_BRACE if p in pair else 0.0) for p in range(size - 1, K)}
            top = max(ends.values())
            if top == -math.inf:
                return None, f"a system of {size} staves that may stand nowhere in the roster"
            fits = [p for p, v in ends.items() if abs(v - top) <= EPS]
            if sum(ways[size - 1][p] for p in fits) != 1:
                return None, f"a system of {size} staves fits the roster's {K} more than one way"
            p = fits[0]
            places = [p]
            for i in range(size - 1, 0, -1):
                p = back[i][p]
                places.append(p)
            places.reverse()
        for i in own[n]:                                  # a brace joins the two staves of one two-staff part
            if places[i] not in pair or places[i + 1] != places[i] + 1:
                return None, "a brace joins two staves that are not one two-staff part of the roster"
        place.append({p: k for p, k in zip(places, sy.staves)})
    return (plans, place), None
