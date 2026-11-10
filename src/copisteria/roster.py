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
"""
from __future__ import annotations

import math
from collections import Counter

from .read import Reading

EPS = 1e-9             # two fits of a system to the roster score the same


def plan(rd: Reading, counts: list[int]) -> tuple[tuple[list, list[dict[int, int]]] | None, str | None]:
    """((plans, place), None) for a page the roster explains: the roster's parts, and per system place -> staff
    as write._align_systems gives it; else (None, why not)."""
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
    braces = [s for s in L.syms if W._real(rd, s.i) and W._cls(rd, s.i) == "grpSym_brace"]
    own = [W._braced_pairs(L, sy, braces) for sy in L.systems]
    clefs = [[W._opening_clef(rd, k) for k in sy.staves] for sy in L.systems]
    # what stands at each place: the opening clef most of the page's whole systems show there
    whole = [n for n, sy in enumerate(L.systems) if len(sy.staves) == K]
    t_clef: list = []
    for p in range(K):
        seen = Counter(clefs[n][p] for n in whole if clefs[n][p])
        t_clef.append(seen.most_common(1)[0][0] if seen else None)
    place = []
    for n, sy in enumerate(L.systems):
        size = len(sy.staves)
        if size == K:
            places = list(range(K))
        else:
            if own[n] and not pair:
                return None, "a system braces two staves, and the roster has no two-staff part"

            def unary(i: int, p: int) -> float:
                # what the staff says of its place: its opening clef against the one the page's whole systems show there
                if clefs[n][i] and t_clef[p]:
                    return W.ALIGN_CLEF if clefs[n][i] == t_clef[p] else -W.ALIGN_CLEF
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
