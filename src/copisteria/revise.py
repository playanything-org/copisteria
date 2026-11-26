# Modified by Play Anything (https://github.com/playanything-org/copisteria): see README.md.
"""Structure from the readings: bar lines the detector missed, found because the music says a bar is there.

The front end cuts bars from what it sees -- a detected bar line, a stroke over the staff's height -- and on one
staff a stroke alone is not enough (a stem is one too). A bar line the detector missed then leaves one bar holding
two bars of music. Once the page is read, the readings say so: in such a bar the main voice's durations add up to
twice (or more) the meter in effect, and the running sum reaches exactly one bar between two notes, where a stroke
over the staff's full height stands that no note's stem explains. There the bar line is added, and the page is read
again with it (``read_revised``). Ink-path pages only (the cuts are the joint segmentation's).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction

OVERFULL = Fraction(7, 4)          # a voice holding this many bars of the meter is a candidate for a missed line
STROKE = 0.9                       # share of the staff's height a bar-line stroke covers


def _bar_len(time) -> Fraction | None:
    try:
        a, b = (int(v) for v in str(time).split("/"))
        return Fraction(4 * a, b)
    except (ValueError, ZeroDivisionError):
        return None


def missed_bar_lines(rd, gray) -> dict:
    """{top staff of a system: [x of each bar line to add]}."""
    import numpy as np

    from . import front
    from .write import _duration, _type_of
    L = rd.layout
    if not L.ink or gray is None:
        return {}
    mask = front.ink_mask(np.asarray(gray))
    tok = rd.tok
    # the meter in effect per bar column, carried over the page from the bars that print one
    time_at: dict = {}
    cur = None
    for n, sy in enumerate(L.systems):
        ncol = 1 + max((max(c) for c in sy.columns if c), default=0)
        for col in range(ncol):
            for pos, k in enumerate(sy.staves):
                st = L.staves[k]
                for j, i in enumerate(st.bars):
                    if sy.columns[pos][j] == col and i in tok and tok[i].get("time", {}).get("v") not in (None, "other"):
                        cur = tok[i]["time"]["v"]
            time_at[(n, col)] = cur
    out: dict = defaultdict(list)
    for n, sy in enumerate(L.systems):
        S = [L.staves[k] for k in sy.staves]
        for pos, k in enumerate(sy.staves):
            st = L.staves[k]
            sp = st.sp
            for j, bi in enumerate(st.bars):
                blen = _bar_len(time_at.get((n, sy.columns[pos][j])))
                if not blen:
                    continue
                bx0, _, bx1, _ = L.syms[bi].box
                evs = defaultdict(list)
                heads = []
                for s in L.syms:
                    if s.staff != k or s.bar != j or s.i not in tok or tok[s.i]["real"]["p"] < 0.5:
                        continue
                    r = tok[s.i]
                    c = r["cls"]["v"]
                    fam = c.split("_", 1)[0]
                    if fam not in ("note", "rest"):
                        continue
                    if fam == "note":
                        heads.append(s.box)
                        if r.get("chord", 0) > 0.5 or r.get("grace", {}).get("v", "none") != "none":
                            continue
                    d = _duration(_type_of(c), r["dots"]["v"], r["tup"]["v"])
                    evs[r["voice"]["v"]].append((s.cx, s.box, d))
                for v, es in evs.items():
                    es.sort(key=lambda e: e[0])
                    if sum(e[2] for e in es) < OVERFULL * blen:
                        continue
                    cum = Fraction(0)
                    for a, b in zip(es, es[1:]):
                        cum += a[2]
                        if cum % blen:
                            continue
                        # the running sum fills a whole number of bars here: a stroke over the staff's height between
                        # the two notes, clear of every note head's column (a stem stands next to its head)
                        lo, hi = int(a[1][2]), int(b[1][0])
                        xs = [x for x in range(lo, hi + 1)
                              if front._ink_run(mask, x, st.y0 + 1, st.y1 - 1, reach=1) >= STROKE
                              and not any(h[0] - 0.4 * sp <= x <= h[2] + 0.4 * sp for h in heads)]
                        if not xs:
                            continue
                        x = sorted(xs)[len(xs) // 2]
                        others = [o for o in S if o is not st]
                        if not others or sum(front._ink_run(mask, x, o.y0 + 1, o.y1 - 1, reach=2) >= STROKE
                                             for o in others) >= len(others) / 2:
                            out[sy.staves[0]].append(float(x))
    return {k: sorted(set(v)) for k, v in out.items()}


def read_revised(dets, width, height, model, device="cpu", attention=True, image=None, **kw):
    """read.read, then again with the bar lines the first reading shows were missed (if any)."""
    from . import read
    rd = read.read(dets, width, height, model, device=device, attention=attention, image=image, **kw)
    cuts = missed_bar_lines(rd, image)
    if cuts:
        rd = read.read(dets, width, height, model, device=device, attention=attention, image=image, cuts=cuts, **kw)
        rd.revised = cuts
    if rd.layout.staff_trials:
        def read_staff(base, layout):
            from .read import Reading

            track = len(layout.staves) - 1
            feats = kw.get("feats") if model.feat_dim else None
            decoded = read.read_layout(layout, model, device=device, attention=attention, feats=feats,
                                       system=layout.staves[track].system)
            # Existing tracks keep the base's decoding. Only the extra track has a new reading.
            tok = {i: deepcopy(t) for i, t in base.tok.items() if layout.syms[i].staff != track}
            tok.update((i, t) for i, t in decoded.tok.items() if layout.syms[i].staff == track)
            attn = {i: a for i, a in base.attn.items() if layout.syms[i].staff != track}
            attn.update((i, a) for i, a in decoded.attn.items() if layout.syms[i].staff == track)
            return Reading(layout=layout, tok=tok, attn=attn)
        rd.read_staff = read_staff
    return rd


def read_for_opening_trials(read, *args, **kwargs):
    """Read as before the opening recovery. The writer can then try each proposed gap
    from these same inputs, including when a caller picks one of several readings."""
    from .front import OpeningJoins

    found = OpeningJoins()
    base = read(*args, **kwargs, opening_joins=found)
    if found.found or getattr(base, "read_staff", None) is not None:
        def trial(base, write, base_written, *, audit=None):
            accepted = frozenset()
            best = base, base_written
            partial = found.found - found.legacy
            best_facts = audit(base) if audit is not None else None

            def try_gaps(gaps, *, guarded):
                nonlocal accepted, best, best_facts
                original, (base_notes, _) = best
                note_ids = set().union(*base_notes)
                base_columns = [_note_columns(original.layout, ids) for ids in base_notes]
                try:
                    base_audit = _written_page(original.layout, best_facts) if guarded and best_facts is not None else None
                except TimeoutError:
                    return False
                for gap in sorted(gaps):
                    allowed = accepted | {gap}
                    try:
                        candidate = read(*args, **kwargs, opening_joins=OpeningJoins(allowed))
                        written = write(candidate)
                        candidate_notes, _ = written
                        if any(not old.issubset(new) for old, new in zip(base_notes, candidate_notes, strict=True)):
                            continue
                        columns = _note_columns(candidate.layout, note_ids)
                        if any(_merges_written_bars(before, columns) for before in base_columns):
                            continue
                        candidate_facts = audit(candidate) if best_facts is not None else None
                        if base_audit is not None and _regresses_written_page(
                                base_audit, _written_page(candidate.layout, candidate_facts)):
                            continue
                    except TimeoutError:
                        return False
                    except Exception:
                        continue
                    accepted, best, best_facts = allowed, (candidate, written), candidate_facts
                return True

            # The older openings already belong to the installed reader's output. Establish that
            # reading first, then compare each new partial opening to what the installed reader writes.
            completed = not found.legacy or try_gaps(found.legacy, guarded=False)
            if completed and partial:
                completed = try_gaps(partial, guarded=True)
            if completed and best_facts is not None and getattr(best[0], "read_staff", None) is not None:
                from .staff_recovery import recover
                return recover(*best, write, audit, best_facts)
            return best

        base.opening_trial = trial
    return base


def _note_columns(layout, note_ids):
    return {s.det: (s.staff, layout.column(s.staff, s.bar)) for s in layout.syms
            if s.det in note_ids and s.staff >= 0 and s.bar >= 0}


def _merges_written_bars(before, after):
    """A shared grid can combine two previous bars without cutting a single note.
    Preserve a written division between notes on the same original staff."""
    bars = {}
    for det, (staff, bar) in before.items():
        location = after.get(det)
        if location is None:
            return True
        slot = staff, location
        if bars.setdefault(slot, bar) != bar:
            return True
    return False


@dataclass(frozen=True)
class _WrittenStaff:
    part: int
    number: int
    count: int
    system: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class _WrittenVariant:
    plan: list[tuple[int, ...]]
    staves: dict[tuple[float, float], _WrittenStaff]
    notes: dict[tuple, Counter]
    occurrences: Counter[int]


def _written_page(layout, writes):
    from .front import System, _braced
    from .write import _braced_pairs

    tracks = [(staff.y0, staff.y1) for staff in layout.staves]
    physical = sorted(range(len(tracks)), key=lambda staff: tracks[staff])
    whole_page = System(physical)
    braces = set()
    for symbol in layout.syms:
        if symbol.cls == "grpSym_brace" and symbol.det >= 0:
            for i in _braced_pairs(layout, whole_page, [symbol]):
                upper, lower = physical[i:i + 2]
                if _braced([symbol], layout.staves[upper], layout.staves[lower]):
                    braces.add((tracks[upper], tracks[lower]))

    variants = []
    for facts in writes:
        plans = [tuple(plan.staves) for plan in facts["plans"]]
        slots = {slot: (part, number, len(plan)) for part, plan in enumerate(plans)
                 for number, slot in enumerate(plan, 1)}
        placed = {}
        for system, places in zip(layout.systems, facts["place"], strict=True):
            rows = tuple(tracks[k] for k in system.staves)
            for slot, k in places.items():
                placed[tracks[k]] = _WrittenStaff(*slots[slot], rows)
        notes = defaultdict(Counter)
        occurrences = Counter()
        for det, staff, number, bar, *sounding in facts["trial_notes"]:
            notes[(det, tracks[staff], number, bar)][tuple(sounding)] += 1
            occurrences[det] += 1
        variants.append(_WrittenVariant(plans, placed, notes, occurrences))
    return braces, variants


def _regresses_written_page(before, after):
    old_braces, prior = before
    new_braces, proposed = after
    for old, new in zip(prior, proposed, strict=True):
        if _regresses_written_parts(old.staves, new.staves, old_braces, new_braces):
            return True
        if any(old.notes[key] - new.notes[key] for key in old.notes.keys() & new.notes.keys()):
            return True
        # Repeating a figure into other bars preserves the set of detector IDs while adding music.
        if any(new.occurrences[det] > count for det, count in old.occurrences.items()):
            return True
    if len(prior) == 2:
        old_primary, old_placed = prior
        primary, placed = proposed
        if old_primary.plan == old_placed.plan and primary.plan == placed.plan:
            # A larger joined system can become the template for placement across the page. It must
            # not send a previously agreed staff in an untouched system to a different part or staff.
            old_a, old_b, new_a, new_b = (v.staves for v in (*prior, *proposed))
            for track in old_a.keys() & old_b.keys() & new_a.keys() & new_b.keys():
                if (old_a[track] == old_b[track] and old_a[track].system == new_a[track].system and
                        new_a[track] != new_b[track]):
                    return True
    return False


def _regresses_written_parts(old, new, old_braces, new_braces):
    slots = {(place.system, place.part, place.number): track for track, place in new.items()}
    original_parts, proposed_parts = {}, {}
    for track in old.keys() & new.keys():
        a, b = old[track], new[track]
        upper = slots.get((b.system, b.part, b.number - 1))
        if a.count == 1 and b.number > 1 and (upper, track) not in new_braces:
            return True
        if a.system == b.system:
            # Part numbers can change. The physical staves sharing one part across unchanged
            # systems must still belong together, and separate parts must stay separate.
            if original_parts.setdefault(a.part, b.part) != b.part or proposed_parts.setdefault(b.part, a.part) != a.part:
                return True
    for upper, lower in old_braces:
        if upper in old and lower in old and old[upper].part == old[lower].part:
            if upper in new and lower in new and new[upper].part != new[lower].part:
                return True
    return False
