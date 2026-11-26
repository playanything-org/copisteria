# Added by Play Anything (https://github.com/playanything-org/copisteria): see README.md.
"""Try an unused five-line track against the page as the writer already reads it."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from math import ceil

from . import front


MIN_WIDTH = .6
STAFF_MUSIC = frozenset(("note", "rest", "mRest"))
LOCAL_SYMBOLS = STAFF_MUSIC | {"accid", "artic", "barLine", "beam", "clef", "dots", "flag", "keySig", "meterSig", "stem"}


@dataclass
class Proposal:
    staff: front.Staff
    reference: int
    members: tuple[int, ...]


def _overlap(a, b):
    return min(a.x1, b.x1) - max(a.x0, b.x0)


def _on_row(symbol, staff):
    return (staff.x0 - staff.sp <= symbol.cx <= staff.x1 + staff.sp and
            staff.y0 - .6 * staff.sp <= symbol.cy <= staff.y1 + .6 * staff.sp)


def _staff_stroke(mask, staff, x):
    pad = ceil(.55 * staff.sp)
    return any(front._ink_run(mask, xx, staff.y0 + 1, staff.y1 - 1, reach=1) >= .9
               for xx in range(round(x) - pad, round(x) + pad + 1))


def _six_lines(mask, staff):
    # The line finder will take five out of a tablature staff. A sixth line at either edge
    # counts only if it runs along the body of the candidate, clear of the glyphs at the opening.
    left, right = round(staff.x0 + 3 * staff.sp), round(staff.x1 - staff.sp)
    for y in (staff.y0 - staff.sp, staff.y1 + staff.sp):
        top, bottom = max(0, round(y - .18 * staff.sp)), min(mask.shape[0], round(y + .18 * staff.sp) + 1)
        if bottom > top and right > left and mask[top:bottom, left:right].any(axis=0).mean() >= MIN_WIDTH:
            return True
    return False


def _attached(mask, staff, other, syms):
    upper, lower = sorted((staff, other), key=lambda st: st.mid)
    if front._braced([s for s in syms if s.cls == "grpSym_brace" and s.det >= 0], upper, lower):
        return True
    # Joining the end of a system requires a stroke contacting the rules at both ends of the
    # gap. A nearby system's identical bar positions are not a reason to give it this staff.
    left = max(upper.line_x0 or upper.x0, lower.line_x0 or lower.x0)
    sp = min(staff.sp, other.sp)
    return any(front._ink_run(mask, x, upper.y1, lower.y0) >= .9 and
               front._meets_staves(mask, x, upper, lower)
               for x in range(round(left - sp), round(left + 2 * sp) + 1))


def _reference(layout, staff, mask):
    neighbors = [k for k, st in enumerate(layout.staves)
                 if not st.one_line and _overlap(st, staff) >= MIN_WIDTH * (st.x1 - st.x0)
                 and .7 * st.sp <= staff.sp <= 1.35 * st.sp
                 and abs(staff.mid - st.mid) <= 16 * staff.sp]
    above = max((k for k in neighbors if layout.staves[k].mid < staff.mid),
                key=lambda k: layout.staves[k].mid, default=None)
    below = min((k for k in neighbors if layout.staves[k].mid > staff.mid),
                key=lambda k: layout.staves[k].mid, default=None)
    adjacent = [k for k in (above, below) if k is not None]
    if len(adjacent) < 2 or layout.staves[above].system != layout.staves[below].system:
        adjacent = [k for k in adjacent if _attached(mask, staff, layout.staves[k], layout.syms)]
        if len(adjacent) != 1:
            return None
    for k in sorted(adjacent, key=lambda index: abs(layout.staves[index].mid - staff.mid)):
        st = layout.staves[k]
        columns = layout.systems[st.system].columns[st.pos]
        ncol = 1 + max((max(row) for row in layout.systems[st.system].columns if row), default=-1)
        if not ncol or columns != list(range(ncol)):
            continue
        boundaries = [layout.syms[i].box[2] for i in st.bars]
        if sum(_staff_stroke(mask, staff, x) for x in boundaries) >= min(2, ncol):
            return k
    return None


def proposals(layout, tracks, mask):
    """A track can supply only a seat in a system whose bar grid it follows."""
    result = []
    for track in sorted(tracks, key=lambda tr: -sum(line.score for line in tr.lines)):
        left, right = max(line.x0 for line in track.lines), min(line.x1 for line in track.lines)
        if right <= left:
            continue
        staff = front.Staff([], track.spacing, track.top, track.bottom, float(left), float(right), line_x0=float(left))
        if any(min(st.y1, staff.y1) > max(st.y0, staff.y0) and _overlap(st, staff) > 0
               for st in layout.staves):
            continue
        if any(abs(o.staff.mid - staff.mid) < 2 * staff.sp for o in result):
            continue
        ref = _reference(layout, staff, mask)
        if ref is None or _six_lines(mask, staff):
            continue
        local = [s for s in layout.syms if s.det >= 0 and _on_row(s, staff) and "inline_rest" not in s.attrs]
        music = [s for s in local if s.p >= .5 and s.fam in STAFF_MUSIC]
        if not music:
            continue
        clef = any(s.fam == "clef" and s.p >= .6 and s.cx - staff.x0 <= 9 * staff.sp for s in local)
        bars = any(s.fam == "measure" and s.p >= front.MEASURE_LOW_P for s in local)
        if not (clef or bars or len(music) >= 3):
            continue
        members = tuple(s.i for s in local if s.fam in LOCAL_SYMBOLS)
        result.append(Proposal(staff, ref, members))
    return result


def _install(layout, proposal, protected):
    """Preserve every existing bar box and column. Give the extra seat the system's bars."""
    ref = layout.staves[proposal.reference]
    sy = layout.systems[ref.system]
    track = len(layout.staves)
    staff = replace(proposal.staff, bars=[], system=ref.system,
                    pos=sum(layout.staves[i].mid < proposal.staff.mid for i in sy.staves))
    staves = [replace(st) for st in layout.staves] + [staff]
    syms = list(layout.syms)
    cuts = [layout.syms[i].box for i in ref.bars]
    for j, bounds in enumerate(cuts):
        existing = next((s for s in layout.syms if s.fam == "measure" and s.staff < 0 and s.det >= 0
                         and _on_row(s, staff) and abs(s.box[0] - bounds[0]) <= staff.sp
                         and abs(s.box[2] - bounds[2]) <= staff.sp), None)
        box = (bounds[0], staff.y0, bounds[2], staff.y1)
        if existing is not None:
            member = replace(existing, box=box, staff=track, bar=j)
            syms[member.i] = member
        else:
            member = front.Sym(i=len(syms), cls="measure", post={front.CLS_ID["measure"]: 1.0}, p=0., n=0,
                               box=box, staff=track, bar=j, synthetic=True)
            syms.append(member)
        staff.bars.append(member.i)
    for i in proposal.members:
        s = syms[i]
        if s.det not in protected:
            bar = front._band(syms, staff, s.box[2] if s.cls == "barLine_repeat_start" else s.cx)[0]
            syms[i] = replace(s, staff=track, bar=bar)
    systems = list(layout.systems)
    seats = sy.staves[:staff.pos] + [track] + sy.staves[staff.pos:]
    columns = sy.columns[:staff.pos] + [sy.columns[ref.pos]] + sy.columns[staff.pos:]
    for pos, i in enumerate(seats):
        staves[i].pos = pos
    systems[staff.system] = front.System(seats, columns)
    return replace(layout, staves=staves, syms=syms, systems=systems, staff_trials=[])


def _written_staffs(facts):
    positions = {slot: (n, k) for n, part in enumerate(facts["plans"]) for k, slot in enumerate(part.staves, 1)}
    return {staff: positions[slot] for system in facts["place"] for slot, staff in system.items()}


def _counted_bars(facts):
    return {column: n for column, n in facts["rest_columns"].items() if isinstance(n, int)}


def _guard(trial, before, after, recovered, sources):
    from .write import _braced_pairs

    for a, b in zip(before, after, strict=True):
        if a["columns"] != b["columns"] or _counted_bars(a) != _counted_bars(b):
            return False
        old, new = _written_staffs(a), _written_staffs(b)
        if recovered not in new or not old.keys() <= new.keys():
            return False
        forward, backward = {}, {}
        for track, (op, os) in old.items():
            np, ns = new[track]
            if os != ns or forward.setdefault(op, np) != np or backward.setdefault(np, op) != op:
                return False
        # A two-staff part can acquire only two adjacent physical staves with a detector brace
        # spanning exactly those two. Existing parts and written staff numbers keep their identity.
        part, number = new[recovered]
        if len(b["plans"][part].staves) != 1:
            system = trial.systems[trial.staves[recovered].system]
            pairs = [i for i in system.staves if i != recovered and new.get(i) == (part, 3 - number)]
            braces = [s for s in trial.syms if s.cls == "grpSym_brace" and s.det >= 0 and s.p >= front.GRP_MIN_P]
            if len(b["plans"][part].staves) != 2 or len(pairs) != 1:
                return False
            pos = min(trial.staves[recovered].pos, trial.staves[pairs[0]].pos)
            if {recovered, pairs[0]} != set(system.staves[pos:pos + 2]) or not any(
                    pos in _braced_pairs(trial, system, [s]) and
                    front._braced([s], *(trial.staves[i] for i in system.staves[pos:pos + 2])) for s in braces):
                return False
        old_notes = Counter(a["trial_notes"])
        other = Counter(note for note in b["trial_notes"] if note[1] != recovered)
        if old_notes != other:
            return False
        if Counter(a["trial_rests"]) != Counter(rest for rest in b["trial_rests"] if rest[1] != recovered):
            return False
        introduced_notes = [note[0] for note in b["trial_notes"] if note[1] == recovered]
        introduced_rests = [rest[0] for rest in b["trial_rests"] if rest[1] == recovered]
        if not (introduced_notes or introduced_rests):
            return False
        if (not all(sources.get(det) == "note" for det in introduced_notes) or
                not all(sources.get(det) in ("rest", "mRest") for det in introduced_rests) or
                len(set(introduced_notes)) != len(introduced_notes)):
            return False
    return True


def recover(base, written, write, audit, before):
    """Offer each unused staff separately. A rejected read leaves the original write intact."""
    if not before or not all("trial_rests" in facts for facts in before):
        return base, written
    protected = {row[0] for facts in before for field in ("trial_notes", "trial_rests") for row in facts[field]}
    for proposal in base.layout.staff_trials:
        sources = {s.det: s.fam for i in proposal.members if (s := base.layout.syms[i]).fam in STAFF_MUSIC
                   and s.det not in protected}
        if not sources:
            continue
        try:
            layout = _install(base.layout, proposal, protected)
            trial = base.read_staff(base, layout)
            candidate = write(trial)
            if _guard(layout, before, audit(trial), len(base.layout.staves), sources):
                return trial, candidate
        except (TimeoutError, MemoryError):
            return base, written
        except Exception:
            continue
    return base, written
