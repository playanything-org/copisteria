# Modified by Play Anything (https://github.com/playanything-org/copisteria): see README.md.
from dataclasses import replace
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from copisteria import front, links, write
from copisteria.read import Reading
from copisteria.revise import read_for_opening_trials
from tests.test_front_write import _tokens


def _orchestral_fragments(monkeypatch, sizes=(2, 2, 2), brace=(2, 3), clefs=None, notation=None):
    tops = [60 + 190 * staff for staff in range(sum(sizes))]
    width, height = 640, tops[-1] + 90
    columns = (120, 260, 400, 540, 600)
    upper = sum(sizes[:-1]) - 1
    lower = upper + 1
    ink = np.zeros((height, width), dtype=bool)
    syms, staves = [], []
    for staff, y in enumerate(tops):
        ink[y:y + 41:10, 120:600] = True
        bars = []
        for bar, (left, right) in enumerate(zip(columns, columns[1:])):
            index = len(syms)
            bars.append(index)
            syms.append(front.Sym(index, "measure", {}, .9, 1, (left, y, right, y + 40),
                                  staff=staff, bar=bar, det=index))
        # The distinct note x positions keep each physical staff observable in the written part.
        index, x = len(syms), 150 + 10 * staff
        syms.append(front.Sym(index, "note_quarter", {}, .9, 1, (x, y + 16, x + 7, y + 24),
                              staff=staff, bar=0, det=index))
        if staff == 0 and notation == "tie":
            index = len(syms)
            syms.append(front.Sym(index, "note_quarter", {}, .9, 1, (280, y + 16, 287, y + 24),
                                  staff=staff, bar=1, det=index))
        start = 101 if staff == upper else 120
        staves.append(front.Staff(bars, 10, y, y + 40, start, 600, line_x0=start))
    ink[tops[0]:tops[-1] + 41, 120] = True
    barlines = []
    for x in columns[1:]:
        ink[tops[upper]:tops[upper] + 41, x] = ink[tops[lower]:tops[lower] + 41, x] = True
        barlines.append(front.Sym(len(barlines), "barLine_single", {}, .9, 1,
                                  (x, tops[upper], x + 1, tops[upper] + 40)))
    ink[tops[upper]:tops[lower] + 41, 400] = True
    if brace:
        index = len(syms)
        syms.append(front.Sym(index, "grpSym_brace", {}, .9, 1,
                              (104, tops[brace[0]], 112, tops[brace[1]] + 40), det=index))
    monkeypatch.setattr(links, "load", lambda: None)

    def read(*, opening_joins):
        # The column boxes below are the reader's recovered bar grid. The
        # opening search sees only the barline detections through this gap.
        before_columns = [replace(staff, bars=[]) for staff in staves[upper:upper + 3]]
        linked, _ = front._links(barlines, before_columns, ink, opening_joins=opening_joins)
        assert linked[1]
        groups, start = [], 0
        for size in sizes:
            groups.append(list(range(start, start + size)))
            start += size
        if linked[0]:
            groups[-2].extend(groups.pop())
        placed = [None] * len(staves)
        systems = []
        for n, group in enumerate(groups):
            systems.append(front.System(group, [list(range(len(columns) - 1)) for _ in group]))
            for pos, staff in enumerate(group):
                placed[staff] = replace(staves[staff], system=n, pos=pos)
        layout = front.Layout(syms, placed, systems, 10, width, height)
        tokens = _tokens(layout, lambda s: clefs[s.staff] if clefs else "G2")
        for s in syms:
            if s.staff != 0:
                continue
            if notation == "pitch":
                if s.fam == "measure":
                    tokens[s.i]["key"]["v"] = 7 if linked[0] else -1
                elif s.fam == "note":
                    tokens[s.i]["alter"]["v"] = "key"
            if notation == "tie" and s.fam == "note" and s.bar == 0:
                tokens[s.i]["tie"] = 0.0 if linked[0] else 1.0
            if notation == "duration" and s.fam == "note" and linked[0]:
                tokens[s.i]["cls"]["v"] = "note_eighth"
        return Reading(layout=layout, tok=tokens)

    return read


def _try_opening(read, observe, *, align=False):
    baseline = read_for_opening_trials(read)
    assert baseline.opening_trial is not None
    facts, attempts = {}, []

    def written(reading):
        nonlocal facts
        ids, facts = set(), {}
        xml = write.write(reading, align=align, facts=facts, note_ids=ids, trial_notes=True)
        root = ET.fromstring(xml.split("\n", 2)[2])
        attempts.append(observe(root, facts))
        return (ids,), xml

    base = written(baseline)
    selected, result = baseline.opening_trial(baseline, written, base, audit=lambda _: (facts,))
    assert result[0] == base[0]
    if selected is baseline:
        assert result == base
    return selected is baseline, attempts


def _parts_with_notes(score, bar):
    return [part.attrib["id"] for part in score.findall("part")
            if part.find(f"measure[@number='{bar}']/note/pitch/step") is not None]


def test_recovered_opening_does_not_absorb_another_fragments_independent_staff(monkeypatch):
    def observe(root, facts):
        staves = [n.findtext("staff", "1") for n in root.findall("part[1]/measure[1]/note")
                  if n.find("pitch") is not None]
        return [len(plan.staves) for plan in facts["plans"]], staves

    fallback, attempts = _try_opening(_orchestral_fragments(monkeypatch), observe)
    assert attempts == [([1, 1], ["1"]), ([2, 1, 1], ["1", "2"])]
    assert fallback


def test_recovered_opening_keeps_a_part_together_across_unaffected_systems(monkeypatch):
    read = _orchestral_fragments(monkeypatch, sizes=(2, 2, 2, 2), brace=None,
                                 clefs=("G2", "F4", "G2", "C4", "G2", "F4", "C3", "C4"))

    def observe(score, facts):
        assert all(len(plan.staves) == 1 for plan in facts["plans"])
        return [_parts_with_notes(score, bar) for bar in (1, 5)]

    fallback, attempts = _try_opening(read, observe, align=True)
    assert attempts == [[["P1", "P2"], ["P1", "P2"]], [["P1", "P2"], ["P1", "P4"]]]
    assert fallback


def test_recovered_opening_can_group_the_two_staves_joined_by_their_own_brace(monkeypatch):
    read = _orchestral_fragments(monkeypatch, sizes=(3, 3, 2), brace=(6, 7))
    fallback, attempts = _try_opening(read, lambda _, facts: [len(plan.staves) for plan in facts["plans"]])
    assert attempts == [[1, 1, 1], [1, 1, 1, 2]]
    assert not fallback


def test_recovered_opening_keeps_the_staves_of_a_written_braced_part_together(monkeypatch):
    read = _orchestral_fragments(monkeypatch, brace=(0, 1))
    fallback, attempts = _try_opening(read, lambda score, _: _parts_with_notes(score, 1))
    assert attempts == [["P1"], ["P1", "P2"]]
    assert fallback


@pytest.mark.parametrize("notation, unchanged, changed", (("pitch", "-1", "1"),
                                                         ("tie", ["start", "stop"], []),
                                                         ("duration", "quarter", "eighth")))
def test_recovered_opening_preserves_the_music_on_the_same_staff_and_bar(monkeypatch, notation, unchanged, changed):
    read = _orchestral_fragments(monkeypatch, brace=None, notation=notation)

    def observe(root, _):
        if notation == "pitch":
            return root.findtext("part[1]/measure[1]/note/pitch/alter")
        if notation == "duration":
            return root.findtext("part[1]/measure[1]/note/type")
        return [tie.get("type") for n in (1, 2) for tie in root.findall(f"part[1]/measure[{n}]/note/tie")]

    fallback, attempts = _try_opening(read, observe)
    assert attempts == [unchanged, changed]
    assert fallback


def test_a_partial_opening_must_keep_notes_recovered_by_the_installed_reader():
    staves = [front.Staff([], 10, y, y + 40, 120, 600) for y in (60, 250, 440)]

    def read(*, opening_joins):
        partial = opening_joins.accepts(*staves[:2], partial=True)
        legacy = opening_joins.accepts(*staves[1:])
        return SimpleNamespace(legacy=legacy, partial=partial, layout=SimpleNamespace(syms=[]))

    attempted = []

    def written(reading):
        attempted.append((reading.legacy, reading.partial))
        ids = {1, 2} if reading.legacy and not reading.partial else {1}
        return (ids,), reading

    baseline = read_for_opening_trials(read)
    selected, result = baseline.opening_trial(baseline, written, written(baseline))
    assert attempted == [(False, False), (True, False), (True, True)]
    assert selected.legacy and not selected.partial
    assert result[0] == ({1, 2},)


def test_recovered_opening_does_not_copy_a_written_note_into_another_bar(monkeypatch):
    original = _orchestral_fragments(monkeypatch, brace=None)

    def read(*, opening_joins):
        reading = original(opening_joins=opening_joins)
        accepted = len(reading.layout.systems) == 2
        syms = reading.layout.syms
        i = len(syms)
        repeat = front.Sym(i, "mRpt", {}, .9, 1, (315, 76, 335, 84), staff=0, bar=1, det=i)
        reading.layout = replace(reading.layout, syms=[*syms, repeat])
        reading.tok[i] = {"real": {"p": float(accepted)}, "cls": {"v": "mRpt"}}
        return reading

    def observe(score, _):
        return [bool(score.findall(f"part[1]/measure[{bar}]/note/pitch")) for bar in (1, 2)]

    fallback, attempts = _try_opening(read, observe)
    assert attempts == [[True, False], [True, True]]
    assert fallback
