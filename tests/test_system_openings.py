# Modified by Play Anything (https://github.com/playanything-org/copisteria): see README.md.
from math import sin
from types import SimpleNamespace

import numpy as np
import pytest
from copisteria import front, links
from copisteria.revise import read_for_opening_trials


def _staff(mask, top, left, right):
    mask[top:top + 41:10, left:right] = True


def _linked(mask, a, b, symbols=()):
    return front._links(list(symbols), [a, b], mask)[0] == [True]


def test_solid_opening_at_both_measured_starts():
    mask = np.zeros((420, 640), dtype=bool)
    _staff(mask, 60, 120, 600)
    _staff(mask, 250, 120, 600)
    mask[60:291, 120] = True
    a = front.Staff([], 10, 60, 100, 119, 600, line_x0=120)
    b = front.Staff([], 10, 250, 290, 101, 600, line_x0=101)
    assert _linked(mask, a, b)


def test_opening_rechecks_staff_lines_locally_on_a_distorted_scan():
    mask = np.zeros((420, 640), dtype=bool)
    _staff(mask, 101, 120, 600)
    _staff(mask, 291, 120, 600)
    mask[101:332, 120] = True
    a = front.Staff([], 10, 60, 100, 540, 600, line_x0=540)
    b = front.Staff([], 10, 250, 290, 120, 600, line_x0=530)
    assert _linked(mask, a, b)


def test_interior_bars_use_the_staff_lines_at_each_bar():
    mask = np.zeros((420, 640), dtype=bool)
    _staff(mask, 108, 100, 600)
    _staff(mask, 250, 100, 600)
    a = front.Staff([], 10, 60, 100, 100, 260)
    b = front.Staff([], 10, 250, 290, 100, 600)
    bars = []
    for x in (300, 450):
        mask[108:291, x] = True
        for top, bottom in ((88, 108), (250, 290)):
            bars.append(front.Sym(len(bars), 'barLine_single', {}, 0.9, 1, (x - 1, top, x + 1, bottom)))
    assert _linked(mask, a, b, bars)


def test_new_search_needs_contacts_at_both_staves():
    mask = np.zeros((420, 640), dtype=bool)
    _staff(mask, 60, 120, 600)
    _staff(mask, 250, 128, 600)
    mask[60:291, 120] = True
    a = front.Staff([], 10, 60, 100, 119, 600, line_x0=120)
    b = front.Staff([], 10, 250, 290, 101, 600, line_x0=101)
    assert not _linked(mask, a, b)


def test_a_bowed_binding_band_can_touch_both_staff_rules_without_joining_them():
    mask = np.zeros((420, 640), dtype=bool)
    _staff(mask, 60, 120, 600)
    _staff(mask, 250, 120, 600)
    for y in range(45, 299):
        shift = round(4 * sin(y / 50))
        mask[y, 77 + shift:129 + shift] = True
    a = front.Staff([], 10, 60, 100, 120, 600)
    b = front.Staff([], 10, 250, 290, 120, 600)
    assert not _linked(mask, a, b)


def test_each_recovered_join_must_keep_the_baseline_note_ids_even_at_equal_counts():
    mask = np.zeros((810, 640), dtype=bool)
    tops = (60, 250, 440, 630)
    for top in tops:
        _staff(mask, top, 120, 600)
    mask[60:291, 120] = True
    mask[440:671, 120] = True
    staves = [front.Staff([], 10, top, top + 40, start, 600, line_x0=measured)
              for top, start, measured in zip(tops, (119, 101, 119, 101), (120, 101, 120, 101))]

    def read(*, opening_joins=None):
        linked, _ = front._links([], staves, mask, opening_joins=opening_joins)
        return SimpleNamespace(linked=linked, layout=SimpleNamespace(syms=[]))

    def write(reading):
        first, _, last = reading.linked
        ids = {8 if first else 4, 5}
        if last:
            ids.add(9)
        return (ids,), reading.linked

    old = read()
    assert old.linked == [True, False, True]
    base = read_for_opening_trials(read)
    assert base.linked == [False, False, False]
    selected, ((notes,), linked) = base.opening_trial(base, write, write(base))
    assert selected.linked == linked == [False, False, True]
    assert notes == {4, 5, 9}


def test_a_recovered_join_cannot_merge_written_bars_even_if_all_notes_fit():
    upper = front.Staff([], 10, 60, 100, 120, 600)
    lower = front.Staff([], 10, 250, 290, 120, 600)

    def read(*, opening_joins):
        joined = opening_joins.accepts(upper, lower)
        notes = [SimpleNamespace(det=8, staff=0, bar=0),
                 SimpleNamespace(det=9, staff=0, bar=0 if joined else 1)]
        layout = SimpleNamespace(syms=notes, column=lambda staff, bar: bar)
        return SimpleNamespace(joined=joined, layout=layout)

    def written(reading):
        return ({8, 9},), reading.joined

    baseline = read_for_opening_trials(read)
    reading, ((ids,), joined) = baseline.opening_trial(baseline, written, written(baseline))
    assert reading is baseline
    assert ids == {8, 9} and not joined


@pytest.mark.parametrize('crossing', (120, 400))
@pytest.mark.parametrize('start', (101, 107))
def test_a_stroke_across_staff_rules_cannot_open_their_system(monkeypatch, crossing, start):
    mask = np.zeros((540, 640), dtype=bool)
    _staff(mask, 60, 120, 600)
    _staff(mask, 250, start, 600)
    _staff(mask, 440, start, 600)
    mask[53:297, 120] = True  # A narrow binding shadow reaches both staves.
    mask[250:481, start] = True
    barlines = []
    for x in (crossing, 260, 440, 540, 599):
        mask[60:101, x] = mask[250:291, x] = True
        barlines.append(front.Sym(len(barlines), 'barLine_single', {}, 0.9, 1, (x, 60, x + 1, 100)))
    mask[60:291, crossing] = True
    upper = front.Staff([], 10, 60, 100, 119, 600, line_x0=120)
    lower = front.Staff([], 10, 250, 290, start, 600, line_x0=start)
    bass = front.Staff([], 10, 440, 480, start, 600, line_x0=start)
    monkeypatch.setattr(links, 'load', lambda: None)
    offers = front.OpeningJoins()

    assert front._links(barlines, [upper, lower, bass], mask, opening_joins=offers)[0] == [False, True]
    assert not offers.found


def test_a_shadow_across_two_solo_systems_has_no_established_neighbor(monkeypatch):
    mask = np.zeros((525, 640), dtype=bool)
    for y in (60, 250, 440):
        _staff(mask, y, 120, 600)
    mask[53:297, 120] = True
    bars = [front.Sym(i, 'barLine_single', {}, 0.9, 1, (x, 60, x + 1, 100))
            for i, x in enumerate((120, 260, 400, 540, 599))]
    staves = [front.Staff([], 10, y, y + 40, x, 600, line_x0=x)
              for y, x in ((60, 107), (250, 120), (440, 120))]
    monkeypatch.setattr(links, 'load', lambda: None)
    offers = front.OpeningJoins()

    assert front._links(bars, staves, mask, opening_joins=offers)[0] == [False, False]
    assert not offers.found


@pytest.mark.parametrize('objection', (None, 'note', 'bar', 'timeout', 'writer'))
def test_a_partial_bar_connection_can_still_offer_an_opening_trial(monkeypatch, objection):
    mask = np.zeros((540, 640), dtype=bool)
    _staff(mask, 60, 120, 600)
    _staff(mask, 250, 120, 600)
    _staff(mask, 440, 120, 600)
    mask[60:481, 120] = True
    # The upper staff's estimated start is left of its rules, so the opening
    # lies past the earlier search. Only one interior barline crosses the gap.
    barlines = []
    for x in (260, 400, 540, 599):
        mask[60:101, x] = mask[250:291, x] = True
        barlines.append(front.Sym(len(barlines), 'barLine_single', {}, 0.9, 1, (x, 60, x + 1, 100)))
    mask[60:291, 400] = True
    upper = front.Staff([], 10, 60, 100, 101, 600, line_x0=101)
    lower = front.Staff([], 10, 250, 290, 119, 600, line_x0=120)
    bass = front.Staff([], 10, 440, 480, 120, 600, line_x0=120)
    monkeypatch.setattr(links, 'load', lambda: None)
    writes = []

    def read(*, opening_joins):
        linked, _ = front._links(barlines, [upper, lower, bass], mask, opening_joins=opening_joins)
        assert linked[1]
        joined = linked[0]
        notes = [SimpleNamespace(det=1, staff=0, bar=0),
                 SimpleNamespace(det=2, staff=0, bar=0 if joined and objection == 'bar' else 1)]
        layout = SimpleNamespace(syms=notes, column=lambda _staff, bar: bar)
        return SimpleNamespace(joined=joined, layout=layout)

    def write(reading):
        writes.append(reading.joined)
        if reading.joined and objection == 'timeout':
            raise TimeoutError()
        if reading.joined and objection == 'writer':
            raise ValueError()
        ids = {1, 3} if reading.joined and objection == 'note' else {1, 2}
        return (ids, ids), reading.joined

    base = read_for_opening_trials(read)
    assert not base.joined
    selected, (ids, joined) = base.opening_trial(base, write, write(base))
    assert writes == [False, True]
    assert ids == ({1, 2}, {1, 2})
    assert selected.joined == joined == (objection is None)
    if objection:
        assert selected is base
