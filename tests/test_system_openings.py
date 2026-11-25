from math import sin

import numpy as np
from copisteria import front


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
