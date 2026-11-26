# Added by Play Anything (https://github.com/playanything-org/copisteria): see README.md.
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
import pytest

from copisteria import front, read, write
from copisteria.read import Reading
from copisteria.revise import read_for_opening_trials, read_revised
from tests.test_front_write import _tokens, det, staff_row


def scene(kind):
    image = Image.new("L", (660, 402), 255)
    ink = ImageDraw.Draw(image)
    bounds = (50, 250, 450, 608)
    outer = (60, 300)
    detections = []
    for top in outer:
        for y in range(top, top + 41, 10):
            ink.line((bounds[0], y, bounds[-1], y), fill=0)
        for x in bounds[1:]:
            ink.line((x, top, x, top + 40), fill=0)
            detections.append(det("barLine_single", (x - 1, top, x + 1, top + 40)))
        detections += staff_row(top, bounds)
        for x in (140, 340, 530):
            detections.append(det("note_quarter", (x, top + 16, x + 11, top + 24), staff_position=0))
    ink.line((bounds[0], outer[0], bounds[0], outer[-1] + 40), fill=0)

    mid = 180
    spacing, left, right = 10, bounds[0], bounds[-1]
    count = 5
    if kind == "tablature":
        count = 6
    elif kind == "chord diagram":
        left, right = 350, 450
    elif kind == "ossia":
        spacing = 5
    elif kind == "lyrics":
        count = 0
        ink.text((60, mid + 16), "The words go like this " * 4, fill=0)
    for k in range(count):
        y = mid + k * spacing
        ink.line((left, y, right, y), fill=0)
    if kind != "beam group":
        for x in bounds[1:]:
            if left <= x <= right:
                ink.line((x, mid, x, mid + (count - 1) * spacing), fill=0)
    if kind == "measure and track":
        detections += [det("measure", (bounds[0], mid, bounds[1], mid + 40), conf=.18)]
    detections += [det("clef_F4", (56, mid, 71, mid + 40), conf=.46 if kind == "measure and track" else .96)]
    detections += [det("note_quarter", (385, mid + 16, 395, mid + 24), staff_position=0)]
    if kind == "brace":
        detections += [det("grpSym_brace", (37, outer[0], 47, mid + 40))]
        ink.line(((47, outer[0]), (41, 72), (41, 128), (37, 140), (41, 152), (41, 208), (47, mid + 40)),
                 fill=0, width=2)
    return image, detections


def run(monkeypatch, kind, change=None):
    from copisteria import links

    image, detections = scene(kind)
    monkeypatch.setattr(links, "load", lambda: None)

    def tokens(layout, model, **kwargs):
        tok = _tokens(layout)
        for symbol in layout.syms:
            if symbol.staff >= 0 and symbol.fam not in ("note", "measure"):
                tok[symbol.i]["cls"]["v"] = symbol.cls
        return Reading(layout, tok)

    if hasattr(read, "read_layout"):
        monkeypatch.setattr(read, "read_layout", tokens)
    else:
        def read_base(dets, width, height, model, *, image, device, attention, **kw):
            return tokens(front.build(dets, width, height, image=image, **kw), model)
        monkeypatch.setattr(read, "read", read_base)
    if kind in ("finder only", "measure and track"):
        original = front._ink_staves_from

        def maps(page, gray, syms, spare, *tracks):
            # The box is too weak to create a row. The opening pass detects the lines
            # even if neither structural pass does. In the other case the cleaned pass sees them.
            if page is gray or kind == "measure and track":
                page = gray.copy()
                page[180:222, 51:] = 255
                return original(page, gray, syms, spare, *tracks)
            return original(gray, gray, syms, spare, *tracks)

        monkeypatch.setattr(front, "_ink_staves_from", maps)
    base = read_for_opening_trials(read_revised, detections, *image.size, SimpleNamespace(feat_dim=0), image=image)
    facts = {}

    def written(reading):
        nonlocal facts
        facts, ids = {}, set()
        text = write.write(reading, align=False, image=image, facts=facts, note_ids=ids, trial_notes=True)
        if reading is not base and change:
            change(reading, facts)
        return (ids,), text

    old = written(base)
    if base.opening_trial is None:
        return base, old, base, old
    chosen, result = base.opening_trial(base, written, old, audit=lambda _: (facts,))
    return base, old, chosen, result


@pytest.mark.parametrize("kind", ("spare", "finder only", "measure and track"))
def test_recover_rows_the_base_leaves_out(monkeypatch, kind):
    base, old, chosen, result = run(monkeypatch, kind)
    assert [len(s.staves) for s in base.layout.systems] == [2]
    assert chosen is not base
    assert chosen.layout.systems[0].staves == [0, 2, 1]
    assert old[0][0] < result[0][0]
    assert len(result[0][0] - old[0][0]) == 1
    assert [chosen.layout.bar_box(0, bar) for bar in range(3)] == [base.layout.bar_box(0, bar) for bar in range(3)]
    before, after = [ET.fromstring(xml).findall("part") for xml in (old[1], result[1])]
    for original, retained in zip(before, (after[0], after[2]), strict=True):
        retained.attrib["id"] = original.attrib["id"]
        assert ET.tostring(original) == ET.tostring(retained)


@pytest.mark.parametrize("kind", ("lyrics", "tablature", "chord diagram", "ossia", "beam group"))
def test_nonstaff_witness_keeps_the_base_musicxml(monkeypatch, kind):
    base, old, chosen, result = run(monkeypatch, kind)
    assert chosen is base
    assert result == old


@pytest.mark.parametrize("fault", ("loss", "staff", "measure", "system bars", "unbraced part"))
def test_reject_changes_to_written_music(monkeypatch, fault):
    def change(reading, facts):
        first, *rest = facts["trial_notes"]
        match fault:
            case "loss":
                facts["trial_notes"] = rest
            case "staff":
                facts["trial_notes"] = [(first[0], 1, *first[2:]), *rest]
            case "measure":
                facts["trial_notes"] = [(*first[:3], first[3] + 1, *first[4:]), *rest]
            case "system bars":
                facts["columns"] = facts["columns"][:-1]
            case "unbraced part":
                facts["plans"][0].staves.extend(facts["plans"].pop(1).staves)

    base, old, chosen, result = run(monkeypatch, "spare", change)
    assert base.opening_trial is not None
    assert chosen is base
    assert result == old


def test_new_second_staff_needs_the_actual_brace_detection(monkeypatch):
    base, _, chosen, result = run(monkeypatch, "brace")
    assert chosen is not base
    music = ET.fromstring(result[1])
    assert [part.findtext("measure/attributes/staves", default="1") for part in music.findall("part")] == ["2", "1"]

    def synthetic(reading, _facts):
        next(s for s in reading.layout.syms if s.cls == "grpSym_brace").det = -1

    base, old, chosen, result = run(monkeypatch, "brace", synthetic)
    assert chosen is base
    assert result == old


def test_staff_trial_uses_the_selected_opening_write(monkeypatch):
    from copisteria import staff_recovery

    stamp = {"audit": None}

    def read_page(*, opening_joins):
        gap = (80, 120)
        opening_joins.found.add(gap)
        opening_joins.legacy.add(gap)
        return SimpleNamespace(layout=SimpleNamespace(syms=[]), read_staff=lambda: None, tried=bool(opening_joins.allowed))

    def written(rd):
        stamp["audit"] = [rd.tried]
        return ((set() if rd.tried else {1},), rd.tried)

    def recover(base, result, _write, _audit, before):
        assert before == [False]
        assert stamp["audit"] == [True]
        return base, result

    monkeypatch.setattr(staff_recovery, "recover", recover)
    base = read_for_opening_trials(read_page)
    old = written(base)
    assert base.opening_trial(base, written, old, audit=lambda _: stamp["audit"]) == (base, old)
