# Modified by Play Anything (https://github.com/playanything-org/copisteria): see README.md.
from fractions import Fraction

import xml.etree.ElementTree as ET

from copisteria import front, write
from copisteria.read import Reading
from copisteria.vocab import CLS_ID


def det(cls, box, conf=0.9, **attrs):
    d = {"cls": cls, "conf": conf, "xyxy": list(box)}
    if attrs:
        d["attrs"] = {k: str(v) for k, v in attrs.items()}
        d["attr_p"] = {k: 0.9 for k in attrs}
    return d


def staff_row(y0, xs, sp=10.0):
    return [det("measure", (a, y0, b, y0 + 4 * sp)) for a, b in zip(xs, xs[1:])]


def test_merge_same_family_only():
    syms = front.merge([det("note_quarter", (0, 0, 10, 10), 0.8), det("note_half", (1, 0, 11, 10), 0.4),
                        det("dots_1", (0, 0, 10, 10), 0.5)])
    assert len(syms) == 2
    note = next(s for s in syms if s.fam == "note")
    assert note.cls == "note_quarter" and note.n == 2
    assert abs(note.post[CLS_ID["note_quarter"]] - 0.8 / 1.2) < 1e-6
    assert abs(note.p - (1 - 0.2 * 0.6)) < 1e-6


def test_staff_chains_across_a_missed_bar_and_fills_it():
    dets = staff_row(100, [0, 100, 200]) + staff_row(100, [300, 400])        # bar 200-300 missed
    dets.append(det("note_quarter", (240, 115, 250, 125), staff_position=0))
    L = front.build(dets, 500, 300)
    assert len(L.staves) == 1
    st = L.staves[0]
    assert len(st.bars) == 4
    gap = L.syms[st.bars[2]]
    assert gap.synthetic and gap.box[0] == 200 and gap.box[2] == 300
    note = next(s for s in L.syms if s.fam == "note")
    assert note.bar == 2


def test_aligned_staves_form_a_system_unaligned_do_not(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)              # the hand rule, not the fitted model
    dets = staff_row(100, [0, 100, 210, 300]) + staff_row(180, [0, 101, 209, 300]) + \
        staff_row(400, [0, 140, 230, 300])
    L = front.build(dets, 400, 600)
    assert [sy.staves for sy in L.systems] == [[0, 1], [2]]


def test_a_scans_frame_is_not_ink():
    import numpy as np
    page = np.full((400, 300), 230, np.uint8)
    page[100:104, 50:250] = 40                                    # a staff line
    page[60:300, 50:52] = 40                                      # a system's opening line
    framed = page.copy()
    framed[:, :12] = 25                                           # the scanner bed around the sheet
    framed[:, -9:] = 25
    framed[:10, :] = 25
    assert front.unframe(page) is page                            # no frame: the page as it is
    assert (front.ink_mask(framed) == front.ink_mask(page)).all()
    gap = front.ink_mask(framed)[150:250]                         # the gap below the staff, left of the line
    assert not gap[:, :45].any() and gap[:, 50:52].all()


def test_a_thick_frame_round_a_dense_page_ends_where_the_paper_starts():
    import numpy as np
    page = np.full((400, 300), 230, np.uint8)
    page[50:351:6, 40:261] = 40                                   # dense staff lines: 13 % of every column is ink
    page[50:351, 60:62] = 40                                      # a system's opening line
    page[:40] = 25                                                # a thick scanner bed on every side: with it,
    page[-40:] = 25                                               # a third of every column is ink
    page[:, :30] = 25
    page[:, -30:] = 25
    x0, y0, x1, y1 = front._frame(front._dark(page))
    assert 30 <= x0 <= 36 and 40 <= y0 <= 46 and 264 <= x1 <= 270 and 354 <= y1 <= 360
    assert front.ink_mask(page)[100:300, 60].all()                # the opening line is still ink


def test_pitch_under_clefs():
    assert write._pitch("G2", 0) == ("B", 4)
    assert write._pitch("G2", -6) == ("C", 4)
    assert write._pitch("F4", 0) == ("D", 3)
    assert write._pitch("C3", 0) == ("C", 4)
    assert write._pitch("G2_8b", 0) == ("B", 3)


def test_durations():
    assert write._duration("quarter", 1, "none") == Fraction(3, 2)
    assert write._duration("eighth", 0, "3/2") == Fraction(1, 3)
    assert write._duration("half", 2, "none") == Fraction(7, 2)


def _reading(L, tok):
    return Reading(layout=L, tok=tok)


def test_write_one_bar():
    dets = staff_row(100, [0, 200]) + [det("note_quarter", (50, 115, 60, 125), staff_position=0),
                                      det("note_quarter", (100, 115, 110, 125), staff_position=2),
                                      det("meterSig_2_4", (20, 100, 30, 140))]
    L = front.build(dets, 300, 300)
    tok = {}
    for s in L.syms:
        if s.fam == "measure":
            tok[s.i] = {"real": {"p": 1.0, "det": 1.0, "ev": 0}, "cls": {"v": "measure"},
                        "key": {"v": -1}, "time": {"v": "2/4"}, "clef": {"v": "G2"}}
        elif s.fam == "meterSig":
            tok[s.i] = {"real": {"p": 1.0, "det": 1.0, "ev": 0}, "cls": {"v": "meterSig_2_4"}}
        else:
            pos = 0 if s.cx < 80 else 2
            tok[s.i] = {"real": {"p": 1.0, "det": 1.0, "ev": 0}, "cls": {"v": "note_quarter"}, "dots": {"v": 0},
                        "pos": {"v": pos}, "voice": {"v": 1}, "tup": {"v": "none"}, "alter": {"v": "key" if pos == 0 else 0},
                        "grace": {"v": "none"}, "chord": 0.0, "tie": 0.0}
    xml = write.write(_reading(L, tok))
    root = ET.fromstring(xml.split("\n", 2)[2])
    notes = root.findall(".//note")
    assert [n.findtext("pitch/step") for n in notes] == ["B", "D"]
    assert notes[0].findtext("pitch/alter") == "-1"
    assert root.findtext(".//key/fifths") == "-1" and root.findtext(".//time/beats") == "2"


def _tied_bar(curves, ties=(1.0, 0.0, 0.0, 0.0)):
    """One 4/4 bar of four quarter Bs, the first read as tied, with these curves drawn; its notes as written."""
    xs = (40, 80, 120, 160)
    dets = staff_row(100, [0, 200]) + [det("meterSig_4_4", (10, 100, 20, 140))]
    dets += [det("note_quarter", (x - 5, 115, x + 5, 125), staff_position=0) for x in xs]
    dets += [det("curve", box) for box in curves]
    L = front.build(dets, 300, 300)
    tok = {}
    for s in L.syms:
        real = {"real": {"p": 1.0, "det": 1.0, "ev": 0}}
        if s.fam == "measure":
            tok[s.i] = {**real, "cls": {"v": "measure"}, "key": {"v": 0}, "time": {"v": "4/4"}, "clef": {"v": "G2"}}
        elif s.fam == "meterSig":
            tok[s.i] = {**real, "cls": {"v": "meterSig_4_4"}}
        elif s.cls == "curve":
            tok[s.i] = {**real, "cls": {"v": "curve"}}
        else:
            tok[s.i] = {**real, "cls": {"v": "note_quarter"}, "dots": {"v": 0}, "pos": {"v": 0}, "voice": {"v": 1},
                        "tup": {"v": "none"}, "alter": {"v": 0}, "grace": {"v": "none"}, "chord": 0.0,
                        "tie": ties[xs.index(round(s.cx))]}
    root = ET.fromstring(write.write(_reading(L, tok)).split("\n", 2)[2])
    return root.findall(".//note")


def test_a_tie_needs_the_page_to_agree():
    ties = lambda notes: [[t.get("type") for t in n.findall("tie")] for n in notes]     # noqa: E731
    # drawn between the two heads: a tie
    assert ties(_tied_bar([(46, 125, 74, 131)])) == [["start"], ["stop"], [], []]
    # no curve found (a short tie is often missed), the partner the next note: still a tie
    assert ties(_tied_bar([])) == [["start"], ["stop"], [], []]
    # the curve starting at the note ends two notes on: a slur, not a tie
    notes = _tied_bar([(44, 126, 156, 134)])
    assert ties(notes) == [[], [], [], []]
    assert [s.get("type") for s in notes[0].findall("notations/slur")] == ["start"]


def test_chord_member_joins_nearest_root():
    E = write.Event
    root, member, other = E(x=10, kind="note", typ="quarter"), E(x=11, kind="note", typ="eighth", chord=True), \
        E(x=40, kind="note", typ="quarter")
    vs = write._voices([member, other, root], sp=10)
    evs = vs[1]
    assert [e is m for e, m in zip(evs, (root, member, other))] == [True, True, True]
    assert member.chord and member.dur == root.dur == 1


def test_lone_member_stands_alone_and_tuplets_close():
    E = write.Event
    a = E(x=10, kind="note", typ="eighth", tup="3/2")
    b = E(x=20, kind="note", typ="eighth", tup="3/2", chord=True)      # no root within 1.5 sp: its own onset
    c = E(x=30, kind="note", typ="eighth", tup="3/2")
    write._voices([a, b, c], sp=5)
    assert not b.chord
    assert (a.tuplet_mark, b.tuplet_mark, c.tuplet_mark) == ("start", "", "stop")
    assert a.dur == Fraction(1, 3)


def _staff_bar(*voices, meter="3/4"):
    return {"voices": dict(enumerate(voices, 1)), "marks": {"meter": False, "right": None},
            "state": {"time": meter}}


def test_an_overfull_voice_is_cut_at_the_length_most_staves_give_the_bar():
    E = write.Event
    q = lambda x, **kw: E(x=x, kind="note", typ="quarter", dur=Fraction(1), **kw)         # noqa: E731
    full = [q(10), q(20), q(30)]
    over = [q(10, beams=[(1, "begin")]), q(20, beams=[(1, "end")]), E(x=30, kind="note", typ="half", dur=Fraction(2)),
            E(x=40, kind="mrest", tie_stop=True)]
    over[2].tie = True
    over[3].step, over[3].octave = over[2].step, over[2].octave
    built = [[{"staves": [_staff_bar(full)]}], [{"staves": [_staff_bar(over)]}], [{"staves": [_staff_bar(list(full))]}]]
    assert write._column_lengths(built) == [3]                       # 3, 5 + a bar, 3: the bar is 3 beats
    out = write._chop(over, Fraction(3))
    assert [(e.typ, e.dur) for e in out] == [("quarter", 1), ("quarter", 1), ("quarter", 1)]    # the half ends at 3
    assert not out[2].tie                                            # its partner was cut away
    assert write._chop(full, Fraction(3)) is full                    # a voice in time is left as it is


def test_a_cut_closes_the_beams_and_tuplets_it_breaks():
    E = write.Event
    t = lambda x, mark="": E(x=x, kind="note", typ="eighth", tup="3/2", dur=Fraction(1, 3), tuplet_mark=mark)  # noqa
    evs = [E(x=0, kind="note", typ="half", dur=Fraction(2))] + [t(10, "start"), t(20), t(30, "stop")]
    for e, b in zip(evs[1:], ("begin", "continue", "end")):
        e.beams = [(1, b)]
    out = write._chop(evs, Fraction(8, 3))                           # the last triplet eighth falls past the cut
    assert len(out) == 3
    assert (out[1].tuplet_mark, out[2].tuplet_mark) == ("start", "stop")
    assert (out[1].beams, out[2].beams) == ([(1, "begin")], [(1, "end")])


def _dist(onset, tup_none=0.6):
    beat = [0.0] * 16; frac = [0.0] * 48
    b = int(onset); f = round((onset - b) * 48)
    beat[b] = 0.9; frac[f] = 0.9
    for k in range(16):
        beat[k] += 0.1 / 16
    for k in range(48):
        frac[k] += 0.1 / 48
    tup = [tup_none, 1 - tup_none] + [0.0] * 6
    return {"beat": beat, "frac": frac, "tup": tup}


def test_rhythm_decoding_reads_unmarked_triplets_from_onsets():
    E = write.Event
    evs = [E(x=10 * k, kind="note", typ="eighth", dist=_dist(k / 3)) for k in range(6)]
    write._decode_rhythm(evs)
    assert [e.tup for e in evs[:-1]] == ["3/2"] * 5
    assert [round(e.onset, 3) for e in evs] == [round(k / 3, 3) for k in range(6)]


def test_rhythm_decoding_keeps_plain_eighths():
    E = write.Event
    evs = [E(x=10 * k, kind="note", typ="eighth", dist=_dist(k / 2, tup_none=0.4)) for k in range(4)]
    write._decode_rhythm(evs)
    assert [e.tup for e in evs[:-1]] == ["none"] * 3


def _tokens(L, clef=lambda s: "G2"):
    tok = {}
    for s in L.syms:
        real = {"p": 1.0, "det": 1.0, "ev": 0}
        if s.fam == "measure":
            tok[s.i] = {"real": real, "cls": {"v": "measure"}, "key": {"v": 0}, "time": {"v": "4/4"},
                        "clef": {"v": clef(s)}}
        elif s.fam == "grpSym":
            tok[s.i] = {"real": real, "cls": {"v": "grpSym_brace"}}
        else:
            tok[s.i] = {"real": real, "cls": {"v": "note_quarter"}, "dots": {"v": 0}, "pos": {"v": 0},
                        "voice": {"v": 1}, "tup": {"v": "none"}, "alter": {"v": 0}, "grace": {"v": "none"},
                        "chord": 0.0, "tie": 0.0}
    return tok


def test_a_pedal_mark_is_written_under_a_keyboard_and_not_under_a_part_of_one_staff(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    xs = [0, 150, 300]
    # a voice over a braced piano, a note on the voice's staff and one on the piano's lower staff, and under each of
    # those staves something the detector calls a pedal mark: under the voice it is a syllable of its lyrics
    rows = staff_row(100, xs) + staff_row(200, xs) + staff_row(280, xs)
    marks = [det("pedal", (40, 146, 70, 160)), det("pedal", (40, 326, 70, 340))]
    notes = [det("note_quarter", (50, 115, 60, 125), staff_position=0), det("note_quarter", (50, 295, 60, 305), staff_position=0)]
    L = front.build(rows + [det("grpSym_brace", (0, 200, 8, 320))] + notes + marks, 400, 450)
    assert [len(sy.staves) for sy in L.systems] == [3]
    tok = _tokens(L)
    for s in L.syms:
        if s.fam == "pedal":
            tok[s.i] = {"real": {"p": 1.0, "det": 1.0, "ev": 0}, "cls": {"v": "pedal"}}
    assert sorted(L.staves[s.staff].pos for s in L.syms if s.fam == "pedal") == [0, 2]
    root = ET.fromstring(write.write(_reading(L, tok)).split("\n", 2)[2])
    assert [len(p.findall(".//pedal")) for p in root.findall("part")] == [0, 2]


def test_a_keyboard_part_that_enters_late_declares_its_staves(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    # an a cappella opening (the voice alone), then the voice over the braced piano
    dets = staff_row(100, [0, 150, 300]) + \
        staff_row(400, [0, 100, 200, 300]) + staff_row(480, [0, 100, 200, 300]) + \
        staff_row(560, [0, 100, 200, 300]) + \
        [det("grpSym_brace", (0, 480, 8, 600)),
         det("note_quarter", (50, 115, 60, 125), staff_position=0),
         det("note_quarter", (50, 495, 60, 505), staff_position=0)]
    L = front.build(dets, 400, 700)
    assert [len(sy.staves) for sy in L.systems] == [1, 3]
    root = ET.fromstring(write.write(_reading(L, _tokens(L))).split("\n", 2)[2])
    voice, piano = root.findall("part")
    assert piano.find("measure").find("note") is None                 # the piano is silent in bar 1 ...
    assert piano.findtext("measure/attributes/staves") == "2"          # ... and says there that it has two staves


def _parts(L, tok):
    """Per part: its staves, and per bar whether it sounds."""
    root = ET.fromstring(write.write(_reading(L, tok)).split("\n", 2)[2])
    return [(p.findtext("measure/attributes/staves") or "1", [m.find("note/pitch") is not None for m in p.findall("measure")])
            for p in root.findall("part")]


def test_a_piano_introduction_is_written_into_the_piano(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    # a piano introduction (two braced staves), then the full system: a voice over the braced piano
    dets = staff_row(100, [0, 150, 300]) + staff_row(180, [0, 150, 300]) + \
        staff_row(400, [0, 100, 200, 300]) + staff_row(480, [0, 100, 200, 300]) + \
        staff_row(560, [0, 100, 200, 300]) + \
        [det("grpSym_brace", (0, 100, 8, 220)), det("grpSym_brace", (0, 480, 8, 600)),
         det("note_quarter", (50, 115, 60, 125), staff_position=0),       # the introduction's right hand
         det("note_quarter", (50, 415, 60, 425), staff_position=0)]       # the voice's first note
    L = front.build(dets, 400, 700)
    assert [len(sy.staves) for sy in L.systems] == [2, 3]
    voice, piano = _parts(L, _tokens(L))
    assert piano == ("2", [True, False, False, False, False])
    assert voice == ("1", [False, False, True, False, False])


def test_a_piano_introduction_without_its_brace_is_placed_by_its_clefs(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    dets = staff_row(100, [0, 150, 300]) + staff_row(180, [0, 150, 300]) + \
        staff_row(400, [0, 100, 200, 300]) + staff_row(480, [0, 100, 200, 300]) + \
        staff_row(560, [0, 100, 200, 300]) + \
        [det("grpSym_brace", (0, 480, 8, 600)),                            # the introduction's brace was missed
         det("note_quarter", (50, 115, 60, 125), staff_position=0)]
    L = front.build(dets, 400, 700)
    bass = lambda s: "F4" if s.box[1] in (180, 560) else "G2"            # noqa: E731  the piano's left hand
    voice, piano = _parts(L, _tokens(L, clef=bass))
    assert piano == ("2", [True, False, False, False, False]) and not any(voice[1])


def test_with_nothing_to_tell_a_system_is_counted_from_the_top(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    dets = staff_row(100, [0, 150, 300]) + staff_row(180, [0, 150, 300]) + \
        staff_row(400, [0, 100, 200, 300]) + staff_row(480, [0, 100, 200, 300]) + \
        staff_row(560, [0, 100, 200, 300]) + \
        [det("note_quarter", (50, 115, 60, 125), staff_position=0), det("note_quarter", (50, 195, 60, 205), staff_position=0)]
    L = front.build(dets, 400, 700)
    assert [len(sy.staves) for sy in L.systems] == [2, 3]
    first, second, third = _parts(L, _tokens(L))
    assert first[1][0] and second[1][0] and not third[1][0]


def test_near_white_noise_does_not_carry_a_staff_onto_its_brace():
    import numpy as np
    from PIL import Image

    from copisteria.vision.page.staves import find_staves
    page = np.full((200, 500), 255, np.uint8)
    for y in range(60, 110, 10):
        page[y, 60:460] = 0                          # the staff's lines, from x = 60
        page[y, 40:60] = 251                         # a JPEG's faint noise between them and the brace
    page[50:120, 36:40] = 0                          # the brace
    assert find_staves(Image.fromarray(page)).staves[0].x0 == 36            # 254: the noise is ink
    assert find_staves(Image.fromarray(page), paper_tol=front.STAFF_PAPER_TOL).staves[0].x0 == 60


def test_a_fullest_system_without_the_braced_pair_is_no_guide(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    # a piano page whose first grand staff was read as three staves (G, G, F, no brace); the next is a braced pair
    dets = staff_row(100, [0, 150, 300]) + staff_row(180, [0, 150, 300]) + staff_row(260, [0, 150, 300]) + \
        staff_row(500, [0, 100, 200, 300]) + staff_row(580, [0, 100, 200, 300]) + \
        [det("grpSym_brace", (0, 500, 8, 620)), det("note_quarter", (50, 515, 60, 525), staff_position=0),
         det("note_quarter", (50, 595, 60, 605), staff_position=0)]
    L = front.build(dets, 400, 800)
    assert [len(sy.staves) for sy in L.systems] == [3, 2]
    bass = lambda s: "F4" if s.box[1] in (260, 580) else "G2"            # noqa: E731
    parts = _parts(L, _tokens(L, clef=bass))
    assert [sounds[2] for _, sounds in parts] == [True, True, False]       # the pair keeps the top two places


def test_a_pale_one_pixel_line_is_evidence_a_grey_patch_is_not():
    import numpy as np
    levels = np.full((300, 400), 255, np.uint8)
    levels[40:60, 50:350] = 0                        # ink, so the page has a dark end
    a = front.Staff(bars=[], sp=4.5, y0=100, y1=118, x0=40, x1=380)
    b = front.Staff(bars=[], sp=4.5, y0=180, y1=198, x0=40, x1=380)
    line = levels.copy()
    line[100:199, 38] = 204                          # a small render's opening line: one pixel, grey
    patch = levels.copy()
    patch[100:199, 20:45] = 204                      # a stamp over the margin
    assert front._pale_line(line, 255, 0, a, b, 40) > 0.9
    assert front._pale_line(patch, 255, 0, a, b, 40) < 0.5


def test_the_edge_of_a_bindings_shadow_is_not_a_pale_line_and_a_lines_pale_shoulder_is():
    import numpy as np
    levels = np.full((1200, 600), 255, np.uint8)
    levels[40:60, 100:500] = 0                       # ink, so the page has a dark end
    a = front.Staff(bars=[], sp=24.0, y0=300, y1=396, x0=24, x1=560)
    b = front.Staff(bars=[], sp=24.0, y0=640, y1=736, x0=24, x1=560)
    # a scan's binding shadow with its frame taken off: white up to x = 24, then the shadow's edge, fading to paper
    # over a staff space, for the whole height of the page (the levels of a production scan)
    shadow = levels.copy()
    for k, tone in enumerate([219, 219, 221, 221, 222, 225, 225, 225, 228, 228, 231, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253]):
        shadow[:, 24 + k] = tone
    assert front._pale_line(shadow, 255, 0, a, b, 24) < 0.5
    # a steeper shadow passes for a line between two staves, and is known by where it goes on: past the page's first
    # staff and past its last, where no system's opening line is
    steep = levels.copy()
    for k, tone in enumerate([212, 214, 218, 224, 232, 242, 250]):
        steep[:, 24 + k] = tone
    share, x = front._pale_at(steep, 255, 0, a, b, 24)
    assert share > 0.9 and front._runs_on(steep, 255, 0, x, [a, b]) and front._pale_at(steep, 255, 0, a, b, 24, [a, b])[0] < 0.5
    e = front.Staff(bars=[], sp=24.0, y0=300, y1=396, x0=40, x1=560)
    g = front.Staff(bars=[], sp=24.0, y0=640, y1=736, x0=40, x1=560)
    opening = levels.copy()
    opening[300:737, 38] = 204                       # a system's own pale opening line ends with its staves
    share, x = front._pale_at(opening, 255, 0, e, g, 40, [e, g])
    assert share > 0.9 and not front._runs_on(opening, 255, 0, x, [e, g])
    # where the page has no margin to look at past its staves, nothing is said and the line counts
    low = front.Staff(bars=[], sp=24.0, y0=1090, y1=1186, x0=40, x1=560)
    assert not front._runs_on(steep, 255, 0, x, [a, low])
    # a shadow's edge in a scan's noise: the columns of its tail are a line in the gap on most rows and past the
    # staves on fewer, and are the shadow's still
    for seed in range(8):
        noisy = np.full((1200, 600), 255.0)
        noisy[40:60, 100:500] = 0
        for k in range(12):
            noisy[:, 24 + k] = 215 + 40 * k / 12
        noisy[:, 24:60] += np.random.default_rng(seed).normal(0, 4, (1200, 36))
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        assert front._pale_at(noisy, 255, 0, a, b, 24, [a, b])[0] < 0.85, seed
    # a pale line on paper that is locally darker than the page's is a line still
    dim = np.full((1200, 600), 250, np.uint8)
    dim[40:60, 100:500] = 30
    dim[:, :200] = 228
    dim[300:737, 38] = 210
    assert front._pale_line(dim, 250, 30, e, g, 40) > 0.9
    # the opening line itself, dark, a few pixels past the columns searched, with its pale shoulder in them: on one
    # side the paper, on the other the line
    c = front.Staff(bars=[], sp=11.0, y0=300, y1=344, x0=392, x1=560)
    d = front.Staff(bars=[], sp=11.0, y0=420, y1=464, x0=384, x1=560)
    shoulder = levels.copy()
    for k, tone in enumerate([242, 242, 236, 236, 236, 214, 213, 214, 145, 145, 232]):
        shoulder[:, 391 + k] = tone
    assert front._pale_line(shoulder, 255, 0, c, d, 384) > 0.9


def test_a_pale_opening_line_joins_two_staves_and_a_white_gap_does_not():
    import numpy as np
    from PIL import Image
    page = np.full((360, 620), 255, np.uint8)
    for top in (100, 220):                           # a voice's staff and, below it, the piano's top staff
        for y in range(top, top + 50, 10):
            page[y, 60:560] = 0
        for x in (60, 310, 559):
            page[top:top + 41, x] = 0                # bar lines on each staff, none through the gap
    dets = staff_row(100, [60, 310, 560]) + staff_row(220, [60, 310, 560])
    white = front.build(dets, 620, 360, image=Image.fromarray(page))
    page[100:261, 58] = 204                          # the system's opening line, one pixel, too pale for the mask
    pale = front.build(dets, 620, 360, image=Image.fromarray(page))
    assert [len(sy.staves) for sy in white.systems] == [1, 1]
    assert [len(sy.staves) for sy in pale.systems] == [2]
    page[:, 58] = 204                                # the same line for the page's whole height: the sheet's
    edge = front.build(dets, 620, 360, image=Image.fromarray(page))
    assert [len(sy.staves) for sy in edge.systems] == [1, 1]


def test_a_staff_the_line_finder_finds_twice_is_one_staff():
    """A bowed scan gives each staff line two runs of rows, and the finder a second staff half a space under the
    first: the one the measure boxes sit on stays, and a staff a space below another is left alone."""
    from copisteria.vision.page.staves import Line, Staff as Found

    def found(top, score=0.5, sp=10.0):
        return Found(lines=[Line(y=top + k * sp, x0=0, x1=400, thickness=2, score=score) for k in range(5)])

    syms = front.merge(staff_row(100, [0, 200, 400]) + staff_row(200, [0, 200, 400]))
    upper, double, lower = found(100, 0.4), found(106, 0.9), found(200)
    assert front._without_doubles([double, upper, lower], syms) == [upper, lower]
    assert front._without_doubles([upper, lower], syms) == [upper, lower]
    # with no measure box to tell, the one whose lines score higher
    assert front._without_doubles([upper, double], []) == [double]
    # staves that only touch are two staves
    assert front._without_doubles([found(100), found(138)], []) == [found(100), found(138)]


def _rostered(L, tok, roster):
    """Per part its staves and per bar whether it sounds, and why the roster was not used (None where it was)."""
    facts = {}
    root = ET.fromstring(write.write(_reading(L, tok), roster=roster, facts=facts).split("\n", 2)[2])
    return [(p.findtext("measure/attributes/staves") or "1", [m.find("note/pitch") is not None for m in p.findall("measure")])
            for p in root.findall("part")], facts["roster"]


def test_a_page_of_piano_alone_is_written_into_the_rosters_piano(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    # a song's page that shows only its piano: by itself one part of two staves; with the song's roster (a voice
    # and a piano of two staves) the voice's part is there and silent, and the piano holds the music, brace or not
    rows = staff_row(100, [0, 150, 300]) + staff_row(180, [0, 150, 300])
    note = [det("note_quarter", (50, 115, 60, 125), staff_position=0)]
    for brace in ([det("grpSym_brace", (0, 100, 8, 220))], []):
        L = front.build(rows + brace + note, 400, 300)
        assert [len(sy.staves) for sy in L.systems] == [2]
        parts, why = _rostered(L, _tokens(L), [1, 2])
        assert why is None and parts == [("1", [False, False]), ("2", [True, False])]


def test_a_voice_alone_is_the_rosters_voice_and_a_page_the_roster_does_not_explain_is_planned_as_before(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    voice = staff_row(100, [0, 150, 300]) + [det("note_quarter", (50, 115, 60, 125), staff_position=0)]
    L = front.build(voice, 400, 300)
    parts, why = _rostered(L, _tokens(L), [1, 2])
    assert why is None and parts == [("1", [True, False]), ("2", [False, False])]
    # three staves where the roster has two: the page keeps its own plan, and says why
    L = front.build(staff_row(100, [0, 150, 300]) + staff_row(180, [0, 150, 300]) + staff_row(260, [0, 150, 300]), 400, 400)
    parts, why = _rostered(L, _tokens(L), [2])
    assert len(parts) == 3 and "3 staves" in why
    # a brace over the two staves the roster gives to two instruments: not this roster's page
    L = front.build(staff_row(100, [0, 150, 300]) + staff_row(180, [0, 150, 300]) + [det("grpSym_brace", (0, 100, 8, 220))], 400, 300)
    parts, why = _rostered(L, _tokens(L), [1, 1])
    assert parts == [("2", [False, False])] and "brace" in why


def test_a_system_the_roster_fits_more_than_one_way_is_not_placed_by_a_guess(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    xs = [0, 150, 300]
    brace = lambda top: det("grpSym_brace", (0, top, 8, top + 120))              # noqa: E731
    # bells, a chorus, two voices and a piano; the page shows two treble staves over the braced piano. They are any
    # two of the four single staves: nothing on the page says which, so the roster does not explain it, and the page
    # keeps its own three parts rather than giving two players' music to the first two
    rows = staff_row(100, xs) + staff_row(180, xs) + staff_row(260, xs) + staff_row(340, xs)
    L = front.build(rows + [brace(260)] + [det("note_quarter", (50, 115, 60, 125), staff_position=0)], 400, 500)
    assert [len(sy.staves) for sy in L.systems] == [4]
    parts, why = _rostered(L, _tokens(L), [1, 1, 1, 1, 2])
    assert "more than one way" in why and [staves for staves, _ in parts] == ["1", "1", "2"]
    # one way only: a voice over the piano is the roster's one voice, and the piano alone is its piano
    parts, why = _rostered(L, _tokens(L), [1, 1, 2])
    assert why is None and [staves for staves, _ in parts] == ["1", "1", "2"]
    # the clefs tell: a soprano and a bass over the piano on the page's whole system, then the bass alone with it
    ys = [0, 100, 200, 300]                                                      # other bars: another system
    below = staff_row(500, ys) + staff_row(580, ys) + staff_row(660, ys)
    whole = staff_row(100, xs) + staff_row(180, xs) + staff_row(260, xs) + staff_row(340, xs)
    L = front.build(whole + [brace(260)] + below + [brace(580)] + [det("note_quarter", (50, 515, 60, 525), staff_position=0)], 400, 800)
    assert [len(sy.staves) for sy in L.systems] == [4, 3]
    low = [sy.staves[k] for sy, ks in zip(L.systems, ([1, 3], [0, 2])) for k in ks]
    bass = lambda s: "F4" if s.staff in low else "G2"                            # noqa: E731
    parts, why = _rostered(L, _tokens(L, clef=bass), [1, 1, 2])
    assert why is None
    # the lone staff's notes are the bass's, the roster's second part, and the soprano rests there
    assert [any(sounds) for _, sounds in parts] == [False, True, False]
    # the same page with every clef alike: the lone staff is the soprano as readily as the bass
    parts, why = _rostered(L, _tokens(L), [1, 1, 2])
    assert "more than one way" in why


def _placed(L, tok, roster, **read_in_print):
    """Whether each part sounds anywhere, why the roster was not used (None where it was), and the page as written."""
    facts = {}
    text = write.write(_reading(L, tok), roster=roster, facts=facts, **read_in_print)
    root = ET.fromstring(text.split("\n", 2)[2])
    return [any(m.find("note/pitch") is not None for m in p.findall("measure")) for p in root.findall("part")], facts["roster"], text


def _choir_page():
    """Two treble staves over a braced piano, each of the upper two with a note: a page of a score for four voices
    and piano. Braces and clefs do not say which two voices sing."""
    xs = [0, 150, 300]
    rows = staff_row(100, xs) + staff_row(180, xs) + staff_row(260, xs) + staff_row(340, xs)
    L = front.build(rows + [det("grpSym_brace", (0, 260, 8, 380)), det("note_quarter", (50, 115, 60, 125), staff_position=0),
                            det("note_quarter", (50, 195, 60, 205), staff_position=0)], 400, 500)
    assert [len(sy.staves) for sy in L.systems] == [4]
    return L


def test_a_staff_stands_only_at_an_instrument_the_caller_says_it_may_be(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    L, satb = _choir_page(), [1, 1, 1, 1, 2]
    assert "more than one way" in _placed(L, _tokens(L), satb)[1]
    # the caller read "T." and "B." by the upper staves: the tenor and the bass sing, the others rest
    sounds, why, text = _placed(L, _tokens(L), satb, among=[[[2], [3], [4], None]])
    assert why is None and sounds == [False, False, True, True, False]
    # nothing the caller says is written: the parts are numbered as on any page
    assert [n.text for n in ET.fromstring(text.split("\n", 2)[2]).iter("part-name")] == [f"Part {n}" for n in range(1, 6)]
    assert _placed(L, _tokens(L), satb, among=[[[0], [1], None, None]])[:2] == ([True, True, False, False, False], None)
    # one staff is enough where it leaves one fit: the staff over the alto's is no one's but the soprano's ...
    assert _placed(L, _tokens(L), satb, among=[[None, [1], None, None]])[:2] == ([True, True, False, False, False], None)
    # ... and is not where it leaves two (the staff under the soprano's is any of three)
    assert "more than one way" in _placed(L, _tokens(L), satb, among=[[[0], None, None, None]])[1]
    # a staff that may be either of two instruments is placed by the rest, or not at all
    assert _placed(L, _tokens(L), satb, among=[[[0, 1], [3], None, None]])[1] == "a system of 4 staves fits the roster's 6 more than one way"
    assert _placed(L, _tokens(L), satb, among=[[[1, 2], [2, 3], [4], [4]]])[1] == "a system of 4 staves fits the roster's 6 more than one way"
    assert _placed(L, _tokens(L), satb, among=[[[2, 3], [2, 3], [4], [4]]])[:2] == ([False, False, True, True, False], None)
    # staves that may stand nowhere in order (the bass above the soprano): not this roster's page, which keeps its own parts
    sounds, why, _ = _placed(L, _tokens(L), satb, among=[[[3], [0], None, None]])
    assert "nowhere" in why and len(sounds) == 3
    # the piano's staves named as a voice: nowhere either
    assert "nowhere" in _placed(L, _tokens(L), satb, among=[[None, None, [0], None]])[1]


def test_what_the_caller_says_is_used_only_as_far_as_it_is_well_formed(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    L, satb = _choir_page(), [1, 1, 1, 1, 2]
    good = [[[2], [3], [4], None]]
    assert _placed(L, _tokens(L), satb, among=good)[1] is None
    # not one entry for each staff of each system, an instrument the roster has not, an empty list, a number, a
    # string, a truth value: what the caller read is not of these systems, the page is not explained, and nothing
    # raises
    for bad in ([[[2], [3], [4]]], [[[2], [3], [4], None], [[0]]], [[[2], [9], [4], None]], [[[2], [], [4], None]], [[2, 3, 4, None]],
                [["2", [3], [4], None]], [[[True], [3], [4], None]], [[[2.0], [3], [4], None]], "among", [None], {}, 7):
        assert _placed(L, _tokens(L), satb, among=bad)[1] == "the caller's systems are not the ones found", bad
    for bad in (["G2"], "clefs", ["G2", "G2", "G2", "F4", "G2", 4], [None] * 6, ["nonsense"] * 6):
        assert _placed(L, _tokens(L), satb, among=good, clefs=bad)[1] is None, bad


def test_rows_that_do_not_fit_the_systems_found_leave_the_page_unexplained(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    xs, ys = [0, 150, 300], [0, 100, 200, 300]
    # two systems of two treble staves each, of a score for soprano, alto and bass. The caller read three systems
    # (soprano and alto; alto and bass; all three): the reader lost one, so the caller's second row has no system here,
    # and its first is not known to be this page's first either
    page = staff_row(100, xs) + staff_row(180, xs) + staff_row(500, ys) + staff_row(580, ys)
    L = front.build(page + [det("note_quarter", (50, 115, 60, 125), staff_position=0)], 400, 700)
    assert [len(sy.staves) for sy in L.systems] == [2, 2]
    rows = [[[0], [1]], [[1], [2]]]
    assert _placed(L, _tokens(L), [1, 1, 1], among=rows)[:2] == ([True, False, False], None)
    for shifted in ([[[0], [1]], [[1], [2]], [[0], [1], [2]]], [[[0], [1]], [[0], [1], [2]]], [[[0], [1]]]):
        assert _placed(L, _tokens(L), [1, 1, 1], among=shifted)[1] == "the caller's systems are not the ones found", shifted


def test_a_page_read_as_one_system_where_the_caller_saw_several_does_not_take_the_roster(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    xs = [0, 150, 300]
    # a part's page: three lines of one bass-clef staff each, which the reader joined into one system of three staves.
    # The roster (two treble staves, three bass staves) would hold them one way by the clefs alone, as its three bass
    # instruments at once; the caller saw three systems of one staff, so nothing it sent is of this system
    page = staff_row(100, xs) + staff_row(180, xs) + staff_row(260, xs)
    L = front.build(page + [det("note_quarter", (50, 115, 60, 125), staff_position=0)], 400, 400)
    assert [len(sy.staves) for sy in L.systems] == [3]
    tok, clefs = _tokens(L, clef=lambda s: "F4"), ["G2", "G2", "F4", "F4", "F4"]
    assert _placed(L, tok, [1, 1, 1, 1, 1], clefs=clefs)[:2] == ([False, False, True, False, False], None)
    assert _placed(L, tok, [1, 1, 1, 1, 1], clefs=clefs, among=[[None], [None], [None]])[1] == "the caller's systems are not the ones found"


def test_a_whole_system_with_a_staff_that_may_not_be_its_instrument_is_not_the_rosters(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    xs, ys = [0, 150, 300], [0, 100, 200, 300]
    brace = lambda top: det("grpSym_brace", (0, top, 8, top + 120))              # noqa: E731
    # a voice over the braced piano, then the piano alone: by the roster two parts, the second system the piano's
    page = staff_row(100, xs) + staff_row(180, xs) + staff_row(260, xs) + [brace(180)] + staff_row(500, ys) + staff_row(580, ys) + [brace(500)]
    L = front.build(page + [det("note_quarter", (50, 115, 60, 125), staff_position=0)], 400, 700)
    assert [len(sy.staves) for sy in L.systems] == [3, 2]
    assert _placed(L, _tokens(L), [1, 2], among=[[[0], [1], None], [[1], None]])[:2] == ([True, False], None)
    # the caller read a flute's label by the top staff, where the roster has the voice: another score's roster
    sounds, why, _ = _placed(L, _tokens(L), [1, 2], among=[[[1], [1], None], [[1], None]])
    assert "whole system" in why and len(sounds) == 2


def test_the_clefs_read_in_print_stand_in_where_the_page_shows_no_whole_system(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    xs = [0, 150, 300]
    # a soprano, a bass and a piano; this page shows one voice over the braced piano and no whole system, so the page
    # itself cannot say which voice. The clefs read off the score's whole system can: the staff opens with a bass clef
    rows = staff_row(100, xs) + staff_row(180, xs) + staff_row(260, xs)
    L = front.build(rows + [det("grpSym_brace", (0, 180, 8, 300)), det("note_quarter", (50, 115, 60, 125), staff_position=0)], 400, 400)
    low = (L.systems[0].staves[0], L.systems[0].staves[2])
    bass = lambda s: "F4" if s.staff in low else "G2"                            # noqa: E731
    assert "more than one way" in _placed(L, _tokens(L, clef=bass), [1, 1, 2])[1]
    assert _placed(L, _tokens(L, clef=bass), [1, 1, 2], clefs=["G2", "F4", "G2", "F4"])[:2] == ([False, True, False], None)
    # a clef with and without its octave mark is one clef: a treble staff is the soprano's as readily as the tenor's,
    # read with the 8 or without, and only something else can say which
    for read in ("G2", "G2_8b"):
        tok = _tokens(L, clef=lambda s: "F4" if s.staff == low[1] else read)     # noqa: B023
        assert "more than one way" in _placed(L, tok, [1, 1, 2], clefs=["G2", "G2_8b", "G2", "F4"])[1]
        assert _placed(L, tok, [1, 1, 2], clefs=["G2", "G2_8b", "G2", "F4"], among=[[[1], None, None]])[:2] == ([False, True, False], None)
        assert _placed(L, tok, [1, 1, 2], clefs=["F4", "G2_8b", "G2", "F4"])[:2] == ([False, True, False], None)
    # an unknown clef, read or stated, says nothing
    assert "more than one way" in _placed(L, _tokens(L, clef=bass), [1, 1, 2], clefs=["other", "other", "G2", "F4"])[1]
    # a clef the caller left out is unknown, not another: a treble staff is as readily the instrument whose clef was
    # left out as the one stated treble, where a bass staff is the first's before the second's
    assert "more than one way" in _placed(L, _tokens(L), [1, 1, 2], clefs=[None, "G2", "G2", "F4"])[1]
    assert _placed(L, _tokens(L, clef=bass), [1, 1, 2], clefs=[None, "G2", "G2", "F4"])[:2] == ([True, False, False], None)


def test_the_pages_own_whole_systems_say_the_clefs_before_the_caller_does(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    xs, ys = [0, 150, 300], [0, 100, 200, 300]
    brace = lambda top: det("grpSym_brace", (0, top, 8, top + 120))              # noqa: E731
    # the page's whole system opens treble, bass, piano; the lone staff below with the piano opens with a bass clef
    whole = staff_row(100, xs) + staff_row(180, xs) + staff_row(260, xs) + staff_row(340, xs)
    below = staff_row(500, ys) + staff_row(580, ys) + staff_row(660, ys)
    L = front.build(whole + [brace(260)] + below + [brace(580)] + [det("note_quarter", (50, 515, 60, 525), staff_position=0)], 400, 800)
    assert [len(sy.staves) for sy in L.systems] == [4, 3]
    low = [sy.staves[k] for sy, ks in zip(L.systems, ([1, 3], [0, 2])) for k in ks]
    bass = lambda s: "F4" if s.staff in low else "G2"                            # noqa: E731
    # stated clefs that contradict what this page shows (bass over treble) do not move the staff: the page says
    assert _placed(L, _tokens(L, clef=bass), [1, 1, 2], clefs=["F4", "G2", "G2", "F4"])[:2] == ([False, True, False], None)
    assert _placed(L, _tokens(L, clef=bass), [1, 1, 2])[:2] == ([False, True, False], None)


def test_a_voice_never_holds_two_events_at_once():
    """A keyboard hand whose second chord the model places before the first has ended was written with a backup
    inside the voice; a reader that keeps one event at a time in a voice then drops what overlaps. The overlapping
    events go to a voice of their own, each lane in time order with only forward gaps, and nothing is lost."""
    from fractions import Fraction as F
    E = write.Event

    def note(x, onset, typ="half", **more):
        return E(x=x, kind="note", typ=typ, onset=onset, onset_p=1.0, **more)

    def times(evs):
        out, pos = [], F(0)
        for e in evs:
            if e.kind == "gap":
                assert e.dur > 0
                pos += e.dur
            elif e.chord:
                out.append((out[-1][0], e.step))
            else:
                out.append((pos, e.step))
                pos += e.dur
        return out

    # a half-note chord at 0, a 32nd the model hears at 7/8, a second half-note chord it hears at 1, and a rest
    def events():
        return [note(10, 0.0, step="F"), note(10.5, 0.0, step="A", chord=True), note(40, 0.875, typ="32nd", step="B"),
                note(60, 1.0, step="G"), note(60.5, 1.0, step="C", chord=True),
                E(x=90, kind="rest", typ="quarter", onset=3.0, onset_p=1.0)]
    vs = write._voices(events(), 10.0, single_staff_part=False, bar_len=F(4))
    assert sorted(vs) == [1, 2]
    assert times(vs[1]) == [(F(0), "F"), (F(0), "A")]
    # what follows an event goes where it went while that voice is free: the rest (step is its default) after the chord
    assert times(vs[2]) == [(F(7, 8), "B"), (F(1), "G"), (F(1), "C"), (F(3), "C")]
    assert all(e.voice == 2 for e in vs[2]) and sum(e.kind == "note" for evs in vs.values() for e in evs) == 5
    # a voice with nothing overlapping is the very list it was
    plain = [note(10, 0.0, typ="quarter"), note(30, 1.0, typ="quarter")]
    read = {}
    vs = write._voices(plain, 10.0, single_staff_part=False, bar_len=F(4), read=read)
    assert sorted(vs) == [1] and [e.kind for e in vs[1]] == ["note", "note"] and read == {1: vs[1]}
    # with every voice of the staff taken, the voice is left as it was read rather than losing a lane
    full = [E(x=5 + n, kind="rest", typ="whole", voice=n, onset=0.0, onset_p=1.0) for n in (2, 3, 4)]
    vs = write._voices(events() + full, 10.0, single_staff_part=False, bar_len=F(4))
    assert sorted(vs) == [1, 2, 3, 4] and any(e.kind == "gap" and e.dur < 0 for e in vs[1])
    # a rest the model places over a sounding note says nothing, and opens no voice; the grace note before it is kept
    over = [note(10, 0.0, typ="whole", step="D"), E(x=35, kind="note", typ="eighth", step="E", grace="acciaccatura", onset=1.0, onset_p=1.0),
            E(x=40, kind="rest", typ="quarter", onset=1.0, onset_p=1.0)]
    vs = write._voices(over, 10.0, single_staff_part=False, bar_len=F(4))
    assert sorted(vs) == [1] and [(e.kind, e.step) for e in vs[1]] == [("note", "D"), ("note", "E")]
    # the voice as the model read it is handed back for ties: every event in page order, whatever voice it is written in
    read = {}
    vs = write._voices(events(), 10.0, single_staff_part=False, bar_len=F(4), read=read)
    assert sorted(read) == [1] and [e.step for e in read[1] if e.kind == "note"] == ["F", "A", "B", "G", "C"]
    # a voice read as the staff's second keeps its number, and what overlaps goes above it, not to the first
    second = [E(**{**e.__dict__, "voice": 2}) for e in events()]
    assert sorted(write._voices(second, 10.0, single_staff_part=False, bar_len=F(4))) == [2, 3]
    # the lane that ends in a tied note keeps the voice's number (its tie runs into the next bar's voice), and a note
    # goes to the lane that ends in a tied note of its pitch
    tied = [note(10, 0.0, typ="whole", step="E"), note(40, 2.0, typ="quarter", step="B", tie=True), note(60, 3.0, typ="quarter", step="B", tie=True)]
    vs = write._voices(tied, 10.0, single_staff_part=False, bar_len=F(4))
    assert [e.step for e in vs[1] if e.kind == "note"] == ["B", "B"] and [e.step for e in vs[2] if e.kind == "note"] == ["E"]
    # notes that stand one over the other at one time with one length are one chord, whatever the model called them:
    # a bass note, three eighths at the second eighth, a bass note and two more are one line of four eighths
    hand = [note(10, 0.0, typ="eighth", step="G"), note(30, 0.5, typ="eighth", step="B"), note(31, 0.5, typ="eighth", step="D"),
            note(33, 0.5, typ="eighth", step="F"), note(50, 1.0, typ="eighth", step="A"), note(70, 1.5, typ="eighth", step="C"),
            note(72, 1.5, typ="eighth", step="E")]
    vs = write._voices(hand, 10.0, single_staff_part=False, bar_len=F(2))
    assert sorted(vs) == [1] and times(vs[1]) == [(F(0), "G"), (F(1, 2), "B"), (F(1, 2), "D"), (F(1, 2), "F"), (F(1), "A"), (F(3, 2), "C"), (F(3, 2), "E")]
    assert [e.chord for e in vs[1] if e.kind == "note"] == [False, False, True, True, False, False, True]
    # a note of a pitch the chord already holds is two voices' unison, and stays an event of its own; and a chord takes
    # notes that stand by its first, not by one another across the page
    unison = [note(10, 0.0, typ="eighth", step="G"), note(11, 0.0, typ="eighth", step="G"), note(30, 0.5, typ="eighth", step="B")]
    vs = write._voices(unison, 10.0, single_staff_part=False, bar_len=F(2))
    assert sorted(vs) == [1, 2] and not any(e.chord for evs in vs.values() for e in evs)
    chain = [note(10 + 12 * k, 0.0, typ="eighth", step=s) for k, s in enumerate("CDEF")]
    vs = write._voices(chain, 10.0, single_staff_part=False, bar_len=F(2))
    assert sorted(vs) == [1, 2] and [[e.step for e in vs[v] if e.kind == "note"] for v in (1, 2)] == [["C", "D"], ["E", "F"]]
    assert [e.step for evs in vs.values() for e in evs if e.chord] == ["D", "F"]
    # standing apart on the page they are notes in turn whose onset was read wrong: events of their own, as before
    apart = [note(10, 0.0, typ="eighth", step="G"), note(30, 0.5, typ="eighth", step="B"), note(60, 0.5, typ="eighth", step="D")]
    vs = write._voices(apart, 10.0, single_staff_part=False, bar_len=F(2))
    assert sorted(vs) == [1, 2] and not any(e.chord for evs in vs.values() for e in evs)
    # so are notes at one time of different lengths, and a tuplet's notes
    mixed = [note(10, 0.0, typ="quarter", step="G"), note(12, 0.0, typ="eighth", step="B"), note(30, 0.5, typ="eighth", step="D")]
    vs = write._voices(mixed, 10.0, single_staff_part=False, bar_len=F(2))
    assert sorted(vs) == [1, 2] and not any(e.chord for evs in vs.values() for e in evs)
    triplet = [note(10 + k, 0.0, typ="eighth", step=s, tup="3/2") for k, s in enumerate("GB")]
    assert not any(e.chord for evs in write._voices(triplet, 10.0, single_staff_part=False, bar_len=F(2)).values() for e in evs)
    # with every voice of the staff taken, the voice is left as it was read and no note becomes a chord's
    full = [E(x=5 + n, kind="rest", typ="whole", voice=n, onset=0.0, onset_p=1.0) for n in (2, 3, 4)]
    crowded = [note(10, 0.0, typ="quarter", step="G"), note(11, 0.0, typ="eighth", step="B"), note(12, 0.0, typ="eighth", step="D")]
    vs = write._voices(crowded + full, 10.0, single_staff_part=False, bar_len=F(4))
    assert not any(e.chord for e in vs[1]) and any(e.kind == "gap" and e.dur < 0 for e in vs[1])
    # a run of notes stays in the lane it began in, though the first lane falls free under it: a triplet over a half
    # note keeps its three notes, its bracket and its beam in one voice
    run = [note(10, 0.0, typ="half", step="C")] + [note(20 + 10 * k, onset, typ="eighth", step="G", tup="3/2")
                                                    for k, onset in enumerate((1.0, 1 + 1 / 3, 1 + 2 / 3, 2.0, 2 + 1 / 3, 2 + 2 / 3))]
    vs = write._voices(run, 10.0, single_staff_part=False, bar_len=F(4))
    assert [e.step for e in vs[1] if e.kind == "note"] == ["C"] and [e.step for e in vs[2] if e.kind == "note"] == ["G"] * 6
    assert [e.tuplet_mark for e in vs[2] if e.kind == "note"] == ["start", "", "stop", "start", "", "stop"]


def test_a_tie_follows_its_note_through_a_voice_of_its_own(monkeypatch):
    """Ties are paired along the voice as the model read it. Paired along the voices as written, a tied note moved to
    a voice of its own for one bar lost its tie, or was tied to whatever stood in that voice number bars later and
    gave it its accidental."""
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    flat = {"alter": {"v": -1}}
    # bar 1: a whole E4 and, at beat 3, a half B flat tied over the bar line; bar 2: the B flat, then D5s;
    # bar 3: a whole E4 under a B natural half and a D5 half
    notes = [(40, "note_whole", -4, 0.0, 0.0, {}), (120, "note_half", 0, 2.0, 1.0, flat),
             (240, "note_quarter", 0, 0.0, 0.0, flat), (280, "note_quarter", 2, 1.0, 0.0, {}), (320, "note_quarter", 2, 2.0, 0.0, {}),
             (360, "note_quarter", 2, 3.0, 0.0, {}),
             (440, "note_whole", -4, 0.0, 0.0, {}), (452, "note_half", 0, 0.0, 0.0, {}), (520, "note_half", 2, 2.0, 0.0, {})]
    dets = staff_row(100, [0, 200, 400, 600]) + [det("meterSig_4_4", (10, 100, 20, 140))]
    dets += [det(cls, (x - 5, 115 - pos * 5, x + 5, 125 - pos * 5), staff_position=pos) for x, cls, pos, *_ in notes]
    L = front.build(dets, 700, 300)
    by_x, tok = {x: (cls, pos, onset, tie, more) for x, cls, pos, onset, tie, more in notes}, {}
    for sym in L.syms:
        real = {"real": {"p": 1.0, "det": 1.0, "ev": 0}}
        if sym.fam == "measure":
            tok[sym.i] = {**real, "cls": {"v": "measure"}, "key": {"v": 0}, "time": {"v": "4/4"}, "clef": {"v": "G2"}}
        elif sym.fam == "meterSig":
            tok[sym.i] = {**real, "cls": {"v": "meterSig_4_4"}}
        else:
            cls, pos, onset, tie, more = by_x[round(sym.cx)]
            tok[sym.i] = {**real, "cls": {"v": cls}, "dots": {"v": 0}, "pos": {"v": pos}, "voice": {"v": 1}, "tup": {"v": "none"},
                          "alter": {"v": 0}, "grace": {"v": "none"}, "chord": 0.0, "tie": tie, "onset": onset, "onset_p": 1.0, **more}
    root = ET.fromstring(write.write(_reading(L, tok)).split("\n", 2)[2])
    bars = [[(n.findtext("pitch/step"), n.findtext("pitch/alter"), n.findtext("voice"), [t.get("type") for t in n.findall("tie")])
             for n in m.findall("note") if n.find("rest") is None] for m in root.iter("measure")]
    # the tied B flat keeps the voice and its tie; the E under it has a voice of its own
    assert bars[0] == [("B", "-1", "1", ["start"]), ("E", None, "2", [])]
    assert bars[1][0] == ("B", "-1", "1", ["stop"])
    # bar 3's B is natural and untied, whatever voice it is written in
    assert [(step, alter, ties) for step, alter, _, ties in bars[2]] == [("E", None, []), ("B", None, []), ("D", None, [])]
    # no voice holds two notes at once
    for m in root.iter("measure"):
        at, spans = 0, {}
        for n in m:
            if n.tag in ("backup", "forward"):
                at += int(n.findtext("duration")) * (-1 if n.tag == "backup" else 1)
            elif n.tag == "note" and n.find("chord") is None:
                d, v = int(n.findtext("duration")), n.findtext("voice")
                assert n.find("rest") is not None or all(not (a < at + d and at < b) for a, b in spans.get(v, []))
                spans.setdefault(v, []).append((at, at + d))
                at += d


def test_a_divided_bars_directions_are_written_at_their_own_time(monkeypatch):
    """The staff's directions and clef changes were written with its first voice, before the event they stand at. In
    a bar whose first voice is written in two, they go with the voice whose next event comes first on the page: a
    dynamic and a clef at the last quarter are written there, not after a whole note held under it, nor before the
    gap that leads to it."""
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)

    def bar(notes, extra, voice=1):
        dets = staff_row(100, [0, 200]) + [det("meterSig_4_4", (10, 100, 20, 140))]
        dets += [det(cls, (x - 5, 115 - pos * 5, x + 5, 125 - pos * 5), staff_position=pos) for x, cls, pos, _ in notes]
        dets += [det(cls, box) for cls, box in extra]
        L = front.build(dets, 300, 300)
        by_x, tok = {x: (cls, pos, onset) for x, cls, pos, onset in notes}, {}
        for sym in L.syms:
            real = {"real": {"p": 1.0, "det": 1.0, "ev": 0}}
            if sym.fam == "measure":
                tok[sym.i] = {**real, "cls": {"v": "measure"}, "key": {"v": 0}, "time": {"v": "4/4"}, "clef": {"v": "G2"}}
            elif round(sym.cx) in by_x and sym.fam == "note":
                cls, pos, onset = by_x[round(sym.cx)]
                tok[sym.i] = {**real, "cls": {"v": cls}, "dots": {"v": 0}, "pos": {"v": pos}, "voice": {"v": voice}, "tup": {"v": "none"},
                              "alter": {"v": 0}, "grace": {"v": "none"}, "chord": 0.0, "tie": 0.0, "onset": onset, "onset_p": 1.0}
            else:
                tok[sym.i] = {**real, "cls": {"v": sym.cls}}
        root = ET.fromstring(write.write(_reading(L, tok)).split("\n", 2)[2])
        out = []
        for el in root.find(".//measure"):
            if el.tag == "note":
                out.append("rest" if el.find("rest") is not None else el.findtext("pitch/step") + el.findtext("voice"))
            elif el.tag in ("backup", "forward"):
                out.append(el.tag + el.findtext("duration"))
            elif el.tag == "direction":
                out.append("dynamic")
            elif el.tag == "attributes" and el.find("clef") is not None and el.find("time") is None:
                out.append("clef")
        return out

    whole, last = (40, "note_whole", -4, 0.0), (170, "note_quarter", 2, 3.0)
    quarters = [(44, "note_quarter", 0, 0.0), (80, "note_quarter", 0, 1.0), (120, "note_quarter", 0, 2.0)]
    marks = [("dynam_f", (115, 150, 125, 160)), ("clef_F4", (145, 100, 155, 140))]
    # a whole note under four quarters, a dynamic at the third and a clef before the fourth
    assert bar([whole, *quarters, last], marks) == ["E1", "backup4", "B2", "B2", "dynamic", "B2", "clef", "F2"]
    # the lane of a lone quarter at beat 4 starts with a gap: its clef and dynamic come after the gap, at the quarter
    assert bar([whole, last], [("dynam_f", (165, 150, 175, 160)), marks[1]]) == ["E1", "backup4", "forward3", "clef", "dynamic", "F2"]
    # every note read as the staff's fourth voice: the lane takes a lower number, and the directions stay with the
    # voice as read
    assert bar([(44, "note_whole", -4, 0.0), (40, "note_quarter", 0, 0.0), *quarters[1:], last], marks, voice=4) == \
        ["E1", "backup4", "B4", "B4", "dynamic", "B4", "clef", "F4"]


def test_a_chord_read_as_separate_notes_keeps_its_beam_and_each_voice_under_a_beam_is_beamed(monkeypatch):
    """A left hand's bass note and chord in turn, four eighths under one beam: the model reads the chord's notes as
    events of their own at one onset. They are the bass line's chords, in its voice, and the beam runs over all four.
    Where a beam's notes are written in two voices because their onsets overlap, the notes in each are beamed."""
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)

    def bar(notes, beam):
        dets = staff_row(100, [0, 200]) + [det("meterSig_2_4", (10, 100, 20, 140)), det("beam", beam)]
        dets += [det("note_eighth", (x - 5, 115 - pos * 5, x + 5, 125 - pos * 5), staff_position=pos) for x, pos, _ in notes]
        L = front.build(dets, 300, 300)
        by, tok = {(x, pos): onset for x, pos, onset in notes}, {}
        for sym in L.syms:
            real = {"real": {"p": 1.0, "det": 1.0, "ev": 0}}
            if sym.fam == "measure":
                tok[sym.i] = {**real, "cls": {"v": "measure"}, "key": {"v": 0}, "time": {"v": "2/4"}, "clef": {"v": "G2"}}
            elif sym.fam == "note":
                onset = by[(round(sym.cx), round((120 - sym.cy) / 5))]
                tok[sym.i] = {**real, "cls": {"v": "note_eighth"}, "dots": {"v": 0}, "pos": {"v": round((120 - sym.cy) / 5)}, "voice": {"v": 1},
                              "tup": {"v": "none"}, "alter": {"v": 0}, "grace": {"v": "none"}, "chord": 0.0, "tie": 0.0, "onset": onset, "onset_p": 1.0}
            else:
                tok[sym.i] = {**real, "cls": {"v": sym.cls}}
        root = ET.fromstring(write.write(_reading(L, tok)).split("\n", 2)[2])
        return [(n.findtext("pitch/step"), n.findtext("voice"), n.find("chord") is not None, [b.text for b in n.findall("beam")])
                for n in root.find(".//measure").findall("note") if n.find("rest") is None]

    beam = (40, 80, 172, 86)
    # G, then B and D one over the other at the second eighth, then A, then C and E: one voice, chords, one beam
    hand = [(45, -4, 0.0), (85, 0, 0.5), (86, 2, 0.5), (125, -3, 1.0), (165, 1, 1.5), (166, 3, 1.5)]
    assert bar(hand, beam) == [("E", "1", False, ["begin"]), ("B", "1", False, ["continue"]), ("D", "1", True, []),
                               ("F", "1", False, ["continue"]), ("C", "1", False, ["end"]), ("E", "1", True, [])]
    # four eighths under one beam whose third is read before the second has ended: two voices, each beamed
    run = [(45, 0, 0.0), (85, 0, 0.5), (125, 0, 0.75), (165, 0, 1.25)]
    written = bar(run, beam)
    assert sorted(v for _, v, _, _ in written) == ["1", "1", "2", "2"]
    assert all(beams in (["begin"], ["end"]) for _, _, _, beams in written)


def test_a_line_with_a_percussion_clef_on_it_is_a_one_line_staff():
    from copisteria.vision.page.staves import Line
    staves = [front.Staff(bars=[], sp=10.0, y0=100.0, y1=140.0, x0=0.0, x1=400.0),
              front.Staff(bars=[], sp=10.0, y0=300.0, y1=340.0, x0=0.0, x1=400.0)]
    line = lambda y, x1=400: Line(y=y, x0=0, x1=x1, thickness=2, score=0.5)      # noqa: E731
    clef = lambda y: det("clef_perc", (20, y - 10, 30, y + 10))                  # noqa: E731
    # the triangle's line between the two staves, its clef on it
    (found,) = front._one_line_staves([line(220)], staves, front.merge([clef(221)]))
    assert (found.y0, found.y1, found.sp, found.x0, found.x1) == (200.0, 240.0, 10.0, 0.0, 400.0)
    # a drum kit's clef stands on a five-line staff; a short rule, a line without a clef and a staff's own line are none
    assert front._one_line_staves([line(120)], staves, front.merge([clef(120)])) == []
    assert front._one_line_staves([line(220, x1=100)], staves, front.merge([clef(221)])) == []
    assert front._one_line_staves([line(220)], staves, front.merge([det("clef_G2", (20, 210, 30, 230))])) == []
    assert front._one_line_staves([line(160)], staves, front.merge([clef(160)])) == []


def test_a_layout_the_caller_knows_groups_the_staves_into_its_systems(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    voice, right, left = staff_row(100, [0, 150, 300]), staff_row(180, [0, 100, 200, 300]), staff_row(260, [0, 100, 200, 300])
    # a voice whose bar lines do not line up with the piano's under it: two systems to the front end, one to the page
    L = front.build(voice + right + left, 400, 400)
    assert [len(sy.staves) for sy in L.systems] == [1, 2] and L.prior is None
    L = front.build(voice + right + left, 400, 400, layout=[3])
    assert [len(sy.staves) for sy in L.systems] == [3]
    assert L.prior["used"] and L.prior["own"] == [1, 2] and L.prior["target"] == [3]
    assert [L.staves[k].system for k in range(3)] == [0, 0, 0] and [L.staves[k].pos for k in range(3)] == [0, 1, 2]
    # a layout joins, it does not part: two staves the front end joins stay one system, whatever was counted
    L = front.build(right + left, 400, 400, layout=[1, 1])
    assert [len(sy.staves) for sy in L.systems] == [2] and L.prior["why"] == "the layout parts staves the page joins"
    # a layout that says what the front end found changes nothing
    assert [len(sy.staves) for sy in front.build(right + left, 400, 400, layout=[2]).systems] == [2]


def test_a_layout_drops_only_what_is_hardly_a_staff_and_is_not_used_where_it_cannot_be_met(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    right, left = staff_row(180, [0, 100, 200, 300]), staff_row(260, [0, 100, 200, 300])
    notes = [det("note_quarter", (50 + 100 * k, 195, 60 + 100 * k, 205), staff_position=0) for k in range(3)]
    inset = staff_row(60, [200, 300])                               # an ossia over one bar
    L = front.build(inset + right + left + notes, 400, 400, layout=[2])
    assert [len(sy.staves) for sy in L.systems] == [2] and L.prior["dropped"] == 1 and L.staves[0].y0 == 180
    # a third staff as long as the others is no inset, silent or not (the left hand above holds no note either):
    # the layout is not used, and says why
    L = front.build(staff_row(60, [0, 100, 200, 300]) + right + left + notes, 400, 400, layout=[2])
    assert len(L.staves) == 3 and not L.prior["used"] and L.prior["why"] == "3 staves found and 2 counted"
    assert [len(sy.staves) for sy in L.systems] == L.prior["own"]
    # a layout that counts nothing is not one
    assert front.build(right + left, 400, 400, layout=[]).prior["used"] is False


def test_a_layout_that_counts_more_staves_takes_the_likeliest_the_finder_left_if_something_stands_on_it():
    staves = [front.Staff(bars=[], sp=10.0, y0=100.0, y1=140.0, x0=0.0, x1=400.0),
              front.Staff(bars=[], sp=10.0, y0=300.0, y1=340.0, x0=0.0, x1=400.0)]
    spare = [front.Staff(bars=[], sp=10.0, y0=200.0, y1=240.0, x0=0.0, x1=400.0),
             front.Staff(bars=[], sp=10.0, y0=420.0, y1=460.0, x0=0.0, x1=400.0)]
    on = front.merge([det("rest_quarter", (50 + 100 * k, 215, 60 + 100 * k, 225)) for k in range(2)])
    said = lambda sts: ([False] * (len(sts) - 1), ["white"] * (len(sts) - 1))      # noqa: E731
    prior, fitted, linked = front._with_layout([3], on, staves, spare, said, add=True)
    assert prior["used"] and prior["added"] == 1 and [st.y0 for st in fitted] == [100.0, 200.0, 300.0] and linked == [True, True]
    # a spare line nothing stands on is no staff; nor is any on a page read from its measure boxes
    prior, fitted, linked = front._with_layout([3], [], staves, spare, said, add=True)
    assert not prior["used"] and fitted == staves and linked == [False]
    assert not front._with_layout([3], on, staves, spare, said, add=False)[0]["used"]


def test_a_layout_does_not_part_what_the_page_joins_nor_join_bar_lines_that_do_not_line_up(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    a, b = staff_row(100, [0, 100, 200, 300, 400]), staff_row(180, [0, 100, 200, 300, 400])
    L = front.build(a + b, 500, 300)
    for said in ("line", "unclear"):
        joins = lambda sts: ([True] * (len(sts) - 1), [said] * (len(sts) - 1))     # noqa: E731
        prior, fitted, linked = front._with_layout([1, 1], L.syms, L.staves, [], joins, add=False)
        assert not prior["used"] and prior["why"] == "the layout parts staves the page joins" and linked == [True]
    # two staves whose bar lines fall in different places are not one system, whatever the layout counts
    c = staff_row(180, [0, 150, 250, 350, 400])
    L = front.build(a + c, 500, 300)
    assert [len(sy.staves) for sy in L.systems] == [1, 1]
    L = front.build(a + c, 500, 300, layout=[2])
    assert [len(sy.staves) for sy in L.systems] == [1, 1] and L.prior["why"] == "the layout joins staves whose bar lines do not line up"


def test_one_staff_out_of_line_does_not_part_a_system_whose_other_staves_line_up_across_it():
    """A band score's staff of rests shows a few measure boxes that end two or three staff spaces off, or a different
    few of its system's bar lines than its neighbour: the two staves at that join 'do not line up', and a stated
    ten-staff system was refused for it. A join is judged with the system the layout forms: where a staff above it
    lines up with a staff below it, the staves are one system."""
    said = lambda sts: ([False] * (len(sts) - 1), ["white"] * (len(sts) - 1))      # noqa: E731

    def page(rows):
        dets = [d for y, xs in rows for d in staff_row(y, xs)]
        L = front.build(dets, 700, 100 * len(rows) + 200)
        return L.syms, L.staves

    full = [0, 100, 200, 300, 400, 500, 600]
    # six staves of one system: two that show every bar line, a staff whose boxes end 30 off, and three that show few
    syms, staves = page([(100, full), (200, [0, 100, 400, 600]), (300, [0, 230, 530, 600]), (400, full),
                         (500, [0, 100, 300, 600]), (600, [0, 200, 400, 600])])
    assert front._apart(syms, staves[1], staves[2]) and front._apart(syms, staves[2], staves[3])
    assert front._lined_up(syms, staves[0], staves[3]) and not front._lined_up(syms, staves[1], staves[2])
    prior, fitted, linked = front._with_layout([6], syms, staves, [], said, add=False)
    assert prior["used"] and linked == [True] * 5
    # four lines of one part, each with bar lines of its own, are not one system whatever is counted
    syms, staves = page([(100 * (k + 1), [0, 100 + 30 * k, 230 + 30 * k, 360 + 30 * k, 490 + 30 * k, 600]) for k in range(4)])
    assert not any(front._lined_up(syms, a, b) for a in staves for b in staves if a is not b)
    prior, fitted, linked = front._with_layout([4], syms, staves, [], said, add=False)
    assert not prior["used"] and prior["why"] == "the layout joins staves whose bar lines do not line up" and linked == [False] * 3
    # nor are two systems with different bar lines counted as one: no staff of the upper lines up with one of the lower
    syms, staves = page([(100, full), (200, [0, 100, 400, 600]), (300, [0, 200, 500, 600]),
                         (400, [0, 150, 250, 350, 450, 550, 600]), (500, [0, 150, 450, 600]), (600, [0, 250, 550, 600])])
    assert not front._joined_around(syms, staves[:3], staves[3:])
    assert not front._with_layout([6], syms, staves, [], said, add=False)[0]["used"]
    # nor three systems whose first and third are alike: only the staves next to a join speak for it
    other = [0, 150, 250, 350, 450, 550, 600]
    syms, staves = page([(100, full), (200, full), (300, full), (400, other), (500, other), (600, other), (700, full), (800, full), (900, full)])
    assert front._lined_up(syms, staves[0], staves[8]) and not front._joined_around(syms, staves[:3], staves[3:])
    assert not front._with_layout([9], syms, staves, [], said, add=False)[0]["used"]
    # a staff that shows one bar line says nothing either way, and two staves alone are judged as before
    assert not front._lined_up(syms, staves[0], front.Staff(bars=[], sp=10.0, y0=700.0, y1=740.0, x0=0.0, x1=600.0))


def test_a_join_a_layout_adds_has_to_fit_the_pages_spacing_and_the_scores_size(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)
    rows = lambda ys: sum((staff_row(y, [0, 112, 195, 300] if n % 2 else [0, 100, 200, 300]) for n, y in enumerate(ys)), [])     # noqa: E731
    # four staves the front end reads as four systems (their bar lines waver between them): two tight pairs, a wide gap
    L = front.build(rows([100, 180, 400, 480]), 400, 600)
    assert [len(sy.staves) for sy in L.systems] == [1, 1, 1, 1]
    assert [len(sy.staves) for sy in front.build(rows([100, 180, 400, 480]), 400, 600, layout=[2, 2]).systems] == [2, 2]
    # counted as [1, 3] the layout would join across the wide gap and leave the tight one between two systems
    L = front.build(rows([100, 180, 400, 480]), 400, 600, layout=[1, 3])
    assert not L.prior["used"] and L.prior["why"] == "the layout joins staves as far apart as two of its systems"
    # and a system of more staves than the score has is no system of this score
    L = front.build(rows([100, 180, 400, 480]), 400, 600, layout=[2, 2], most=1)
    assert not L.prior["used"] and L.prior["why"] == "the layout counts a system of 2 staves and the score has 1"


def test_a_join_that_loses_the_bars_its_staves_show_does_not_fit_the_page():
    two = [front.System(staves=[0, 1], columns=[[0, 1], [0, 1]]), front.System(staves=[2], columns=[[0, 1, 2, 3, 4]])]
    # staves showing five and four bars, joined into a system of two: the join is not the page's
    assert front._loses_bars(two, [5, 4, 5], joined=[1])
    # four bars of five is the usual wobble of one missed bar line; a system the layout did not form is not judged
    four = [front.System(staves=[0, 1], columns=[[0, 1, 2, 3], [0, 1, 2, 3]])]
    assert not front._loses_bars(four, [5, 4], joined=[1])
    assert not front._loses_bars(two, [5, 4, 5], joined=[])
