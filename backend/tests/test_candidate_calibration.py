from app.services import candidate_calibration as cal

P = ".video-datasets/"


def _state(main_images, cand_images, excluded=(), other=()):
    return {"characters": [{"id": 1, "images": list(main_images)},
                           {"id": 2, "pending": True, "section": "candidates", "parent": 1, "images": list(cand_images)},
                           {"id": 3, "pending": True, "section": "other", "images": list(other)}],
            "excluded": list(excluded)}


def _sections(n):
    return [{"section": "candidates", "main": 0, "files": [f"f{k}.png" for k in range(n)],
             "scores": [[k / 100, 0, 0, -2] for k in range(n)]}]


def test_answers_read_from_where_images_are_now():
    st = _state([P + "f0.png"], [P + "f2.png"], excluded=[P + "f1.png"], other=[P + "f3.png"])
    rows = {r["rel"]: r["label"] for r in cal.answers(_sections(4), st, P)}
    assert rows == {P + "f0.png": 1, P + "f1.png": 0, P + "f2.png": None, P + "f3.png": 0}


def test_line_found_when_top_answers_are_right():
    n = 100
    right = [P + f"f{k}.png" for k in range(40)]                 # best 40 accepted by the user
    wrong = [P + f"f{k}.png" for k in range(40, 50)]             # next 10 excluded
    rest = [P + f"f{k}.png" for k in range(50, n)]
    rows = cal.answers(_sections(n), _state(right, rest, excluded=wrong), P)
    res = cal.line(rows)
    assert res["answered"] == 50 and res["wrong"] == 10
    assert res["line"] is not None and res["line"]["precision"] >= cal.TARGET
    assert res["line"]["score"] < 0.40 + 1e-9                    # the line stops before the wrong answers


def test_no_line_when_answers_are_mixed():  # negative control: a bad ordering must not be trusted
    n = 80
    right = [P + f"f{k}.png" for k in range(0, n, 2)]
    wrong = [P + f"f{k}.png" for k in range(1, n, 2)]
    res = cal.line(cal.answers(_sections(n), _state(right, [], excluded=wrong), P))
    assert res["line"] is None


def test_no_line_with_too_few_answers():
    right = [P + f"f{k}.png" for k in range(10)]
    res = cal.line(cal.answers(_sections(20), _state(right, []), P))
    assert res["line"] is None


def test_sample_spreads_over_score_range():
    rows = [{"rel": f"r{k}", "score": k / 100, "label": None} for k in range(300)]
    s = cal.sample(rows, 30)
    idx = sorted(int(x[1:]) for x in s)
    assert len(set(idx)) == 30 and idx[0] < 10 and idx[-1] >= 290


def test_check_sample_answers_reach_the_images_between_them():
    n = 300
    secs = _sections(n)
    rels = [P + f for f in secs[0]["files"]]
    judged = {rels[k]: 1 for k in range(0, 200, 5)}               # 40 sampled answers, right, over the best 200
    judged.update({rels[k]: 0 for k in range(200, 300, 5)})       # 20 sampled answers below, wrong
    rows = cal.answers(secs, _state([], rels), P, judged)
    res = cal.line(rows)
    assert res["line"] is not None
    up = cal.above(rows, res["line"]["score"])
    assert 140 <= len(up) <= 160                                  # the unanswered between the right answers, not below them


def test_line_stops_where_the_tail_gets_worse():
    n = 300
    secs = _sections(n)
    rels = [P + f for f in secs[0]["files"]]
    judged = {rels[k]: 1 for k in range(0, 150, 5)}               # 30 right
    judged.update({rels[k]: (1 if k % 10 else 0) for k in range(150, 300, 5)})  # then half wrong
    res = cal.line(cal.answers(secs, _state([], rels), P, judged))
    assert res["line"] is None or res["line"]["score"] < 1.75


def test_above_only_undecided_within_line():
    rows = [{"rel": "a", "score": 0.1, "label": None}, {"rel": "b", "score": 0.1, "label": 1}, {"rel": "c", "score": 0.3, "label": None}]
    assert [r["rel"] for r in cal.above(rows, 0.2)] == ["a"]
