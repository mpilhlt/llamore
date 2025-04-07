import math
from typing import List, Tuple

import numpy as np
import pytest
from llamore.metrics import F1, compute_coarse_f1
from llamore.reference import Person, Reference, References, Organization


@pytest.fixture
def data() -> Tuple[List[Reference], List[Reference]]:
    prediction = [
        Reference(
            analytic_title="at2",
            journal_title="jt",
            authors=[Person(forename="first", surname="last2")],
        )
    ]

    label = [
        Reference(
            analytic_title="at",
            journal_title="jt",
            authors=[
                Person(forename="first", surname="last"),
                Person(surname="last2"),
            ],
        ),
        Reference(
            analytic_title="at2",
            journal_title="jt",
            authors=[
                Person(forename="first", surname="last"),
                Person(forename="first", surname="last2"),
            ],
        ),
    ]

    return prediction, label


def test_count_matches(data):
    f1 = F1()
    assert f1._count_matches(["a", "b"], ["b"]) == 1

    assert f1._count_matches(data[0][0], data[1][1]) == 4
    assert f1._count_matches(data[0][0], data[1][0]) == 2


def test_count_matches_levenshtein(data):
    f1 = F1(levenshtein_distance=1)
    assert f1._count_matches(data[0][0], data[1][0]) == 4
    assert f1._count_matches("test", "taut") == 0
    f1 = F1(levenshtein_distance=2)
    assert f1._count_matches("test", "taut") == 1

    f1 = F1(levenshtein_distance=0.5)
    assert f1._count_matches("test", "tess") == 1
    assert f1._count_matches("test", "tets") == 1
    assert f1._count_matches("test", "tats") == 0


def test_count_not_nones(data):
    f1 = F1()
    assert f1._count_not_nones(data[0][0]) == 4
    assert f1._count_not_nones(data[1][0]) == 5


def test_compute_f1(data):
    f1 = F1()._compute_f1(data[0][0], data[1][1])
    assert np.allclose(f1, 0.8)

    f1 = F1()._compute_f1(data[0][0], data[1][0])
    assert np.allclose(f1, 1.0 / 2.25)


def test_compute_f1s(data):
    f1s = F1()._compute_f1s(data[0], data[1])
    assert np.allclose(f1s, [0.8, 0.0])


def test_compute_f1_macro_average(data):
    f1 = F1().compute_macro_average([data[0]], [data[1]])
    assert f1 == 0.4


def test_compute(data):
    f1 = F1().compute(data[0][0], data[1][1])
    assert isinstance(f1, float)
    assert np.allclose(f1, 0.8)

    f1 = F1().compute(data[0][0], data[1][0])
    assert np.allclose(f1, 1.0 / 2.25)

    f1s = F1().compute(data[0], data[1])
    assert isinstance(f1s, list)
    assert np.allclose(f1s, [0.8, 0.0])


def test_compute_coarse_f1():
    f1_score = compute_coarse_f1(
        predictions=[["a", "b"], ["c"]], labels=[["a", "b"], ["c"]]
    )
    for metric in ["recall", "precision", "f1"]:
        assert math.isclose(f1_score[metric], 1.0)

    f1_score = compute_coarse_f1(
        predictions=[["a", "b"], ["c"]], labels=[["a", "b"], ["d", "e"]]
    )
    assert math.isclose(f1_score["recall"], 0.5)
    assert math.isclose(f1_score["precision"], 2.0 / 3)
    assert math.isclose(f1_score["f1"], 0.5714285714285715)

    f1_score = compute_coarse_f1(
        predictions=[["a", "b"], ["c"]], labels=[["c", "d"], ["e"]]
    )
    for metric in ["recall", "precision", "f1"]:
        assert math.isclose(f1_score[metric], 0.0)


def test_is_match_with_min_max_distance():
    assert F1()._is_match_with_min_distance("aaaaaaaaaa", "aaaaaaaaab", 0.90)
    assert not F1()._is_match_with_min_distance("aaaaaaaaaa", "aaaaaaaaab", 0.91)

    assert F1()._is_match_with_max_distance("aaaaaaaaaa", "aaaaaaaabb", 2)
    assert not F1()._is_match_with_max_distance("aaaaaaaaaa", "aaaaaaaaab", 0)


def test_micro_average():
    ref = Reference(
        analytic_title="a",
        authors=[Person(forename="a", surname="b"), Person(forename="a", surname="d")],
    )
    gold = Reference(
        analytic_title="a",
        journal_title="jt",
        authors=[Person(forename="a", surname="d")],
    )

    f1 = F1()
    metrics = f1.compute_micro_average(ref, gold)

    assert metrics == {
        "analytic_title": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        "journal_title": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        "authors.forename": {"recall": 1.0, "precision": 0.5, "f1": 0.5},
        "authors.surname": {"recall": 1.0, "precision": 0.5, "f1": 0.5},
    }


def test_count_stats_per_field():
    ref = Reference(
        analytic_title="title",
        publication_date="time",
        authors=[
            Person(forename="first", surname="last"),
            Person(forename="first2", surname="last2"),
        ],
    )
    gold = Reference(
        analytic_title="title",
        journal_title="jt",
        authors=[Person(forename="first", surname="last0"), Organization(name="org")],
        publication_place="place",
    )

    stats = F1()._count_stats_per_field(ref, gold)

    assert stats == {
        "Reference.analytic_title": {"predictions": 1, "labels": 1, "matches": 1},
        "Reference.journal_title": {"predictions": 0, "labels": 1, "matches": 0},
        "Reference.publication_date": {"predictions": 1, "labels": 0, "matches": 0},
        "Reference.publication_place": {"predictions": 0, "labels": 1, "matches": 0},
        "Reference.authors.Person.forename": {
            "predictions": 2,
            "labels": 1,
            "matches": 1,
        },
        "Reference.authors.Person.surname": {
            "predictions": 2,
            "labels": 1,
            "matches": 0,
        },
        "Reference.authors.Organization.name": {
            "predictions": 0,
            "labels": 1,
            "matches": 0,
        },
    }


def test_compute_micro_average():
    ref = Reference(
        analytic_title="a",
        journal_title="jt",
        authors=[Person(forename="a", surname="b"), Person(forename="a", surname="d")],
    )
    gold = Reference(
        analytic_title="a",
        journal_title="jt",
        authors=[Person(forename="a", surname="d")],
    )
    gold = Reference(
        analytic_title="a",
        journal_title="jt",
        authors=[Person(forename="a", surname="b"), Person(forename="a", surname="d")],
    )

    metrics = F1().compute_micro_average(References([ref]), References([gold]))

    assert metrics == {
        "micro_average": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        "Reference.analytic_title": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        "Reference.journal_title": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        "Reference.authors.Person.forename": {
            "recall": 1.0,
            "precision": 1.0,
            "f1": 1.0,
        },
        "Reference.authors.Person.surname": {
            "recall": 1.0,
            "precision": 1.0,
            "f1": 1.0,
        },
    }

    import pandas as pd
    print(pd.DataFrame(metrics).transpose())
    assert False