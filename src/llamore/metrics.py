import multiprocessing as mp
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel
from rapidfuzz.distance import Levenshtein
from rich.progress import track
from scipy.optimize import linear_sum_assignment

from llamore.reference import Organization, Person, Reference, References


class F1:
    """Compare predictions and labels and compute a F1 score.

    Args:
        levenshtein_distance:
            If int, this is the maximum Levenshtein distance that still counts as a match.
            If float, this is the minimum normalized distance that still counts as a match.

            The normalized distance between two strings is computed as:
                1 - Levenshtein.distance(prediction, label) / max(len(prediction), len(label))
            => 1.0 means an exact match, 0.0 means no match.

            Set this to int(0) or float(1.0) to require an exact match.
    """

    def __init__(self, levenshtein_distance: Union[int, float] = 0):
        if levenshtein_distance < 0 or (
            isinstance(levenshtein_distance, float) and levenshtein_distance > 1.0
        ):
            raise ValueError(
                "`levenshtein_distance` must be non-negative and, if a float, between 0 and 1."
            )

        self.levensthein_distance = levenshtein_distance

    def compute_macro_average(
        self,
        predictions: Union[References, List[References]],
        labels: Union[References, List[References]],
        show_progress: bool = True,
        num_processes: Optional[int] = None,
    ) -> float:
        """Compute the macro average of the f1 scores between the predictions and the labels.

        Args:
            predictions: The predicted references.
            labels: The ground truth references.
            show_progress: Whether to show a progress bar.
            num_processes: Number of processes to use for parallel computation. If 0, no parallelization is used.
                By default, the number of available CPUs is used.

        Returns:
            The macro average f1 score.
        """
        if num_processes is None:
            # Each CPU should process at least 10 predictions
            num_processes = min(len(predictions) // 10, mp.cpu_count())

        # the next-iter line covers also empty `References`
        if not isinstance(next(iter(predictions), None), list):
            predictions = [predictions]
        if not isinstance(next(iter(labels), None), list):
            labels = [labels]

        f1_scores = []

        if num_processes > 0:
            with mp.Pool(num_processes) as p:
                for f1_score in track(
                    p.imap_unordered(
                        partial(
                            _parallel_compute_f1s,
                            levenshtein_distance=self.levensthein_distance,
                        ),
                        zip(predictions, labels),
                    ),
                    total=len(predictions),
                    disable=not show_progress,
                ):
                    f1_scores += f1_score
        else:
            for prediction, label in track(
                zip(predictions, labels),
                total=len(predictions),
                disable=not show_progress,
            ):
                f1_score = self.compute(prediction, label)
                if isinstance(f1_score, float):
                    f1_score = [f1_score]
                f1_scores += f1_score

        return float(np.mean(f1_scores))

    def compute(
        self,
        prediction: Union[Reference, References],
        label: Union[Reference, References],
    ) -> Union[float, List[float]]:
        """Compute the F1s and match the most similar predictions and labels.

        Args:
            predictions: The predicted references.
            labels: The ground truth references.

        Returns:
            The f1 score(s).
        """
        # Special definition of undefined F1 ...
        if prediction == label == References():
            return 1.0

        if not isinstance(prediction, list):
            prediction = References([prediction])
        if not isinstance(label, list):
            label = References([label])
        result = self._compute_f1s(prediction, label)

        if len(result) == 1:
            return result[0]
        return result

    def _compute_f1s(self, prediction: References, label: References) -> List[float]:
        """Compute the F1s and match the most similar predictions and labels.

        Args:
            predictions:
            labels:

        Returns:
            The f1 scores.
        """
        f1_matrix = np.zeros((len(prediction), len(label)))
        for i, pred in enumerate(prediction):
            for j, lab in enumerate(label):
                f1_matrix[i, j] = self._compute_f1(pred, lab)

        idx = linear_sum_assignment(f1_matrix * -1)
        f1s = [float(f1) for f1 in f1_matrix[idx]]

        # add zeros for missing/extra references
        if f1_matrix.shape[0] != f1_matrix.shape[1]:
            f1s += [0.0 for _ in range(np.abs(f1_matrix.shape[0] - f1_matrix.shape[1]))]

        return f1s

    def _compute_f1(self, prediction: Reference, label: Reference) -> float:
        """Compute the F1 score between a predicted and a gold label reference."""
        # count matches
        total_matches = self._count_matches(prediction, label)

        # count predictions
        n_predictions = self._count_not_nones(prediction)

        # count labels
        n_labels = self._count_not_nones(label)

        return precision_recall_f1(total_matches, n_labels, n_predictions)["f1"]

    def _count_matches(
        self,
        prediction: Union[None, str, int, List, BaseModel],
        label: Union[None, str, int, List, BaseModel],
    ) -> int:
        """Recursively counts the matches between two BaseModels.

        Args:
            prediction: The prediction.
            label: The gold label.

        Return:
            Number of matches.

        Raises:
            TypeError if prediction and label are not the same type (except for Nones).
        """
        if prediction is None or label is None:
            return 0

        if isinstance(prediction, BaseModel):
            # Check if mixing Person and Organization
            if (
                isinstance(prediction, (Person, Organization))
                and isinstance(label, (Person, Organization))
                and type(prediction) is not type(label)
            ):
                return 0

            self._check_for_same_type(prediction, label)

            tot_matches = 0
            for field, info in prediction.model_fields.items():
                if info.exclude:
                    continue
                tot_matches += self._count_matches(
                    getattr(prediction, field),
                    getattr(label, field),
                )
            return tot_matches

        if isinstance(prediction, list):
            self._check_for_same_type(prediction, label)

            match_matrix = np.zeros((len(prediction), len(label)), dtype=np.int32)
            for i, pred in enumerate(prediction):
                for j, lab in enumerate(label):
                    match_matrix[i, j] = self._count_matches(pred, lab)

            if match_matrix.sum() == 0:
                return 0

            idx = linear_sum_assignment(match_matrix * -1)
            matches = match_matrix[idx].sum()

            return int(matches)

        if isinstance(prediction, str):
            self._check_for_same_type(prediction, label)

            if self._is_match(prediction, label):
                return 1
            return 0

        else:
            raise TypeError(f"Type '{type(prediction)}' not supported!")

    @staticmethod
    def _check_for_same_type(prediction: Any, label: Any):
        if type(prediction) is not type(label):
            raise TypeError(
                f"Trying to compare different types: '{type(prediction)}' (prediction) and '{type(label)}' (label)"
            )

    def _is_match(self, prediction: str, label: str) -> bool:
        """Check if two strings are a match taking into account their Levenshtein distance.

        Args:
            prediction: The prediction.
            label: The gold label.

        Returns:
            True if the strings are a match, False otherwise.
        """
        if isinstance(self.levensthein_distance, int):
            return self._is_match_with_max_distance(
                prediction, label, self.levensthein_distance
            )

        if isinstance(self.levensthein_distance, float):
            return self._is_match_with_min_distance(
                prediction, label, self.levensthein_distance
            )

    @staticmethod
    def _is_match_with_max_distance(
        prediction: str, label: str, max_distance: int
    ) -> bool:
        """Check if two strings are a match taking into account their Levenshtein distance.

        Args:
            prediction: The prediction.
            label: The gold label.
            max_distance: The maximum Levenshtein distance that still counts as a match.

        Returns:
            True if the strings are a match, False otherwise.
        """
        if max_distance == 0:
            if prediction == label:
                return True
            return False

        distance = Levenshtein.distance(prediction, label, score_cutoff=max_distance)
        if distance <= max_distance:
            return True
        return False

    @staticmethod
    def _is_match_with_min_distance(
        prediction: str, label: str, min_distance: float
    ) -> bool:
        """Check if two strings are a match taking into account their Levenshtein distance.

        Args:
            prediction: The prediction.
            label: The gold label.
            min_distance: The minimum normalized distance that still counts as a match.

        Returns:
            True if the strings are a match, False otherwise.
        """
        if min_distance >= 1.0:
            if prediction == label:
                return True
            return False

        length = max(len(prediction), len(label))
        score_cutoff = int((1 - min_distance) * length)
        distance = Levenshtein.distance(prediction, label, score_cutoff=score_cutoff)
        normalized_distance = 1 - distance / length
        if normalized_distance >= min_distance:
            return True
        return False

    def _count_not_nones(
        self,
        value: Union[None, str, int, List, BaseModel],
    ) -> int:
        if value is None:
            return 0

        if isinstance(value, (str, int)):
            return 1

        if isinstance(value, BaseModel):
            tot_not_nones = 0
            for field, info in value.model_fields.items():
                if info.exclude:
                    continue
                tot_not_nones += self._count_not_nones(getattr(value, field))
            return tot_not_nones

        if isinstance(value, list):
            tot_not_nones = 0
            for v in value:
                tot_not_nones += self._count_not_nones(v)
            return tot_not_nones


def compute_coarse_f1(
    predictions: List[List[Reference]], labels: List[List[Reference]]
) -> Dict[str, float]:
    """Compute a coarse F1 score, in which we only check if the entire references match exactly.

    Args:
        predictions:
        labels:

    Returns:
        A dict with the keys: precision, recall, f1
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"`predictions` and `labels` must have the same length: {len(predictions)} vs. {len(labels)}"
        )

    complete_matches = 0

    for prediction, label in zip(predictions, labels):
        for pred in prediction:
            complete_matches += pred in label

    n_predictions = sum([len(prediction) for prediction in predictions])
    n_labels = sum([len(label) for label in labels])

    return precision_recall_f1(complete_matches, n_labels, n_predictions)


def precision_recall_f1(
    matches: int, labels: int, predictions: int
) -> Dict[str, float]:
    """Compute the precision, recall and F1.

    Args:
        matches: Number of matches.
        labels: Number of labels or label fields, that are not None.
        predictions: Number of predictions or prediction fields, that are not None.

    Returns:
        A dict with 'precision', 'recall' and 'f1'.
    """
    precision = matches / (predictions or 1)
    recall = matches / (labels or 1)
    f1 = 2 * precision * recall / ((precision + recall) or 1)

    return {"precision": precision, "recall": recall, "f1": f1}


def _parallel_compute_f1s(
    pred_label_tuple: Tuple[List[Reference], List[Reference]],
    levenshtein_distance: Union[int, float] = 0,
):
    """Helper function for the parallel computation of the F1 macro averaged score."""
    f1 = F1(levenshtein_distance=levenshtein_distance)
    prediction, label = pred_label_tuple
    f1_scores = f1.compute(prediction, label)
    if isinstance(f1_scores, float):
        f1_scores = [f1_scores]

    return f1_scores
