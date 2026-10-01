"""Out-of-fold scores of simple baseline classifiers.

Used to ask how much of a signal could be predicted from the input alone
(for example from the question text), without the model's answers. Every
score is out of fold: each row is scored by a model that never saw it.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

FOLDS = 5
"""Cross-validation folds."""


def _char_ngrams() -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
    )


def oof_text_scores(
    texts: Sequence[str], labels: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Out-of-fold scores of a character n-gram logistic regression.

    Folds are stratified and shuffled with ``seed``; the TF-IDF vocabulary
    (character 2-5-grams within word boundaries) is fitted inside each fold.

    Args:
        texts: One text per row.
        labels: Binary labels.
        seed: Fold seed.

    Returns:
        Out-of-fold decision scores (higher = positive).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(labels, int)
    out = np.empty(len(y))
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    for train, test in folds.split(np.zeros(len(y)), y):
        vec = _char_ngrams()
        x_train = vec.fit_transform([texts[i] for i in train])
        model = LogisticRegression(C=4.0, max_iter=3000)
        model.fit(x_train, y[train])
        out[test] = model.decision_function(vec.transform([texts[i] for i in test]))
    return out


def oof_features(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Out-of-fold scores of a logistic regression on a few numeric features.

    Args:
        features: Matrix, one row per item.
        labels: Binary labels.
        seed: Fold seed (stratified, shuffled folds).

    Returns:
        Out-of-fold decision scores.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(labels, int)
    x = np.asarray(features, float)
    out = np.empty(len(y))
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    for train, test in folds.split(x, y):
        model = LogisticRegression(max_iter=2000).fit(x[train], y[train])
        out[test] = model.decision_function(x[test])
    return out


def grouped_oof(
    make: Callable[[], Any],
    x: Any,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    text: bool = False,
) -> np.ndarray:
    """Out-of-fold scores with folds grouped so no group spans train and test.

    Args:
        make: Factory for a fresh scikit-learn classifier.
        x: Feature matrix or, with ``text``, a list of strings.
        y: Binary labels.
        groups: Group id per row.
        text: Fit a character n-gram TF-IDF inside each fold.

    Returns:
        Out-of-fold positive-class probabilities.
    """
    from sklearn.model_selection import GroupKFold

    labels = np.asarray(y, int)
    out = np.empty(len(labels))
    for train, test in GroupKFold(FOLDS).split(np.zeros(len(labels)), labels, groups):
        if text:
            vec = _char_ngrams()
            xtr = vec.fit_transform([x[i] for i in train])
            xte = vec.transform([x[i] for i in test])
        else:
            xtr, xte = x[train], x[test]
        model = make().fit(xtr, labels[train])
        out[test] = model.predict_proba(xte)[:, 1]
    return out
