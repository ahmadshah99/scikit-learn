import os
import pytest
import textwrap

from sklearn import __version__
from sklearn.feature_extraction.text import CountVectorizer


def test_count_vectorizer_lowercase():
    base_url = "dev" if __version__.endswith(".dev0") else "stable"
    x = ['This is Problematic™.', 'THIS IS NOT']

    cv = CountVectorizer(
        lowercase=True,
        strip_accents='unicode',
        ngram_range=(1, 1)
    )

    x_v = cv.fit_transform(x)

    actual = cv.get_feature_names_out()
    expected = ['is', 'not', 'problematicTM', 'this']

    assert all([a == b for a, b in zip(actual, expected)])

def test_count_vectorizer_lowercase_with_param():
    base_url = "dev" if __version__.endswith(".dev0") else "stable"
    x = ['This is Problematic™.', 'THIS IS NOT']

    cv = CountVectorizer(
        lowercase=True,
        strip_accents='unicode',
        lower_after_strip_accents=True,
        ngram_range=(1, 1)
    )

    x_v = cv.fit_transform(x)

    actual = cv.get_feature_names_out()
    expected = ['is', 'not', 'problematictm', 'this']

    assert all([a == b for a, b in zip(actual, expected)])