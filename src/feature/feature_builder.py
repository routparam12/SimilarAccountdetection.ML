import jellyfish
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.preprocessing import (
    normalize_email,
    normalize_matchpoint,
    split_alpha_num,
)
from src.utils.config import PRIVACY_DOMAINS, PUBLIC_DOMAINS

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))


def sort_alpha(alpha: str) -> str:
    """Sort characters of alpha string — makes comparison order-invariant."""
    return "".join(sorted(alpha))


def char_overlap_ratio(a1: str, a2: str) -> float:
    """Fraction of shared unique characters between two strings."""
    if not a1 and not a2:
        return 1.0
    if not a1 or not a2:
        return 0.0
    set1 = set(a1)
    set2 = set(a2)
    overlap = len(set1 & set2)
    total = max(len(set1 | set2), 1)
    return overlap / total


def number_features(n1, n2):
    features = {}

    if not n1 and not n2:
        features["num_exact"] = 1
        features["num_contains"] = 1
        features["num_overlap_ratio"] = 1
        features["num_same_length"] = 1
        features["num_prefix_ratio"] = 1
        return features

    features["num_exact"] = int(n1 == n2)
    features["num_contains"] = int(n1 in n2 or n2 in n1)

    overlap = len(set(n1) & set(n2))
    total = max(len(set(n1) | set(n2)), 1)
    features["num_overlap_ratio"] = overlap / total

    features["num_same_length"] = int(len(n1) == len(n2))

    common_prefix = 0
    for a, b in zip(n1, n2):
        if a == b:
            common_prefix += 1
        else:
            break

    features["num_prefix_ratio"] = common_prefix / max(len(n1), len(n2), 1)

    return features


def domain_type(domain):
    if domain in PRIVACY_DOMAINS:
        return 2
    if domain in PUBLIC_DOMAINS:
        return 1
    return 0


def create_features(row):
    email1 = row["inputEmail"]
    email2 = row["matchEmail"]

    user1, domain1 = normalize_email(email1)
    user2, domain2 = normalize_email(email2)

    alpha1, nums1 = split_alpha_num(user1)
    alpha2, nums2 = split_alpha_num(user2)

    sorted_alpha1 = sort_alpha(alpha1)
    sorted_alpha2 = sort_alpha(alpha2)

    matchpoint_raw = str(row.get("Matchpoint", "") or "")
    matchpoint = normalize_matchpoint(matchpoint_raw)

    try:
        tfidf_matrix = vectorizer.fit_transform([user1, user2])
        cosine = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        cosine = 0.0

    num_feats = number_features(nums1, nums2)

    matchpoint_similarity = (
        fuzz.ratio(user1, matchpoint) if matchpoint else 0
    )

    features = {
        "fuzz_ratio": fuzz.ratio(user1, user2),
        "partial_ratio": fuzz.partial_ratio(user1, user2),
        "token_sort": fuzz.token_sort_ratio(alpha1, alpha2),
        "soundex_same": int(jellyfish.soundex(alpha1) == jellyfish.soundex(alpha2)),
        "metaphone_same": int(
            jellyfish.metaphone(alpha1) == jellyfish.metaphone(alpha2)
        ),
        "length_diff": abs(len(user1) - len(user2)),
        "same_domain": int(domain1 == domain2),
        "same_alpha": int(alpha1 == alpha2),
        "domain_type_1": domain_type(domain1),
        "domain_type_2": domain_type(domain2),
        "char_ngram_cosine": cosine,
        "contains_relation": int(alpha1 in alpha2 or alpha2 in alpha1),
        "matchpoint_similarity": matchpoint_similarity,
        "jaro_winkler_alpha": jellyfish.jaro_winkler_similarity(alpha1, alpha2),
        "jaro_winkler_sorted": jellyfish.jaro_winkler_similarity(
            sorted_alpha1, sorted_alpha2
        ),
        "char_overlap_ratio": char_overlap_ratio(alpha1, alpha2),
        "sorted_fuzz_ratio": fuzz.ratio(sorted_alpha1, sorted_alpha2),
    }

    features.update(num_feats)

    return pd.Series(features)
