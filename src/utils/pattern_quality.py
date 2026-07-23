import pandas as pd

from src.preprocessing.preprocessing import normalize_email, split_alpha_num
from src.utils.config import (
    GENERIC_ALPHA_FRAGMENTS,
    GENERIC_NUMS,
    MASS_PATTERN_PANEL_THRESHOLD,
    MIN_ALPHA_LENGTH_FOR_SERIES,
    MIN_MATCHPOINT_LENGTH,
    MIN_NUM_LENGTH,
    RARE_PATTERN_PANEL_MAX,
)


def is_trivial_number(nums: str) -> bool:
    if not nums:
        return True
    if nums in GENERIC_NUMS:
        return True
    if len(nums) < MIN_NUM_LENGTH:
        return True
    if len(set(nums)) == 1:
        return True
    return False


def is_trivial_matchpoint(matchpoint: str) -> bool:
    if not matchpoint:
        return True

    if "@" in matchpoint:
        num_part, domain = matchpoint.split("@", 1)
        return is_trivial_number(num_part) or not domain

    if len(matchpoint) < 4:
        return True

    if matchpoint in GENERIC_ALPHA_FRAGMENTS:
        return True

    if len(matchpoint) < MIN_MATCHPOINT_LENGTH:
        return matchpoint in GENERIC_ALPHA_FRAGMENTS

    return False


def _pattern_panel_count(matchpoint: str, panelists: pd.DataFrame) -> int:
    if "@" in matchpoint:
        num_part, domain = matchpoint.split("@", 1)
        mask = (panelists["norm_nums"] == num_part) & (
            panelists["norm_domain"].str.replace(".com", "", regex=False) == domain
        )
        return int(mask.sum())

    if len(matchpoint) >= 3:
        return int(
            panelists["norm_user"].str.contains(matchpoint, regex=False, na=False).sum()
        )

    return 0


def is_mass_panel_pattern(matchpoint: str, panelists: pd.DataFrame) -> bool:
    if is_trivial_matchpoint(matchpoint):
        return True

    count = _pattern_panel_count(matchpoint, panelists)
    return count > MASS_PATTERN_PANEL_THRESHOLD


def is_rare_distinctive_cluster(matchpoint: str, panelists: pd.DataFrame) -> bool:
    if is_trivial_matchpoint(matchpoint):
        return False

    count = _pattern_panel_count(matchpoint, panelists)
    return 1 <= count <= RARE_PATTERN_PANEL_MAX


def is_strong_individual_match(input_email: str, candidate_email: str) -> bool:
    from rapidfuzz import fuzz

    user1, _ = normalize_email(input_email)
    user2, _ = normalize_email(candidate_email)
    alpha1, nums1 = split_alpha_num(user1)
    alpha2, nums2 = split_alpha_num(user2)

    if alpha1 and alpha1 == alpha2 and len(alpha1) >= MIN_ALPHA_LENGTH_FOR_SERIES:
        return True

    if fuzz.partial_ratio(user1, user2) >= 92:
        return True

    if (
        not is_trivial_number(nums1)
        and nums1 == nums2
        and (alpha1 in alpha2 or alpha2 in alpha1)
        and fuzz.partial_ratio(user1, user2) >= 85
    ):
        return True

    if (
        len(alpha1) >= 4
        and (alpha1 in alpha2 or alpha2 in alpha1)
        and fuzz.partial_ratio(user1, user2) >= 85
    ):
        return True

    return False


def filter_results_by_pattern_quality(
    results: list[dict],
    input_email: str,
    panelists: pd.DataFrame,
) -> list[dict]:
    if not results:
        return []

    kept = []
    for row in results:
        matchpoint = row["matchpoint"]

        if is_trivial_matchpoint(matchpoint):
            continue

        if is_rare_distinctive_cluster(matchpoint, panelists):
            kept.append(row)
            continue

        if is_mass_panel_pattern(matchpoint, panelists):
            if is_strong_individual_match(input_email, row["email"]):
                kept.append(row)
            continue

        if is_strong_individual_match(input_email, row["email"]):
            kept.append(row)

    return kept
