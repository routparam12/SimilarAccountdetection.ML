from src.preprocessing.preprocessing import normalize_email, split_alpha_num
from src.utils.pattern_quality import is_trivial_number


def _longest_common_substring(a: str, b: str) -> str:
    best = ""
    for i in range(len(a)):
        for j in range(i + 1, len(a) + 1):
            part = a[i:j]
            if len(part) > len(best) and part in b:
                best = part
    return best


def extract_matchpoint(email1: str, email2: str, min_length: int = 6) -> str:
    user1, _ = normalize_email(email1)
    user2, _ = normalize_email(email2)
    alpha1, _ = split_alpha_num(user1)
    alpha2, _ = split_alpha_num(user2)

    if not alpha1 or not alpha2:
        return ""

    if alpha1 in alpha2:
        return alpha1
    if alpha2 in alpha1:
        return alpha2

    common = _longest_common_substring(alpha1, alpha2)
    if len(common) >= min_length:
        return common

    return ""


def derive_matchpoint(email1: str, email2: str) -> str:
    """Build Matchpoint for inference when not provided in training row."""
    from src.preprocessing.preprocessing import normalize_matchpoint
    import re
    from os.path import commonprefix

    user1, domain1 = normalize_email(email1)
    user2, domain2 = normalize_email(email2)

    user1_clean = re.sub(r"^\d+_", "", user1)
    user2_clean = re.sub(r"^\d+_", "", user2)

    alpha1, nums1 = split_alpha_num(user1_clean)
    alpha2, nums2 = split_alpha_num(user2_clean)

    text_match = commonprefix([alpha1, alpha2])
    num_match = commonprefix([nums1, nums2])
    prefix_match = text_match + num_match

    if len(prefix_match) >= 6:
        return normalize_matchpoint(prefix_match)

    if alpha1 and alpha1 == alpha2:
        return normalize_matchpoint(alpha1)

    alpha_match = extract_matchpoint(email1, email2)
    if alpha_match:
        return normalize_matchpoint(alpha_match)

    if nums1 and nums1 == nums2 and not is_trivial_number(nums1):
        domain = domain1 if domain1 == domain2 else domain1
        return normalize_matchpoint(f"{nums1}@{domain}")

    if (
        alpha1
        and alpha2
        and sorted(alpha1) == sorted(alpha2)
        and len(alpha1) >= 6
    ):
        return normalize_matchpoint(min(alpha1, alpha2, key=len))

    return ""
