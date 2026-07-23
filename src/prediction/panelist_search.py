import jellyfish
import pandas as pd
from rapidfuzz import fuzz, process

from src.prediction.prediction import predict_scores_batch
from src.preprocessing.preprocessing import normalize_email, split_alpha_num
from src.utils.config import (
    MIN_ALPHA_LENGTH_FOR_SERIES,
    MIN_MATCHPOINT_LENGTH,
    MIN_NUM_LENGTH,
    MIN_NUM_PREFIX_LENGTH,
    MIN_USERNAME_PARTIAL_RATIO,
    PANELIST_EMAIL_COLUMN,
    PANELIST_PATH,
    PREFILTER_FUZZY_CUTOFF,
    PRIVACY_DOMAINS,
    PUBLIC_DOMAINS,
    SIMILARITY_THRESHOLD,
)
from src.utils.matchpoint import _longest_common_substring, derive_matchpoint
from src.utils.pattern_quality import (
    filter_results_by_pattern_quality,
    is_trivial_matchpoint,
)

_panelists_cache: pd.DataFrame | None = None


def _load_panelists() -> pd.DataFrame:
    global _panelists_cache
    if _panelists_cache is None:
        df = pd.read_csv(PANELIST_PATH, usecols=["panelistId", PANELIST_EMAIL_COLUMN])
        df = df.dropna(subset=[PANELIST_EMAIL_COLUMN])
        df[PANELIST_EMAIL_COLUMN] = (
            df[PANELIST_EMAIL_COLUMN].astype(str).str.strip().str.lower()
        )
        df = df[df[PANELIST_EMAIL_COLUMN].str.contains("@", na=False)]
        df = df.drop_duplicates(subset=[PANELIST_EMAIL_COLUMN])

        norm = df[PANELIST_EMAIL_COLUMN].apply(normalize_email)
        df["norm_user"] = norm.apply(lambda x: x[0])
        df["norm_domain"] = norm.apply(lambda x: x[1])
        df["norm_alpha"] = df["norm_user"].apply(lambda u: split_alpha_num(u)[0])
        df["norm_nums"] = df["norm_user"].apply(lambda u: split_alpha_num(u)[1])
        df["norm_alpha_sorted"] = df["norm_alpha"].apply(
            lambda a: "".join(sorted(a))
        )

        _panelists_cache = df.reset_index(drop=True)
    return _panelists_cache


def _is_privacy_domain(domain: str) -> bool:
    return domain in PRIVACY_DOMAINS


def _domains_must_match(domain1: str, domain2: str) -> bool:
    """Privacy domains stay same-provider; public domains may cross (outlook → gmail)."""
    if _is_privacy_domain(domain1) or _is_privacy_domain(domain2):
        return True
    return False


def _nums_share_prefix(nums1: str, nums2: str, min_prefix: int) -> bool:
    if not nums1 or not nums2:
        return False
    prefix_len = 0
    for a, b in zip(nums1, nums2):
        if a == b:
            prefix_len += 1
        else:
            break
    return prefix_len >= min_prefix


def _prefilter_candidates(input_email: str, panelists: pd.DataFrame) -> list[str]:
    input_email = input_email.strip().lower()
    user1, domain1 = normalize_email(input_email)
    alpha1, nums1 = split_alpha_num(user1)
    sorted_alpha1 = "".join(sorted(alpha1))

    emails = panelists[PANELIST_EMAIL_COLUMN].tolist()
    candidate_indices: set[int] = set()

    has_number_anchor = len(nums1) >= MIN_NUM_LENGTH

    if has_number_anchor and domain1 in PRIVACY_DOMAINS:
        mask = (panelists["norm_nums"] == nums1) & (
            panelists["norm_domain"] == domain1
        )
        for idx in panelists.index[mask]:
            candidate_indices.add(int(idx))
    else:
        if alpha1:
            same_alpha_mask = panelists["norm_alpha"] == alpha1
            for idx in panelists.index[same_alpha_mask]:
                candidate_indices.add(int(idx))

        alphas = panelists["norm_alpha"].tolist()
        alpha_matches = process.extract(
            alpha1,
            alphas,
            scorer=fuzz.partial_ratio,
            score_cutoff=PREFILTER_FUZZY_CUTOFF,
            limit=None,
        )
        for _, _, idx in alpha_matches:
            candidate_indices.add(idx)

        if sorted_alpha1:
            sorted_alphas = panelists["norm_alpha_sorted"].tolist()
            sorted_matches = process.extract(
                sorted_alpha1,
                sorted_alphas,
                scorer=fuzz.ratio,
                score_cutoff=85,
                limit=None,
            )
            for _, _, idx in sorted_matches:
                candidate_indices.add(idx)

        if sorted_alpha1 and len(alpha1) >= 5:
            for idx, sorted_a in enumerate(panelists["norm_alpha_sorted"]):
                if sorted_a and jellyfish.jaro_winkler_similarity(
                    sorted_alpha1, sorted_a
                ) >= 0.85:
                    candidate_indices.add(idx)

        if alpha1:
            set1 = set(alpha1)
            for idx, a in enumerate(panelists["norm_alpha"]):
                if not a:
                    continue
                set2 = set(a)
                total = max(len(set1 | set2), 1)
                if len(set1 & set2) / total >= 0.70:
                    candidate_indices.add(idx)

        if user1:
            users = panelists["norm_user"].tolist()
            user_matches = process.extract(
                user1,
                users,
                scorer=fuzz.partial_ratio,
                score_cutoff=PREFILTER_FUZZY_CUTOFF,
                limit=None,
            )
            for _, _, idx in user_matches:
                candidate_indices.add(idx)

    candidates = []
    for idx in candidate_indices:
        email = emails[idx]
        if email != input_email:
            candidates.append(email)

    return candidates


def _is_strong_match(
    input_email: str, candidate_email: str, score: float, threshold: float
) -> bool:
    if score <= threshold:
        return False

    user1, domain1 = normalize_email(input_email)
    user2, domain2 = normalize_email(candidate_email)
    alpha1, nums1 = split_alpha_num(user1)
    alpha2, nums2 = split_alpha_num(user2)

    if _domains_must_match(domain1, domain2) and domain1 != domain2:
        return False

    username_sim = fuzz.partial_ratio(user1, user2)
    alpha_sim = fuzz.partial_ratio(alpha1, alpha2) if alpha1 and alpha2 else 0

    both_public = domain1 in PUBLIC_DOMAINS and domain2 in PUBLIC_DOMAINS
    cross_domain = domain1 != domain2

    if cross_domain and not both_public:
        return False

    if cross_domain and both_public:
        if alpha1 and alpha1 == alpha2 and len(alpha1) >= MIN_ALPHA_LENGTH_FOR_SERIES:
            return True
        if username_sim >= 92:
            return True
        if (
            len(nums1) >= MIN_NUM_LENGTH
            and nums1 == nums2
            and alpha_sim >= 85
        ):
            return True
        return False

    if alpha1 and alpha1 == alpha2 and len(alpha1) >= MIN_ALPHA_LENGTH_FOR_SERIES:
        return True

    if username_sim >= 92:
        return True

    if (
        len(nums1) >= MIN_NUM_LENGTH
        and nums1 == nums2
        and domain1 in PRIVACY_DOMAINS
    ):
        return True

    if (
        len(nums1) >= MIN_NUM_LENGTH
        and nums1 == nums2
        and alpha_sim >= 85
    ):
        return True

    if (
        alpha1
        and alpha1 == alpha2
        and len(nums1) >= MIN_NUM_LENGTH
        and len(nums2) >= MIN_NUM_LENGTH
        and _nums_share_prefix(nums1, nums2, MIN_NUM_PREFIX_LENGTH)
    ):
        return True

    if username_sim >= MIN_USERNAME_PARTIAL_RATIO:
        matchpoint = derive_matchpoint(input_email, candidate_email)
        if "@" in matchpoint:
            prefix_len = 0
            for c1, c2 in zip(alpha1, alpha2):
                if c1 == c2:
                    prefix_len += 1
                else:
                    break
            suffix_len = 0
            for c1, c2 in zip(reversed(alpha1), reversed(alpha2)):
                if c1 == c2:
                    suffix_len += 1
                else:
                    break
            lcs_len = len(_longest_common_substring(alpha1, alpha2))
            if not (prefix_len >= 2 or suffix_len >= 2 or lcs_len >= 3):
                return False
        return len(matchpoint) >= MIN_MATCHPOINT_LENGTH

    sorted_alpha1 = "".join(sorted(alpha1))
    sorted_alpha2 = "".join(sorted(alpha2))
    if (
        sorted_alpha1 == sorted_alpha2
        and len(alpha1) >= MIN_ALPHA_LENGTH_FOR_SERIES
        and (domain1 == domain2 or both_public)
    ):
        return True

    return False


def _valid_matchpoint(matchpoint: str) -> bool:
    return not is_trivial_matchpoint(matchpoint)


def find_similar_panelists(
    input_email: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    input_email = input_email.strip().lower()
    panelists = _load_panelists()

    print(f"Loaded {len(panelists):,} panelist emails")
    candidates = _prefilter_candidates(input_email, panelists)
    print(f"Pre-filtered to {len(candidates):,} candidates")

    if not candidates:
        return pd.DataFrame(columns=["email", "score", "matchpoint"])

    print("Scoring with ML model...")
    scored = predict_scores_batch(input_email, candidates)

    email_to_id = dict(zip(panelists[PANELIST_EMAIL_COLUMN], panelists["panelistId"]))

    results = []
    for email, score in scored:
        if not _is_strong_match(input_email, email, score, threshold):
            continue

        matchpoint = derive_matchpoint(input_email, email)
        if not _valid_matchpoint(matchpoint):
            continue

        results.append(
            {
                "panelistid": email_to_id.get(email, ""),
                "email": email,
                "score": score,
                "matchpoint": matchpoint,
            }
        )

    results = filter_results_by_pattern_quality(results, input_email, panelists)

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["matchpoint"] = df["matchpoint"].apply(lambda x: x.split("@")[0] if isinstance(x, str) and "@" in x else x)

    print(f"Found {len(df):,} matches after pattern-quality filter")
    return df


if __name__ == "__main__":
    email = input("Enter email to search: ").strip()
    matches = find_similar_panelists(email)
    if matches.empty:
        print("No similar emails found.")
    else:
        print(matches.to_string(index=False))
