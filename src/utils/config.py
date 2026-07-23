from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"

TRAIN_DATA_PATH = BASE_DIR / "data" / "emailmatch.csv"
PANELIST_PATH = BASE_DIR / "data" / "panelists.csv"
PANELIST_EMAIL_COLUMN = "email"

SIMILARITY_THRESHOLD = 85
PREFILTER_FUZZY_CUTOFF = 70
MIN_USERNAME_PARTIAL_RATIO = 75
MIN_MATCHPOINT_LENGTH = 6
MIN_NUM_LENGTH = 4
MIN_ALPHA_LENGTH_FOR_SERIES = 6
MIN_NUM_PREFIX_LENGTH = 5

# Patterns shared by too many panelists are treated as mass/duplicate-farm signals.
MASS_PATTERN_PANEL_THRESHOLD = 150
RARE_PATTERN_PANEL_MAX = 500

GENERIC_ALPHA_FRAGMENTS = {
    "ira",
    "han",
    "son",
    "ana",
    "ash",
    "ian",
    "ack",
    "jac",
    "nam",
    "shi",
    "mail",
    "com",
}

GENERIC_NUMS = {
    "1",
    "12",
    "123",
    "01",
    "001",
    "0001",
    "00001",
    "000001",
}

PRIVACY_DOMAINS = {
    "proton.me",
    "protonmail.com",
    "tutanota.com",
}

PUBLIC_DOMAINS = {
    "gmail.com",
    "outlook.com",
    "hotmail.com",
    "yahoo.com",
}
