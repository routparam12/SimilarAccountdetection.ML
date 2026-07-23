# Similar Account Detection ML

A machine learning system that detects similar/related email accounts across panelist databases using NLP feature engineering and XGBoost regression.

## Project Structure

```
SimilarAccountdetection.ML/
├── Data/
│   ├── emailmatch.csv          # Training data (email pairs + labels)
│   ├── EmailMatches.csv        # Alternative training data
│   └── panelists.csv           # Panelist database for search
├── models/
│   ├── model.pkl               # Trained XGBoost model
│   └── feature_columns.pkl     # Feature column names
├── src/
│   ├── feature/
│   │   └── feature_builder.py  # 21 NLP features extraction
│   ├── prediction/
│   │   ├── prediction.py       # Batch ML scoring
│   │   └── panelist_search.py  # Search similar panelists
│   ├── preprocessing/
│   │   └── preprocessing.py    # Email normalization
│   ├── training/
│   │   └── train_model.py      # Model training pipeline
│   └── utils/
│       ├── config.py           # Thresholds & domain lists
│       ├── matchpoint.py       # Common pattern extraction
│       └── pattern_quality.py  # False-positive filtering
├── main.py                     # CLI entry point
└── requirements.txt
```

## Model Used

**XGBoost Regressor** (`XGBRegressor`)

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
```

- **Task**: Regression (outputs similarity score 0-100)
- **Training**: 80/20 split, evaluated with MAE (Mean Absolute Error)
- **Features**: 21 handcrafted NLP features per email pair

## How It Works

### 1. Preprocessing
- Lowercases and strips emails
- Removes `+` alias suffixes (Gmail feature)
- Removes `.`, `_`, `-` from usernames
- Splits username into alphabetic and numeric parts

### 2. Feature Engineering (21 Features)

| Feature | Description |
|---------|-------------|
| `fuzz_ratio` | Fuzzy string similarity (0-100) |
| `partial_ratio` | Best partial match score |
| `token_sort` | Order-invariant fuzzy match |
| `soundex_same` | Phonetic encoding match |
| `metaphone_same` | Alternative phonetic match |
| `length_diff` | Absolute length difference |
| `same_domain` | Whether domains match |
| `same_alpha` | Whether alphabetic parts match |
| `domain_type_1/2` | Privacy(2), Public(1), Other(0) |
| `char_ngram_cosine` | TF-IDF character n-gram cosine |
| `contains_relation` | Substring containment |
| `matchpoint_similarity` | Similarity to extracted pattern |
| `jaro_winkler_alpha` | Jaro-Winkler on alphabetic parts |
| `jaro_winkler_sorted` | Jaro-Winkler on sorted characters |
| `char_overlap_ratio` | Unique character overlap (Jaccard) |
| `sorted_fuzz_ratio` | Fuzzy match on sorted characters |
| `num_exact/contains/overlap_ratio/same_length/prefix_ratio` | Numeric part features |

### 3. Panelist Search Pipeline

```
Input Email
    │
    ▼
┌─────────────────────┐
│  Pre-filter (70+)   │  Fuzzy alpha matching, sorted alpha, Jaro-Winkler,
│  from 10k+ panelists │  character overlap, same domain numbers
└─────────┬───────────┘
          │ ~500 candidates
          ▼
┌─────────────────────┐
│  ML Model Scoring   │  XGBoost predicts similarity score
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Strong Match Rules │  Domain logic, username similarity, prefix/suffix
│  (85+ threshold)    │  matching, cross-domain rules for public/privacy
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Pattern Quality    │  Filters trivial matches, mass patterns (>150),
│  Filter             │  validates distinctive clusters
└─────────┬───────────┘
          │
          ▼
      Results
```

### 4. Domain Rules
- **Privacy domains** (ProtonMail, Tutanota): Must stay same-provider
- **Public domains** (Gmail, Outlook, Yahoo): Can cross-match with strict rules
- Cross-domain only allowed when alphabetic part matches exactly (6+ chars) or username similarity ≥ 92

## Terminal Output

### Option 1: Train Model
```
1. Train Model
2. Predict (two emails)
3. Find similar panelists (one email vs panelists.csv)
Select: 1
Generating Features...
Training Model...
MAE: 3.42
Model saved to D:\SimilarAccountdetection.ML\models\model.pkl
```

### Option 2: Predict Two Emails
```
Select: 2
Enter Email 1: john.doe123@gmail.com
Enter Email 2: johndoe123@gmail.com
87.5
```

### Option 3: Search Similar Panelists
```
Select: 3
Enter email to search: john.smith42@proton.me
Loaded 10,542 panelist emails
Pre-filtered to 387 candidates
Scoring with ML model...
Found 12 matches after pattern-quality filter
 panelistid  email                       score  matchpoint
      1042  john.smith42@protonmail.com  94.20  johnsmith42
      2891  john.smith42@tutanota.com    91.80  johnsmith42
       567  jsmith42@protonmail.com      78.30  smith42
      ...
```

## Dependencies

```
pandas
numpy
scikit-learn
rapidfuzz
jellyfish
xgboost
tldextract
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```
