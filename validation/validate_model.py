from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "News_dataset"
VALIDATION_DIR = PROJECT_ROOT / "validation"

MODEL_PATH = PROJECT_ROOT / "fake_news_detection_svm.pkl"
VALIDATION_SAMPLES_PATH = VALIDATION_DIR / "app_validation_samples.csv"

FAKE_DATA_PATH = DATA_DIR / "Fake.csv"
TRUE_DATA_PATH = DATA_DIR / "True.csv"

import re
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD MODEL AND DATASET
# ============================================================

print("Loading model and datasets...")

model = joblib.load(MODEL_PATH)

fake_df = pd.read_csv(FAKE_DATA_PATH)
true_df = pd.read_csv(TRUE_DATA_PATH)

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
)

print("Original dataset:", len(df))


# ============================================================
# 2. CLEAN DATA
# ============================================================

df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)
df["date"] = df["date"].fillna("").astype(str)

# Remove empty article bodies
df = df[df["text"].str.strip() != ""].copy()

# Remove article texts associated with conflicting labels
conflicting_texts = (
    df.groupby("text")["label"].nunique()
)

conflicting_texts = conflicting_texts[
    conflicting_texts > 1
].index

df = df[~df["text"].isin(conflicting_texts)].copy()

# Remove repeated article bodies
df = df.drop_duplicates(
    subset=["text"],
    keep="first"
).copy()

print("Cleaned dataset:", len(df))


# ============================================================
# 3. PARSE DATES — CORRECTED
# ============================================================

# Remove leading and trailing spaces
df["date"] = (
    df["date"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Parse long date format: December 31, 2017
date_long = pd.to_datetime(
    df["date"],
    format="%B %d, %Y",
    errors="coerce",
    utc=True
)

# Parse abbreviated month format: Dec 31, 2017
date_short = pd.to_datetime(
    df["date"],
    format="%b %d, %Y",
    errors="coerce",
    utc=True
)

# Parse format: 19-Feb-18
date_day_month = pd.to_datetime(
    df["date"],
    format="%d-%b-%y",
    errors="coerce",
    utc=True
)

# Combine successfully parsed dates
df["parsed_date"] = (
    date_long
    .fillna(date_short)
    .fillna(date_day_month)
)

df["year"] = df["parsed_date"].dt.year

# Chronological split
train_df = df[df["year"] < 2017].copy()
test_df = df[df["year"] == 2017].copy()

print("\nTraining dataset:", len(train_df))
print("Testing dataset:", len(test_df))

print("\n2017 test distribution:")
print(test_df["label"].value_counts().sort_index())

print("\nUnparsed dates:", df["parsed_date"].isna().sum())


# ============================================================
# 4. MODEL PREPROCESSING
# ============================================================

political_terms = [
    "trump", "obama", "hillary", "clinton", "republican",
    "democrat", "gop", "president", "senate", "congress"
]


def remove_text_artifacts(text):

    text = str(text)

    text = re.sub(
        r"\bReuters\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\b(read more|featured image|image via)\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_political_terms(text):

    text = str(text)

    for term in political_terms:

        text = re.sub(
            rf"\b{term}\b",
            " ",
            text,
            flags=re.IGNORECASE
        )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_article(title, article):

    combined = str(title) + " " + str(article)

    combined = remove_text_artifacts(combined)

    combined = remove_political_terms(combined)

    return combined


# ============================================================
# 5. GENERATE PREDICTIONS IN BATCHES
# ============================================================

print("\nPreparing test articles...")

X_test = [
    prepare_article(title, article)
    for title, article in zip(
        test_df["title"],
        test_df["text"]
    )
]

y_test = test_df["label"].to_numpy()

print("Generating predictions...")

predictions = []

batch_size = 250

for start in range(0, len(X_test), batch_size):

    batch = X_test[start:start + batch_size]

    batch_predictions = model.predict(batch)

    predictions.extend(batch_predictions)

    print(
        f"Processed {min(start + batch_size, len(X_test))}"
        f" / {len(X_test)} articles"
    )


# ============================================================
# 6. EVALUATION
# ============================================================

predictions = pd.Series(predictions).to_numpy()
# ============================================================
# 6. EXPORT SAMPLES FOR STREAMLIT APP VALIDATION
# ============================================================

sample_df = test_df[
    ["title", "text", "label"]
].copy().reset_index(drop=True)

sample_df["predicted_label"] = predictions

# Convert numerical labels to readable names
sample_df["actual_class"] = sample_df["label"].map({
    0: "Fake News",
    1: "Real News"
})

sample_df["expected_prediction"] = sample_df[
    "predicted_label"
].map({
    0: "Fake News",
    1: "Real News"
})

sample_df["correct"] = (
    sample_df["label"] == sample_df["predicted_label"]
)

selected_samples = []

# One correct and one incorrect example from each class
for label in [0, 1]:

    class_data = sample_df[sample_df["label"] == label]

    correct = class_data[class_data["correct"]]
    incorrect = class_data[~class_data["correct"]]

    selected_samples.append(
        correct.sample(n=1, random_state=42)
    )

    selected_samples.append(
        incorrect.sample(n=1, random_state=42)
    )

validation_samples = pd.concat(selected_samples)

validation_samples.to_csv(
    "validation/app_validation_samples.csv",
    index=False
)

print("\nFour Streamlit validation samples exported.")
accuracy = accuracy_score(y_test, predictions)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

print("\n" + "=" * 50)
print("MODEL VALIDATION RESULTS")
print("=" * 50)

print(f"Test articles: {len(y_test):,}")

print(f"Accuracy: {accuracy * 100:.4f}%")

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy * 100:.4f}%"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)

print("\nExpected project results:")
print("Accuracy: approximately 96.2416%")
print("Balanced Accuracy: approximately 97.1143%")
print("Confusion Matrix:")
print("[[6545, 56],")
print(" [812, 15682]]")

if len(test_df) != 23095:
    print(
        "\nWARNING: Test dataset size differs from the "
        "expected 23,095. Check dataset cleaning."
    )

print("\nValidation completed.")
