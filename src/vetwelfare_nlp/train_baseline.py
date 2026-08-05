"""Train a multi-label TF-IDF + logistic-regression welfare-domain baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

DEFAULT_LABELS = ["nutrition", "physical_environment", "health", "behavioural_interactions"]


def _load(path: Path, text_column: str, labels: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [column for column in [text_column, *labels] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    return df[text_column].fillna("").astype(str), df[labels].astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x_train, y_train = _load(args.train, args.text_column, args.labels)
    x_val, y_val = _load(args.validation, args.text_column, args.labels)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=args.min_df, max_features=args.max_features,
            sublinear_tf=True, strip_accents="unicode",
        )),
        ("classifier", OneVsRestClassifier(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=args.seed,
        ))),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_val)

    metrics = {
        "micro_f1": float(f1_score(y_val, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_val, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_val, predictions, target_names=args.labels, output_dict=True, zero_division=0
        ),
        "train_rows": len(x_train),
        "validation_rows": len(x_val),
        "labels": args.labels,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": args.labels, "text_column": args.text_column}, args.output / "model.joblib")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"micro_f1": metrics["micro_f1"], "macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
