"""Fine-tune a Hugging Face encoder for multi-label Five-Domains classification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_LABELS = ["nutrition", "physical_environment", "health", "behavioural_interactions"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=Path("artifacts/modernbert"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "VetWelfare-NLP"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    extension = "parquet" if args.train.suffix == ".parquet" else "csv"
    dataset = load_dataset(extension, data_files={"train": str(args.train), "validation": str(args.validation)})
    required = {args.text_column, *args.labels}
    for split, ds in dataset.items():
        missing = sorted(required - set(ds.column_names))
        if missing:
            raise ValueError(f"{split} is missing {missing}; columns={ds.column_names}")

    if args.dry_run:
        print(json.dumps({
            "model": args.model,
            "splits": {name: len(ds) for name, ds in dataset.items()},
            "text_column": args.text_column,
            "labels": args.labels,
        }, indent=2))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    id2label = {i: label for i, label in enumerate(args.labels)}
    label2id = {label: i for i, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(args.labels),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    def tokenize(batch: dict[str, list]) -> dict[str, object]:
        encoded = tokenizer(batch[args.text_column], truncation=True, max_length=args.max_length)
        encoded["labels"] = np.asarray(
            [[float(batch[label][i]) for label in args.labels] for i in range(len(batch[args.text_column]))],
            dtype=np.float32,
        )
        return encoded

    remove_columns = dataset["train"].column_names
    encoded = dataset.map(tokenize, batched=True, remove_columns=remove_columns)

    def metrics(eval_prediction) -> dict[str, float]:
        logits, labels = eval_prediction
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        predictions = (probabilities >= args.threshold).astype(int)
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )
        return {
            "macro_f1": float(macro_f1),
            "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
        }

    report_to = ["wandb"] if os.getenv("WANDB_API_KEY") else []
    training_args = TrainingArguments(
        output_dir=str(args.output),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=report_to,
        run_name=f"{Path(args.model).name}-five-domains",
        seed=args.seed,
        fp16=False,
        bf16=True,
    )
    if report_to:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    final_metrics = trainer.evaluate()
    trainer.save_model(str(args.output / "best_model"))
    tokenizer.save_pretrained(str(args.output / "best_model"))
    (args.output / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
