import ast
import pandas as pd
import os
import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import torch
import itertools
from collections import defaultdict


TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
BATCH_SIZE = 8


def row_to_text(row):
    """
    Convert a row of the dataframe to a text string that can be fed into BERT.
     - Convert the 'items' column (which is a list of "shorts" = alert descriptors) to a string of item tokens
    - Prefixed by "TOK_"
    """
    items = (
        ast.literal_eval(row["items"])
        if isinstance(row["items"], str)
        else row["items"]
    )
    items = sorted(list(items))
    item_tokens = [f"TOK_{x}" for x in items]

    # extra = [
    #     f"TIME_{row['time_of_day']}",
    #     f"NALERTS_{row['n_alerts']}",
    # ]

    return " ".join(item_tokens)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load a Huggingface tokenizer for the given model name.
    """
    return AutoTokenizer.from_pretrained(model_name)


def make_huggingface_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert a pandas dataframe to format expected by Huggingface pipeline.
    Wrapper around Dataset.from_pandas that selects only the relevant columns and renames them to "text" and "label".
    """
    return Dataset.from_pandas(df[["text", "label"]])


def tokenize_dataset(
    dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 64
) -> Dataset:
    """
    Tokenize the text column of the dataset using the provided tokenizer.
    Truncate and pad the sequences to the specified max_length.
    """

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            # padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True)

    cols_to_remove = ["text"]
    if "__index_level_0__" in dataset.column_names:
        cols_to_remove.append("__index_level_0__")

    dataset = dataset.remove_columns(cols_to_remove)
    dataset.set_format("torch")
    return dataset


def prepare_datasets(
    train_df,
    val_df,
    test_df,
    tokenizer: AutoTokenizer,
    max_length: int = 64,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Prepare the train, validation, and test datasets for BERT.
    Converts the dataframes to Huggingface Datasets and tokenizes the text.
    """

    train_ds = tokenize_dataset(
        make_huggingface_dataset(train_df), tokenizer, max_length
    )
    val_ds = tokenize_dataset(make_huggingface_dataset(val_df), tokenizer, max_length)
    test_ds = tokenize_dataset(make_huggingface_dataset(test_df), tokenizer, max_length)
    return train_ds, val_ds, test_ds


def load_model(model_name: str, num_labels: int = 2):
    """
    Load a Huggingface model for sequence classification with the specified number of labels.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None = None,  # optional
) -> dict[str, float]:
    """
    Compute classification metrics + diagnostics.

    - y_probs: optional predicted probabilities for class 1 (attack)
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Predicted attack stats
    n_pred_attack = int((y_pred == 1).sum())
    attack_rate = n_pred_attack / len(y_pred)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pred_attack_count": n_pred_attack,
        "pred_attack_rate": attack_rate,
    }

    # probability diagnostics
    if y_probs is not None:
        metrics.update(
            {
                "avg_attack_prob": float(np.mean(y_probs)),
                "max_attack_prob": float(np.max(y_probs)),
                "min_attack_prob": float(np.min(y_probs)),
            }
        )

    return metrics


def compute_metrics(eval_pred) -> dict[str, float]:
    """
    Wrapper for Hugging Face Trainer.
    Expects (logits, labels).
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return compute_classification_metrics(labels, preds)


def make_training_args(output_dir: str) -> TrainingArguments:
    """Create a TrainingArguments object with the specified output directory and other hyperparameters."""
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )


def make_trainer(
    model,
    tokenizer: AutoTokenizer,
    training_args: TrainingArguments,
    train_ds: Dataset,
    val_ds: Dataset,
) -> Trainer:
    """
    Create a Huggingface Trainer object for training the BERT model.
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


def train_and_evaluate(trainer: Trainer, test_ds: Dataset) -> dict[str, float]:
    """
    Train the model using the provided Trainer and evaluate on the test dataset, returning the evaluation metrics.
    """
    trainer.train()

    pred = trainer.predict(test_ds)
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(axis=1)
    y_probs = torch.softmax(torch.tensor(pred.predictions), dim=1)[:, 1].numpy()

    metrics = compute_classification_metrics(y_true, y_pred, y_probs)
    return metrics


def run_baseline(dataset_name: str, scenario: str, model_name: str):
    """
    Run the BERT baseline for the given dataset, scenario, and model name.
    """
    root = Path.cwd().parent.parent
    sys.path.insert(0, str(root / "thesis"))

    input_path = (
        root.parent / "out" / "eda" / dataset_name / scenario
    )  # contains the processed transactions from the original dataset
    out_path = root.parent / "out" / "bert" / dataset_name / scenario

    os.makedirs(out_path, exist_ok=True)

    print("Running BERT baseline for scenario:", scenario)

    # Load the processed transactions CSV
    df = pd.read_csv(
        os.path.join(input_path, f"{scenario}_transactions.csv"), low_memory=False
    )
    df = df.dropna(subset=["items", "tx_label"]).copy()

    # Convert the 'items' column to text format for BERT
    df["text"] = df.apply(row_to_text, axis=1)

    # Encode labels
    df["label"] = df["tx_label"].map({"benign": 0, "attack": 1})

    # Split data
    print(
        "Splitting data into train/val/test with ratios:",
        1 - TEST_SPLIT,
        VAL_SPLIT,
        TEST_SPLIT,
    )
    train_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, stratify=df["label"], random_state=42
    )

    val_size = VAL_SPLIT / (1 - TEST_SPLIT)

    train_df, val_df = train_test_split(
        train_df, test_size=val_size, stratify=train_df["label"], random_state=42
    )

    print(
        f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}"
    )

    print("Label distribution in each split:")
    print("Train:", train_df["label"].value_counts(normalize=True))
    print("Val:", val_df["label"].value_counts(normalize=True))
    print("Test:", test_df["label"].value_counts(normalize=True))

    # Load tokenizer
    print("Loading tokenizer for model:", model_name)
    tokenizer = load_tokenizer(model_name)

    # Prepare datasets
    train_ds, val_ds, test_ds = prepare_datasets(train_df, val_df, test_df, tokenizer)

    # Load model
    print("Loading model:", model_name)
    model = load_model(model_name)

    # Device check
    print("PyTorch version:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())

    if torch.backends.mps.is_available():
        print("Using Apple GPU via MPS")
    else:
        print("MPS not available, training will use CPU")

    # Make training arguments and trainer
    training_args = make_training_args(str(out_path))

    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_ds=train_ds,
        val_ds=val_ds,
    )

    # Train + eval text classifier
    print("Train and evaluate BERT model...")
    results = train_and_evaluate(trainer, test_ds)

    # Save results

    print("Writing results to:", out_path / "results.txt")
    with open(out_path / "results.txt", "w") as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")

    print("Done.")


def predict_attack_prob(model, tokenizer, text: str, max_length: int = 64) -> float:
    """
    Return P(class=1 = attack) for one text sample.
    """
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        attack_prob = probs[0, 1].item()

    return attack_prob


def remove_tokens_once(tokens: list[str], tokens_to_remove: set[str]) -> list[str]:
    """
    Remove all occurrences of the specified tokens.
    """
    return [t for t in tokens if t not in tokens_to_remove]


def score_token_and_pair_interactions(
    model,
    tokenizer,
    text: str,
    top_k: int = 10,
    max_length: int = 64,
) -> dict:
    """
    Score individual tokens and token pairs for one sample using removal-based interaction.

    Returns:
    - base_prob
    - token_scores: list of (token, delta)
    - pair_scores: list of ((token_a, token_b), interaction, delta_ab)
    """
    tokens = text.split()
    unique_tokens = sorted(set(tokens))

    base_prob = predict_attack_prob(model, tokenizer, text, max_length=max_length)

    # Individual token effects
    single_deltas = {}
    for tok in unique_tokens:
        reduced_tokens = remove_tokens_once(tokens, {tok})
        reduced_text = " ".join(reduced_tokens)
        reduced_prob = predict_attack_prob(
            model, tokenizer, reduced_text, max_length=max_length
        )
        single_deltas[tok] = base_prob - reduced_prob

    # Pair effects
    pair_scores = []
    for tok_a, tok_b in itertools.combinations(unique_tokens, 2):
        reduced_tokens = remove_tokens_once(tokens, {tok_a, tok_b})
        reduced_text = " ".join(reduced_tokens)
        reduced_prob = predict_attack_prob(
            model, tokenizer, reduced_text, max_length=max_length
        )

        delta_ab = base_prob - reduced_prob
        interaction = delta_ab - (single_deltas[tok_a] + single_deltas[tok_b])

        pair_scores.append(((tok_a, tok_b), interaction, delta_ab))

    token_scores = sorted(single_deltas.items(), key=lambda x: abs(x[1]), reverse=True)
    pair_scores = sorted(pair_scores, key=lambda x: abs(x[1]), reverse=True)

    return {
        "base_prob": base_prob,
        "token_scores": token_scores[:top_k],
        "pair_scores": pair_scores[:top_k],
    }


def aggregate_pair_scores_over_dataset(
    model,
    tokenizer,
    texts: list[str],
    top_k_per_sample: int = 10,
    max_length: int = 64,
) -> list[tuple[tuple[str, str], float, int]]:
    """
    Aggregate absolute interaction scores across many samples.

    Returns:
    [((tok_a, tok_b), avg_abs_interaction, count), ...]
    """
    pair_sum = defaultdict(float)
    pair_count = defaultdict(int)

    for text in texts:
        result = score_token_and_pair_interactions(
            model=model,
            tokenizer=tokenizer,
            text=text,
            top_k=top_k_per_sample,
            max_length=max_length,
        )

        for pair, interaction, _ in result["pair_scores"]:
            pair_sum[pair] += abs(interaction)
            pair_count[pair] += 1

    aggregated = []
    for pair in pair_sum:
        avg_score = pair_sum[pair] / pair_count[pair]
        aggregated.append((pair, avg_score, pair_count[pair]))

    aggregated.sort(key=lambda x: x[1], reverse=True)
    return aggregated


def run_pair_inspection():
    pass


def main(dataset_name: str, scenario: str, model_name: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT baseline script")
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Name of the folder containing the original dataset (e.g., 'out/eda/alerts_csv/scenario')",
    )
    parser.add_argument(
        "--scenario", required=True, help="Scenario name for output files"
    )
    parser.add_argument(
        "--model_name",
        help="Huggingface BERT model name (e.g., 'distilbert-base-uncased')",
        default="distilbert-base-uncased",
    )
    args = parser.parse_args()

    main(args.dataset_name, args.scenario, args.model_name)
