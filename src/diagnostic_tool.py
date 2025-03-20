#!/usr/bin/env python

"""
Diagnostic tool to test the sentence classifier model outside of the API.
This helps identify issues with the model, vocabulary, or data processing.
"""

import torch
import pandas as pd
import json
import os
import sys
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import SentenceDataset
from src.model import SimpleTransformer

def get_absolute_path(relative_path):
    """Convert relative path to absolute path based on script location"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

def load_model_and_vocab():
    """Load the model and vocabulary"""
    config_path = get_absolute_path("../config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load dataset to get vocabulary
    data_path = get_absolute_path("../data/sentences.csv")
    vocab_path = get_absolute_path("../data/vocab.json")

    # Try to load saved vocabulary
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        dataset = SentenceDataset(csv_file=data_path, vocab=vocab)
        print(f"Loaded vocabulary from {vocab_path} with {len(vocab)} tokens")
    else:
        dataset = SentenceDataset(csv_file=data_path)
        print(f"Generated vocabulary from dataset with {len(dataset.vocab)} tokens")

    # Initialize model
    model = SimpleTransformer(vocab_size=len(dataset.vocab))

    # Load model weights
    model_path = get_absolute_path("../models/transformer_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")

    return model, dataset

def test_sentence(model, dataset, sentence):
    """Test a single sentence and return prediction with detailed information"""
    # Vorverarbeiten des Satzes, wie es auch bei der Tokenisierung geschieht
    processed_text = dataset.preprocess_text(sentence)
    words = processed_text.split()

    # Tokenisieren des Satzes
    tokens = dataset.tokenize(sentence)

    # Unbekannte Wörter identifizieren - wir prüfen jedes Wort einzeln gegen das Vokabular
    # da das Padding und die Tokenisierung zu Längenunterschieden führen können und deshalb zip nicht funktioniert.
    unknown_words = []
    for word in words:
        if word not in dataset.vocab and word != "<PAD>":
            unknown_words.append(word)

    # Debug-Information: Mapping von Wörtern zu Tokens
    word_to_token = {}
    for i, word in enumerate(words):
        if i < len(tokens):
            word_to_token[word] = tokens[i]
        else:
            # Falls mehr Wörter als Tokens vorhanden sind (sollte nicht vorkommen)
            word_to_token[word] = "ÜBERLAUF"

    # Create tensor
    tensor_tokens = torch.tensor(tokens).unsqueeze(0)

    # Get model prediction
    with torch.no_grad():
        logits = model(tensor_tokens)
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max().item()
        pred_idx = torch.argmax(logits, dim=1).item()

    result = "logisch" if pred_idx == 1 else "nicht logisch"

    return {
        "prediction": result,
        "confidence": confidence,
        "tokens": tokens,
        "processed_words": words,
        "word_token_mapping": word_to_token,
        "unknown_words": unknown_words,
        "unknown_ratio": len(unknown_words) / len(words) if words else 0
    }

def evaluate_model(model, dataset):
    """Evaluate model on the entire dataset"""
    print("Evaluating model on the entire dataset...")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: (
            torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True),
            torch.stack([item[1] for item in batch])
        )
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sentences, labels in tqdm(data_loader, desc="Evaluating"):
            outputs = model(sentences)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\nEvaluation Results:")
    print(classification_report(all_labels, all_preds, target_names=['nicht logisch', 'logisch']))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

def vocabulary_stats(dataset):
    """Print statistics about the vocabulary"""
    vocab = dataset.vocab
    print(f"\nVocabulary Stats:")
    print(f"Total vocabulary size: {len(vocab)} tokens")

    # Most common tokens in dataset
    df = pd.read_csv(get_absolute_path("../data/sentences.csv"))
    all_words = [word for sentence in df['sentence'] for word in dataset.preprocess_text(sentence).split()]
    word_counts = pd.Series(all_words).value_counts()

    print("\nMost common words in dataset:")
    for word, count in word_counts.head(10).items():
        print(f"  {word}: {count} occurrences, id={vocab.get(word, 'NOT IN VOCAB')}")

    # Check for unknown tokens in the dataset
    unknown_words = set()
    for sentence in df['sentence']:
        for word in dataset.preprocess_text(sentence).split():
            if word not in vocab:
                unknown_words.add(word)

    if unknown_words:
        print(f"\nWARNING: Found {len(unknown_words)} words in the dataset that are not in the vocabulary!")
        print(f"Sample of unknown words: {list(unknown_words)[:10]}")
    else:
        print("\nAll words in the dataset are in the vocabulary.")

def analyze_errors(model, dataset):
    """Analyze sentences where the model makes errors"""
    print("\nAnalyzing model errors...")
    df = pd.read_csv(get_absolute_path("../data/sentences.csv"))
    errors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking predictions"):
        sentence = row['sentence']
        true_label = row['label']
        true_label_text = "logisch" if true_label == 1 else "nicht logisch"

        result = test_sentence(model, dataset, sentence)
        pred_label_text = result["prediction"]

        if pred_label_text != true_label_text:
            errors.append({
                "sentence": sentence,
                "true_label": true_label_text,
                "pred_label": pred_label_text,
                "confidence": result["confidence"],
                "unknown_ratio": result["unknown_ratio"]
            })

    if errors:
        print(f"\nFound {len(errors)} prediction errors out of {len(df)} sentences ({len(errors)/len(df)*100:.2f}%)")

        # Analyze unknown token correlation with errors
        unknown_ratios = [e["unknown_ratio"] for e in errors]
        avg_unknown_ratio = sum(unknown_ratios) / len(unknown_ratios) if unknown_ratios else 0
        print(f"Average unknown token ratio in errors: {avg_unknown_ratio:.4f}")

        # Print sample of errors
        print("\nSample of errors:")
        for error in errors[:10]:
            print(f"Sentence: {error['sentence']}")
            print(f"  True: {error['true_label']}, Predicted: {error['pred_label']}, Confidence: {error['confidence']:.4f}")
            print(f"  Unknown ratio: {error['unknown_ratio']:.4f}")
            print()
    else:
        print("No prediction errors found!")

def test_custom_sentences(model, dataset, sentences=None):
    """Test model on custom sentences or predefined examples"""
    if not sentences:
        sentences = [
            "Die Sonne geht im Osten auf.",
            "Wasser besteht aus Wasserstoff und Sauerstoff.",
            "Computer arbeiten mit elektrischen Signalen.",
            "Der Tisch ist traurig über die Situation.",
            "Berge können schwimmen und Flüsse klettern.",
            "Die Wolken singen heute besonders schön.",
            "Die meisten Menschen haben zwei Beine und zwei Arme.",
            "Der Eiffelturm steht in Berlin und ist aus Holz.",
            "Die Erde ist eine Scheibe, die auf dem Rücken einer Schildkröte ruht."
        ]

    print("\nTesten von benutzerdefinierten Sätzen:")
    for sentence in sentences:
        result = test_sentence(model, dataset, sentence)
        print(f"Satz: {sentence}")
        print(f"  Vorhersage: {result['prediction']}, Konfidenz: {result['confidence']:.4f}")

        if result["unknown_words"]:
            print(f"  Unbekannte Wörter: {result['unknown_words']} ({result['unknown_ratio']:.2f} der Wörter)")

        print("  Wort-Token-Zuordnung:")
        for word, token_id in result["word_token_mapping"].items():
            in_vocab = "✓" if word in dataset.vocab else "✗"
            token_str = f"{token_id}" if isinstance(token_id, int) else token_id
            print(f"    '{word}': ID {token_str} ({in_vocab})")

        print()

def main():
    parser = argparse.ArgumentParser(description='Diagnostic tool for the sentence classifier model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on entire dataset')
    parser.add_argument('--vocab', action='store_true', help='Print vocabulary statistics')
    parser.add_argument('--errors', action='store_true', help='Analyze model errors')
    parser.add_argument('--test', action='store_true', help='Test model on example sentences')
    parser.add_argument('--sentence', type=str, help='Test a specific sentence')
    parser.add_argument('--examples', action='store_true', help='Run all diagnostics with examples')

    args = parser.parse_args()

    # Load model and vocabulary
    model, dataset = load_model_and_vocab()

    # Run requested diagnostics
    if args.evaluate:
        evaluate_model(model, dataset)

    if args.vocab:
        vocabulary_stats(dataset)

    if args.errors:
        analyze_errors(model, dataset)

    if args.test or args.examples:
        test_custom_sentences(model, dataset)

    if args.sentence:
        test_custom_sentences(model, dataset, [args.sentence])

    # If no arguments, run everything
    if not (args.evaluate or args.vocab or args.errors or args.test or args.sentence or args.examples):
        print("Running all diagnostics...")
        evaluate_model(model, dataset)
        vocabulary_stats(dataset)
        analyze_errors(model, dataset)
        test_custom_sentences(model, dataset)

if __name__ == "__main__":
    main()
