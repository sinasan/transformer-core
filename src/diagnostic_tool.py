#!/usr/bin/env python

"""
Erweitertes diagnostisches Tool für die Analyse und Fehlerdiagnose des Transformer-Modells.
Dieses Tool bietet umfassende Analysen zur Modellleistung, Vokabularabdeckung und Fehlerfällen.
"""

import torch
import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from collections import Counter, defaultdict

# Pfad für den Import aus src korrigieren
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import SentenceDataset
from src.model import SimpleTransformer

def get_absolute_path(relative_path):
    """Absoluten Pfad basierend auf der Skriptposition berechnen"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

def load_model_and_vocab():
    """Modell und Vokabular laden mit umfassender Fehlerbehandlung"""
    config_path = get_absolute_path("../config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Konfiguration geladen aus {config_path}")
    except Exception as e:
        print(f"Fehler beim Laden der Konfiguration: {e}")
        sys.exit(1)

    # Pfade konfigurieren
    data_path = get_absolute_path("../data/sentences.csv")
    vocab_path = get_absolute_path("../data/vocab.json")
    model_path = get_absolute_path("../models/transformer_model.pth")

    # Prüfen, ob notwendige Dateien existieren
    for path, desc in [(data_path, "Datensatz"), (model_path, "Modell")]:
        if not os.path.exists(path):
            print(f"FEHLER: {desc}-Datei nicht gefunden: {path}")
            sys.exit(1)

    try:
        # Vokabular laden, falls vorhanden
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            dataset = SentenceDataset(csv_file=data_path, vocab=vocab)
            print(f"Vokabular aus {vocab_path} geladen mit {len(vocab)} Tokens")
        else:
            print(f"Kein Vokabular gefunden unter {vocab_path}. Generiere neues Vokabular...")
            dataset = SentenceDataset(csv_file=data_path)
            # Vokabular speichern für zukünftige Verwendung
            with open(vocab_path, 'w') as f:
                json.dump(dataset.vocab, f)
            print(f"Vokabular mit {len(dataset.vocab)} Tokens generiert und gespeichert")

        # Modell initialisieren und laden
        model = SimpleTransformer(vocab_size=len(dataset.vocab))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        print(f"Modell aus {model_path} geladen")

        return model, dataset

    except Exception as e:
        print(f"FEHLER beim Laden von Modell oder Dataset: {e}")
        sys.exit(1)

def test_sentence(model, dataset, sentence, verbose=True, return_details=False):
    """
    Erweiterte Analyse eines einzelnen Satzes mit detaillierten Diagnostikinformationen

    Args:
        model: Das trainierte Modell
        dataset: Das Dataset mit Vokabular
        sentence: Der zu analysierende Satz
        verbose: Ob detaillierte Ausgabe angezeigt werden soll
        return_details: Ob zusätzliche Details zurückgegeben werden sollen

    Returns:
        Dictionary mit Analyseergebnissen
    """
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if start_time:
        start_time.record()

    # Vorverarbeitung des Satzes
    processed_text = dataset.preprocess_text(sentence)
    words = processed_text.split()

    # Tokenisierung
    tokens = dataset.tokenize(sentence)

    # Unbekannte Wörter identifizieren
    unknown_words = []
    for word in words:
        if word not in dataset.vocab and word != "<PAD>":
            unknown_words.append(word)

    # Wort-zu-Token-Mapping erstellen
    word_to_token = {}
    for i, word in enumerate(words):
        if i < len(tokens):
            word_to_token[word] = {
                "token_id": tokens[i],
                "token_name": list(dataset.vocab.keys())[list(dataset.vocab.values()).index(tokens[i])] if tokens[i] in dataset.vocab.values() else "<UNKNOWN>"
            }
        else:
            word_to_token[word] = {"token_id": "OVERFLOW", "token_name": "<OVERFLOW>"}

    # Tensor erstellen und Vorhersage
    tensor_tokens = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        # Forward-Pass durch das Modell
        logits = model(tensor_tokens)

        # Softmax für Wahrscheinlichkeiten
        probs = torch.softmax(logits, dim=1)
        class_probs = probs[0].tolist()  # Beide Klassenwahrscheinlichkeiten
        confidence = probs.max().item()
        pred_idx = torch.argmax(logits, dim=1).item()

    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        inference_time = start_time.elapsed_time(end_time)
    else:
        inference_time = None

    result = "logisch" if pred_idx == 1 else "nicht logisch"

    # Detaillierte Analyseergebnisse
    analysis = {
        "sentence": sentence,
        "processed": processed_text,
        "prediction": result,
        "prediction_index": pred_idx,
        "confidence": confidence,
        "class_probabilities": {
            "nicht logisch": class_probs[0],
            "logisch": class_probs[1]
        },
        "token_ids": tokens,
        "tokens_count": len(tokens),
        "unknown_words": unknown_words,
        "unknown_ratio": len(unknown_words) / len(words) if words else 0,
        "inference_time_ms": inference_time
    }

    if return_details:
        analysis["word_token_mapping"] = word_to_token

    if verbose:
        print(f"\nSatzanalyse: \"{sentence}\"")
        print(f"  Vorverarbeitet: \"{processed_text}\"")
        print(f"  Vorhersage: {result} (Konfidenz: {confidence:.4f})")
        print(f"  Klassenwahrscheinlichkeiten: Nicht logisch: {class_probs[0]:.4f}, Logisch: {class_probs[1]:.4f}")

        if unknown_words:
            print(f"  Unbekannte Wörter: {unknown_words} ({analysis['unknown_ratio']:.2f} aller Wörter)")

        print(f"  Token-IDs: {tokens}")

        if inference_time:
            print(f"  Inferenzzeit: {inference_time:.2f} ms")

    return analysis

def evaluate_model(model, dataset, visualize=False, output_dir=None):
    """
    Umfassende Modellbewertung mit detaillierten Metriken und optionaler Visualisierung

    Args:
        model: Das zu bewertende Modell
        dataset: Das Dataset für die Bewertung
        visualize: Ob Visualisierungen erstellt werden sollen
        output_dir: Verzeichnis für Ausgabedateien (optional)
    """
    print("\n" + "="*80)
    print("Umfassende Modellbewertung")
    print("="*80)

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
    all_probs = []
    all_sentences = dataset.sentences

    # Daten durch das Modell laufen lassen
    print("Evaluiere Modell auf dem gesamten Datensatz...")
    with torch.no_grad():
        for sentences, labels in tqdm(data_loader, desc="Bewertung"):
            outputs = model(sentences)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs[:, 1].tolist())  # Wahrscheinlichkeit für "logisch" (Klasse 1)

    # Grundlegende Metriken
    print("\n1. Klassifikationsbericht:")
    print(classification_report(all_labels, all_preds, target_names=['nicht logisch', 'logisch']))

    # Konfusionsmatrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\n2. Konfusionsmatrix:")
    print(cm)

    # Erweiterte Metriken
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n3. Erweiterte Metriken:")
    print(f"  Genauigkeit (Accuracy): {accuracy:.4f}")
    print(f"  Präzision (Precision): {precision:.4f}")
    print(f"  Sensitivität (Recall): {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Fehlerrate: {1-accuracy:.4f}")

    # Konfidenzverteilung
    confidence_by_class = {
        "korrekt_klassifiziert": [],
        "falsch_klassifiziert": []
    }

    for i, (pred, label, prob) in enumerate(zip(all_preds, all_labels, all_probs)):
        confidence = prob if pred == 1 else (1 - prob)  # Konfidenz für die vorhergesagte Klasse
        if pred == label:
            confidence_by_class["korrekt_klassifiziert"].append(confidence)
        else:
            confidence_by_class["falsch_klassifiziert"].append(confidence)

    print("\n4. Konfidenzanalyse:")
    if confidence_by_class["korrekt_klassifiziert"]:
        print(f"  Durchschnittliche Konfidenz bei korrekten Vorhersagen: {np.mean(confidence_by_class['korrekt_klassifiziert']):.4f}")
        print(f"  Min/Max Konfidenz bei korrekten Vorhersagen: {np.min(confidence_by_class['korrekt_klassifiziert']):.4f} / {np.max(confidence_by_class['korrekt_klassifiziert']):.4f}")

    if confidence_by_class["falsch_klassifiziert"]:
        print(f"  Durchschnittliche Konfidenz bei falschen Vorhersagen: {np.mean(confidence_by_class['falsch_klassifiziert']):.4f}")
        print(f"  Min/Max Konfidenz bei falschen Vorhersagen: {np.min(confidence_by_class['falsch_klassifiziert']):.4f} / {np.max(confidence_by_class['falsch_klassifiziert']):.4f}")
    else:
        print("  Keine falschen Vorhersagen gefunden!")

    # ROC-Kurve und Precision-Recall-Kurve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_curve, precision_curve)

    print(f"\n5. AUC-Metriken:")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Precision-Recall AUC: {pr_auc:.4f}")

    # Visualisierungen
    if visualize:
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Konfusionsmatrix-Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['nicht logisch', 'logisch'],
                    yticklabels=['nicht logisch', 'logisch'])
        plt.ylabel('Tatsächliche Klasse')
        plt.xlabel('Vorhergesagte Klasse')
        plt.title('Konfusionsmatrix')

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            plt.close()
        else:
            plt.show()

        # 2. ROC-Kurve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC-Kurve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Falsch-Positiv-Rate')
        plt.ylabel('Richtig-Positiv-Rate')
        plt.title('ROC-Kurve')
        plt.legend(loc='lower right')

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
            plt.close()
        else:
            plt.show()

        # 3. Precision-Recall-Kurve
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, label=f'PR-Kurve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall-Kurve')
        plt.legend(loc='lower left')

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
            plt.close()
        else:
            plt.show()

        # 4. Konfidenzverteilung
        plt.figure(figsize=(10, 6))
        if confidence_by_class["korrekt_klassifiziert"]:
            sns.histplot(confidence_by_class["korrekt_klassifiziert"], kde=True, label='Korrekte Vorhersagen', color='green', alpha=0.5)
        if confidence_by_class["falsch_klassifiziert"]:
            sns.histplot(confidence_by_class["falsch_klassifiziert"], kde=True, label='Falsche Vorhersagen', color='red', alpha=0.5)
        plt.xlabel('Konfidenz')
        plt.ylabel('Anzahl')
        plt.title('Konfidenzverteilung nach Vorhersageergebnis')
        plt.legend()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
            plt.close()
        else:
            plt.show()

    # Erstellen einer detaillierten Auswertungsdatei
    if output_dir:
        details_df = pd.DataFrame({
            'Satz': all_sentences,
            'Tatsächlich': ['logisch' if l == 1 else 'nicht logisch' for l in all_labels],
            'Vorhersage': ['logisch' if p == 1 else 'nicht logisch' for p in all_preds],
            'Konfidenz': all_probs,
            'Korrekt': [p == l for p, l in zip(all_preds, all_labels)]
        })
        details_df.to_csv(os.path.join(output_dir, 'evaluation_details.csv'), index=False)
        print(f"\nDetailauswertung gespeichert in {os.path.join(output_dir, 'evaluation_details.csv')}")

def vocabulary_stats(dataset, visualize=False, output_dir=None):
    """
    Erweiterte Vokabularstatistiken und -analyse

    Args:
        dataset: Das zu analysierende Dataset
        visualize: Ob Visualisierungen erstellt werden sollen
        output_dir: Verzeichnis für Ausgabedateien (optional)
    """
    print("\n" + "="*80)
    print("Erweiterte Vokabularanalyse")
    print("="*80)

    vocab = dataset.vocab
    total_vocab_size = len(vocab)
    print(f"Gesamtvokabulargröße: {total_vocab_size} Tokens")

    # Häufigkeiten der Tokens im Datensatz analysieren
    df = pd.read_csv(get_absolute_path("../data/sentences.csv"))

    # Alle Wörter aus dem Datensatz extrahieren
    all_words = []
    for sentence in df['sentence']:
        processed = dataset.preprocess_text(sentence)
        all_words.extend(processed.split())

    # Wortzählungen
    word_counts = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_counts)

    print(f"\n1. Grundlegende Statistiken:")
    print(f"  Gesamtzahl an Wörtern im Datensatz: {total_words}")
    print(f"  Anzahl einzigartiger Wörter: {unique_words}")
    print(f"  Vokabularabdeckung: {total_vocab_size / unique_words:.2%}")

    # Satzzeichen im Vokabular
    punct_in_vocab = [p for p in string.punctuation if p in vocab]
    print(f"\n2. Satzzeichen im Vokabular: {len(punct_in_vocab)}/{len(string.punctuation)}")
    if punct_in_vocab:
        print(f"  Vorhandene Satzzeichen: {''.join(punct_in_vocab)}")

    # Häufigste und seltenste Wörter
    most_common = word_counts.most_common(20)
    print("\n3. Häufigste Wörter im Datensatz:")
    for word, count in most_common:
        in_vocab = word in vocab
        in_vocab_str = "✓" if in_vocab else "✗"
        vocab_id = vocab.get(word, "N/A")
        print(f"  {word}: {count} Vorkommen ({count/total_words:.1%}), ID={vocab_id} {in_vocab_str}")

    # Seltenste Wörter (mindestens 1 Vorkommen)
    least_common = word_counts.most_common()[:-21:-1]
    print("\n4. Seltenste Wörter im Datensatz:")
    for word, count in least_common:
        in_vocab = word in vocab
        in_vocab_str = "✓" if in_vocab else "✗"
        vocab_id = vocab.get(word, "N/A")
        print(f"  {word}: {count} Vorkommen, ID={vocab_id} {in_vocab_str}")

    # Prüfen auf fehlende häufige Tokens
    missing_common = [word for word, count in word_counts.most_common(100) if word not in vocab]
    if missing_common:
        print(f"\n5. WARNUNG: {len(missing_common)} häufige Wörter fehlen im Vokabular!")
        print(f"  Fehlende häufige Wörter: {missing_common[:10]}" + ("..." if len(missing_common) > 10 else ""))
    else:
        print("\n5. Alle häufigen Wörter sind im Vokabular enthalten.")

    # Nicht verwendete Tokens im Vokabular
    unused_tokens = [token for token in vocab.keys() if token not in word_counts and token not in ["<PAD>", "<UNK>"]]
    if unused_tokens:
        print(f"\n6. {len(unused_tokens)} Tokens im Vokabular werden im Datensatz nicht verwendet:")
        print(f"  Beispiele: {unused_tokens[:10]}" + ("..." if len(unused_tokens) > 10 else ""))
    else:
        print("\n6. Alle Tokens im Vokabular werden verwendet.")

    # Vokabularabdeckung pro Satz
    coverage_per_sentence = []
    unknown_per_sentence = []
    tokens_per_sentence = []

    for sentence in df['sentence']:
        processed = dataset.preprocess_text(sentence)
        words = processed.split()
        tokens = dataset.tokenize(sentence)

        unknown = sum(1 for word in words if word not in vocab)
        coverage_per_sentence.append(1 - (unknown / len(words) if words else 0))
        unknown_per_sentence.append(unknown)
        tokens_per_sentence.append(len(tokens))

    avg_coverage = np.mean(coverage_per_sentence)
    avg_unknown = np.mean(unknown_per_sentence)
    avg_tokens = np.mean(tokens_per_sentence)

    print(f"\n7. Vokabularabdeckung pro Satz:")
    print(f"  Durchschnittliche Abdeckung: {avg_coverage:.2%}")
    print(f"  Durchschnittlich {avg_unknown:.2f} unbekannte Wörter pro Satz")
    print(f"  Durchschnittlich {avg_tokens:.2f} Tokens pro Satz")

    # Visualisierungen
    if visualize:
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Verteilung der häufigsten Wörter
        top_words = dict(most_common)

        plt.figure(figsize=(12, 8))
        plt.bar(list(top_words.keys()), list(top_words.values()))
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Wörter')
        plt.ylabel('Häufigkeit')
        plt.title('Häufigste Wörter im Datensatz')
        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'top_words.png'))
            plt.close()
        else:
            plt.show()

        # 2. Verteilung der Vokabularabdeckung pro Satz
        plt.figure(figsize=(10, 6))
        plt.hist(coverage_per_sentence, bins=20, alpha=0.7)
        plt.axvline(avg_coverage, color='r', linestyle='--', label=f'Durchschnitt: {avg_coverage:.2%}')
        plt.xlabel('Vokabularabdeckung')
        plt.ylabel('Anzahl der Sätze')
        plt.title('Verteilung der Vokabularabdeckung pro Satz')
        plt.legend()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'vocabulary_coverage.png'))
            plt.close()
        else:
            plt.show()

        # 3. Verteilung der Tokenlänge pro Satz
        plt.figure(figsize=(10, 6))
        plt.hist(tokens_per_sentence, bins=20, alpha=0.7)
        plt.axvline(avg_tokens, color='r', linestyle='--', label=f'Durchschnitt: {avg_tokens:.2f}')
        plt.xlabel('Anzahl der Tokens')
        plt.ylabel('Anzahl der Sätze')
        plt.title('Verteilung der Tokenlänge pro Satz')
        plt.legend()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'tokens_per_sentence.png'))
            plt.close()
        else:
            plt.show()

    # Detaillierte Vokabularliste speichern
    if output_dir:
        vocab_df = pd.DataFrame(
            [(token, idx, word_counts.get(token, 0), word_counts.get(token, 0)/total_words if total_words else 0)
             for token, idx in vocab.items()],
            columns=['Token', 'ID', 'Häufigkeit', 'Anteil']
        ).sort_values('Häufigkeit', ascending=False)

        vocab_df.to_csv(os.path.join(output_dir, 'vocabulary_analysis.csv'), index=False)
        print(f"\nDetailanalyse des Vokabulars gespeichert in {os.path.join(output_dir, 'vocabulary_analysis.csv')}")

def analyze_errors(model, dataset, visualize=False, output_dir=None, top_n=50):
    """
    Erweiterte Fehleranalyse mit Kategorisierung und Mustererkennung

    Args:
        model: Das zu bewertende Modell
        dataset: Das Dataset für die Bewertung
        visualize: Ob Visualisierungen erstellt werden sollen
        output_dir: Verzeichnis für Ausgabedateien (optional)
        top_n: Anzahl der Top-Fehler, die analysiert werden sollen
    """
    print("\n" + "="*80)
    print("Erweiterte Fehleranalyse")
    print("="*80)

    print("Analysiere Modellvorhersagen und identifiziere Fehlertypen...")
    df = pd.read_csv(get_absolute_path("../data/sentences.csv"))

    errors = []
    correctly_classified = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analysiere Vorhersagen"):
        sentence = row['sentence']
        true_label = row['label']
        true_label_text = "logisch" if true_label == 1 else "nicht logisch"

        analysis = test_sentence(model, dataset, sentence, verbose=False, return_details=True)
        pred_label_text = analysis["prediction"]
        pred_label = 1 if pred_label_text == "logisch" else 0

        if pred_label_text != true_label_text:
            errors.append({
                "sentence": sentence,
                "true_label": true_label_text,
                "true_label_idx": true_label,
                "pred_label": pred_label_text,
                "pred_label_idx": pred_label,
                "confidence": analysis["confidence"],
                "class_probabilities": analysis["class_probabilities"],
                "processed": analysis["processed"],
                "tokens": analysis["token_ids"],
                "unknown_words": analysis["unknown_words"],
                "unknown_ratio": analysis["unknown_ratio"],
                "word_token_mapping": analysis["word_token_mapping"]
            })
        else:
            correctly_classified.append({
                "sentence": sentence,
                "true_label": true_label_text,
                "pred_label": pred_label_text,
                "confidence": analysis["confidence"],
                "unknown_ratio": analysis["unknown_ratio"]
            })

    if errors:
        error_count = len(errors)
        error_rate = error_count / len(df) * 100
        print(f"\n1. Allgemeine Fehlerstatistik:")
        print(f"  {error_count} Fehler bei {len(df)} Sätzen ({error_rate:.2f}%)")

        # Analysiere Fehler nach Label-Typ
        errors_0_as_1 = [e for e in errors if e["true_label_idx"] == 0 and e["pred_label_idx"] == 1]
        errors_1_as_0 = [e for e in errors if e["true_label_idx"] == 1 and e["pred_label_idx"] == 0]

        print("\n2. Fehlertypen:")
        print(f"  'nicht logisch' als 'logisch' fehlklassifiziert: {len(errors_0_as_1)} ({len(errors_0_as_1)/error_count*100:.1f}% aller Fehler)")
        print(f"  'logisch' als 'nicht logisch' fehlklassifiziert: {len(errors_1_as_0)} ({len(errors_1_as_0)/error_count*100:.1f}% aller Fehler)")

        # Analysiere Konfidenz bei Fehlern
        confidence_errors_0_as_1 = [e["confidence"] for e in errors_0_as_1]
        confidence_errors_1_as_0 = [e["confidence"] for e in errors_1_as_0]

        print("\n3. Konfidenz bei Fehlern:")
        if confidence_errors_0_as_1:
            print(f"  'nicht logisch' als 'logisch': Durchschnitt {np.mean(confidence_errors_0_as_1):.4f}, Min {np.min(confidence_errors_0_as_1):.4f}, Max {np.max(confidence_errors_0_as_1):.4f}")
        if confidence_errors_1_as_0:
            print(f"  'logisch' als 'nicht logisch': Durchschnitt {np.mean(confidence_errors_1_as_0):.4f}, Min {np.min(confidence_errors_1_as_0):.4f}, Max {np.max(confidence_errors_1_as_0):.4f}")

        # Analysiere unbekannte Wörter bei Fehlern
        unknown_ratio_errors = [e["unknown_ratio"] for e in errors]
        unknown_ratio_correct = [c["unknown_ratio"] for c in correctly_classified]

        print("\n4. Unbekannte Wörter:")
        print(f"  Bei Fehlern: Durchschnitt {np.mean(unknown_ratio_errors):.4f}, Min {np.min(unknown_ratio_errors):.4f}, Max {np.max(unknown_ratio_errors):.4f}")
        print(f"  Bei korrekten Klassifikationen: Durchschnitt {np.mean(unknown_ratio_correct):.4f}, Min {np.min(unknown_ratio_correct):.4f}, Max {np.max(unknown_ratio_correct):.4f}")

        # Identifizierung von Mustern in Fehlern
        print("\n5. Musteranalyse in Fehlern:")

        # Textlängenanalyse
        error_lengths = [len(e["processed"].split()) for e in errors]
        correct_lengths = [len(c["sentence"].split()) for c in correctly_classified]

        print(f"  Durchschnittliche Satzlänge bei Fehlern: {np.mean(error_lengths):.2f} Wörter")
        print(f"  Durchschnittliche Satzlänge bei korrekten Vorhersagen: {np.mean(correct_lengths):.2f} Wörter")

        # Häufige Wörter in Fehlfällen
        error_words = []
        for e in errors:
            error_words.extend(e["processed"].split())

        error_word_counts = Counter(error_words)
        print("\n6. Häufigste Wörter in Fehlfällen:")
        for word, count in error_word_counts.most_common(10):
            print(f"  '{word}': {count} Vorkommen")

        # Häufigste Unbekannte Wörter
        unknown_words_in_errors = []
        for e in errors:
            unknown_words_in_errors.extend(e["unknown_words"])

        if unknown_words_in_errors:
            unknown_word_counts = Counter(unknown_words_in_errors)
            print("\n7. Häufigste unbekannte Wörter in Fehlfällen:")
            for word, count in unknown_word_counts.most_common(10):
                print(f"  '{word}': {count} Vorkommen")
        else:
            print("\n7. Keine unbekannten Wörter in Fehlfällen gefunden.")

        # Top N Fehler mit höchster Konfidenz
        top_confidence_errors = sorted(errors, key=lambda e: e["confidence"], reverse=True)[:top_n]
        print(f"\n8. Top {min(top_n, len(top_confidence_errors))} Fehler mit höchster Konfidenz:")
        for i, error in enumerate(top_confidence_errors[:10], 1):  # Zeige nur die ersten 10 für die Konsole
            print(f"  {i}. \"{error['sentence']}\"")
            print(f"     Tatsächlich: {error['true_label']}, Vorhersage: {error['pred_label']}, Konfidenz: {error['confidence']:.4f}")
            print(f"     Unbekannte Wörter: {error['unknown_words'] if error['unknown_words'] else 'keine'}")
            print()

        # Visualisierungen
        if visualize:
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 1. Konfidenzverteilung nach Fehlertyp
            plt.figure(figsize=(10, 6))
            if confidence_errors_0_as_1:
                sns.histplot(confidence_errors_0_as_1, kde=True, label="'nicht logisch' als 'logisch'", alpha=0.6)
            if confidence_errors_1_as_0:
                sns.histplot(confidence_errors_1_as_0, kde=True, label="'logisch' als 'nicht logisch'", alpha=0.6)
            plt.xlabel('Konfidenz')
            plt.ylabel('Anzahl')
            plt.title('Konfidenzverteilung nach Fehlertyp')
            plt.legend()

            if output_dir:
                plt.savefig(os.path.join(output_dir, 'error_confidence_distribution.png'))
                plt.close()
            else:
                plt.show()

            # 2. Verhältnis unbekannter Wörter bei Fehlern vs. korrekten Vorhersagen
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=[unknown_ratio_errors, unknown_ratio_correct],
                      labels=['Fehler', 'Korrekt'])
            plt.ylabel('Anteil unbekannter Wörter')
            plt.title('Unbekannte Wörter bei Fehlern vs. korrekten Vorhersagen')

            if output_dir:
                plt.savefig(os.path.join(output_dir, 'unknown_word_ratio.png'))
                plt.close()
            else:
                plt.show()

            # 3. Satzlängenverteilung bei Fehlern vs. korrekten Vorhersagen
            plt.figure(figsize=(10, 6))
            sns.histplot(error_lengths, kde=True, label='Fehler', alpha=0.6)
            sns.histplot(correct_lengths, kde=True, label='Korrekt', alpha=0.6)
            plt.xlabel('Satzlänge (Anzahl Wörter)')
            plt.ylabel('Anzahl')
            plt.title('Satzlängenverteilung: Fehler vs. korrekte Vorhersagen')
            plt.legend()

            if output_dir:
                plt.savefig(os.path.join(output_dir, 'sentence_length_distribution.png'))
                plt.close()
            else:
                plt.show()

        # Detaillierte Fehleranalyse speichern
        if output_dir:
            errors_df = pd.DataFrame([{
                'Satz': e['sentence'],
                'Tatsächlich': e['true_label'],
                'Vorhersage': e['pred_label'],
                'Konfidenz': e['confidence'],
                'Unbekannte_Wörter': ', '.join(e['unknown_words']) if e['unknown_words'] else '',
                'Unbekannter_Anteil': e['unknown_ratio'],
                'Tokens': str(e['tokens']),
                'Wahrscheinlichkeit_logisch': e['class_probabilities']['logisch'],
                'Wahrscheinlichkeit_nicht_logisch': e['class_probabilities']['nicht logisch']
            } for e in errors])

            errors_df.to_csv(os.path.join(output_dir, 'error_analysis.csv'), index=False)
            print(f"\nDetaillierte Fehleranalyse gespeichert in {os.path.join(output_dir, 'error_analysis.csv')}")
    else:
        print("\nKeine Fehler gefunden! Das Modell hat alle Testsätze korrekt klassifiziert.")

def test_custom_sentences(model, dataset, sentences=None, output_dir=None):
    """
    Erweiterte Testfunktion für benutzerdefinierte Sätze mit detaillierten Ergebnissen

    Args:
        model: Das trainierte Modell
        dataset: Das Dataset mit Vokabular
        sentences: Liste von zu testenden Sätzen (optional)
        output_dir: Verzeichnis für Ausgabedateien (optional)
    """
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

    print("\n" + "="*80)
    print("Test mit benutzerdefinierten Sätzen")
    print("="*80)

    results = []

    for sentence in sentences:
        analysis = test_sentence(model, dataset, sentence, verbose=True, return_details=True)
        results.append(analysis)

    # Speichere Ergebnisse, falls ein Ausgabeverzeichnis angegeben wurde
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results_df = pd.DataFrame([{
            'Satz': r['sentence'],
            'Vorhersage': r['prediction'],
            'Konfidenz': r['confidence'],
            'Tokens': str(r['token_ids']),
            'Unbekannte_Wörter': ', '.join(r['unknown_words']) if r['unknown_words'] else '',
            'Unbekannter_Anteil': r['unknown_ratio'],
            'Wahrscheinlichkeit_logisch': r['class_probabilities']['logisch'],
            'Wahrscheinlichkeit_nicht_logisch': r['class_probabilities']['nicht logisch']
        } for r in results])

        results_df.to_csv(os.path.join(output_dir, 'custom_sentences_results.csv'), index=False)
        print(f"\nErgebnisse gespeichert in {os.path.join(output_dir, 'custom_sentences_results.csv')}")

def main():
    parser = argparse.ArgumentParser(description='Erweitertes diagnostisches Tool für das Sentence-Classifier-Modell')
    parser.add_argument('--evaluate', action='store_true', help='Führe eine vollständige Modellbewertung durch')
    parser.add_argument('--vocab', action='store_true', help='Analysiere das Vokabular')
    parser.add_argument('--errors', action='store_true', help='Führe eine detaillierte Fehleranalyse durch')
    parser.add_argument('--test', action='store_true', help='Teste mit Beispielsätzen')
    parser.add_argument('--sentence', type=str, help='Teste einen spezifischen Satz')
    parser.add_argument('--examples', action='store_true', help='Führe alle Diagnosefunktionen mit Beispielen durch')
    parser.add_argument('--visualize', action='store_true', help='Erstelle Visualisierungen')
    parser.add_argument('--output-dir', type=str, default="diagnostic_results", help='Verzeichnis für Ausgabedateien')
    parser.add_argument('--top-errors', type=int, default=50, help='Anzahl der Top-Fehler für die Analyse')
    parser.add_argument('--all', action='store_true', help='Führe alle Diagnosefunktionen aus')

    args = parser.parse_args()

    # Modell und Vokabular laden
    model, dataset = load_model_and_vocab()

    # Ausgabeverzeichnis erstellen, falls erforderlich
    if args.visualize or args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Angeforderte Diagnosen ausführen
    if args.evaluate or args.all:
        evaluate_model(model, dataset, visualize=args.visualize, output_dir=args.output_dir)

    if args.vocab or args.all:
        vocabulary_stats(dataset, visualize=args.visualize, output_dir=args.output_dir)

    if args.errors or args.all:
        analyze_errors(model, dataset, visualize=args.visualize, output_dir=args.output_dir, top_n=args.top_errors)

    if args.test or args.all:
        test_custom_sentences(model, dataset, output_dir=args.output_dir)

    if args.sentence:
        test_sentence(model, dataset, args.sentence)

    if args.examples:
        print("\nBeispiel-Diagnose wird durchgeführt...")
        evaluate_model(model, dataset, visualize=args.visualize, output_dir=args.output_dir)
        vocabulary_stats(dataset, visualize=args.visualize, output_dir=args.output_dir)
        analyze_errors(model, dataset, visualize=args.visualize, output_dir=args.output_dir)
        test_custom_sentences(model, dataset, output_dir=args.output_dir)

    # Wenn keine spezifischen Argumente angegeben wurden, zeige die Hilfe an
    if not (args.evaluate or args.vocab or args.errors or args.test or args.sentence or args.examples or args.all):
        parser.print_help()

if __name__ == "__main__":
    import string  # Für Satzzeichenerkennung
    main()

