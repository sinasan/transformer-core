#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from dataset import SentenceDataset, collate_fn
from model import SimpleTransformer
import json
import os
import argparse
import numpy as np
from tqdm import tqdm


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup-Scheduler, der die Lernrate linear erhöht und dann auf einen anderen Scheduler übergeht
    """
    def __init__(self, optimizer, warmup_steps, after_scheduler=None):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = 0
        super(WarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_steps:
            if self.after_scheduler and not self.finished:
                self.finished = True
                self.after_scheduler.base_lrs = self.base_lrs
                return self.after_scheduler.get_lr()
            return self.base_lrs
        return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_steps)
        else:
            return super(WarmupScheduler, self).step(epoch)


def train(force_rebuild_vocab=False):
    # Konfiguration laden
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Hyperparameter
    embedding_dim = config.get("embedding_dim", 128)
    num_heads = config.get("num_heads", 4)
    num_layers = config.get("num_layers", 2)
    num_classes = config.get("num_classes", 2)
    batch_size = config.get("batch_size", 32)
    num_epochs = config.get("num_epochs", 10)
    learning_rate = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.01)
    gradient_clip_val = config.get("gradient_clip_val", 1.0)
    validation_split = config.get("validation_split", 0.1)

    # Scheduler-Parameter
    use_lr_scheduler = config.get("use_lr_scheduler", False)
    lr_scheduler_type = config.get("lr_scheduler_type", "reduce_on_plateau")
    lr_scheduler_factor = config.get("lr_scheduler_factor", 0.5)
    lr_scheduler_patience = config.get("lr_scheduler_patience", 2)
    use_warmup = config.get("use_warmup", False)
    warmup_steps = config.get("warmup_steps", 100)

    # Early Stopping
    use_early_stopping = config.get("early_stopping", True)
    early_stopping_patience = config.get("early_stopping_patience", 5)
    early_stopping_min_delta = config.get("early_stopping_min_delta", 0.0)

    # Metrik für Early Stopping
    eval_metric = config.get("eval_metric", "f1")  # 'accuracy', 'f1', oder 'balanced_accuracy'

    # Datensatz laden mit Option zum erzwungenen Neuaufbau des Vokabulars
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(PROJECT_ROOT, "data", "sentences.csv")

    print(f"Lade Datensatz aus {data_path}...")
    print(f"Vokabular-Neuaufbau erzwingen: {force_rebuild_vocab}")

    dataset = SentenceDataset(csv_file=data_path, force_rebuild_vocab=force_rebuild_vocab)

    # Datensatz stratifiziert aufteilen
    dataset_size = len(dataset)
    train_size = int((1.0 - validation_split) * dataset_size)
    val_size = dataset_size - train_size

    indices = list(range(dataset_size))
    np.random.seed(42)  # Für Reproduzierbarkeit
    np.random.shuffle(indices)

    # Stratifizierte Aufteilung - sicherstellen, dass beide Splits ähnliche Klassenverteilungen haben
    train_indices = []
    val_indices = []
    class_counts = [0, 0]  # Anzahl der Instanzen für jede Klasse

    # Zähle die Klasseninstanzen im Datensatz
    for i in range(dataset_size):
        _, label = dataset[i]
        class_counts[label.item()] += 1

    # Ziel-Verhältnis für jede Klasse im Validierungssatz
    target_val_counts = [int(count * validation_split) for count in class_counts]
    current_val_counts = [0, 0]

    for idx in indices:
        _, label = dataset[idx]
        label_idx = label.item()

        if current_val_counts[label_idx] < target_val_counts[label_idx]:
            val_indices.append(idx)
            current_val_counts[label_idx] += 1
        else:
            train_indices.append(idx)

    # Erstelle Subset-Datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Datensatz stratifiziert aufgeteilt: {len(train_indices)} Trainingssätze, {len(val_indices)} Validierungssätze")
    print(f"Klassenverteilung im Validierungssatz: {current_val_counts}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Modell initialisieren
    model = SimpleTransformer(vocab_size=len(dataset.vocab))
    print(f"Modell initialisiert mit Vokabulargröße: {len(dataset.vocab)}")
    print(f"Modellparameter: Embedding-Dim={embedding_dim}, Heads={num_heads}, Layers={num_layers}")

    # Standard Loss-Funktion
    criterion = nn.CrossEntropyLoss()
    print("Standard-Verlustfunktion ohne Klassengewichte verwendet")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning Rate Scheduler konfigurieren
    scheduler = None
    if use_lr_scheduler:
        if lr_scheduler_type == "reduce_on_plateau":
            base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=lr_scheduler_factor,
                patience=lr_scheduler_patience, min_lr=1e-6
            )
        elif lr_scheduler_type == "cosine_annealing":
            base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-6
            )
        else:
            print(f"Warnung: Unbekannter Scheduler-Typ '{lr_scheduler_type}', verwende ReduceLROnPlateau")
            base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=lr_scheduler_factor,
                patience=lr_scheduler_patience
            )

        # Warmup-Wrapper um den Basis-Scheduler (falls aktiviert)
        if use_warmup and warmup_steps > 0:
            print(f"Verwende Warmup-Scheduler mit {warmup_steps} Warmup-Schritten")
            if lr_scheduler_type != "reduce_on_plateau":
                # Für Scheduler, die pro Schritt aufgerufen werden
                scheduler = WarmupScheduler(optimizer, warmup_steps, base_scheduler)
            else:
                # ReduceLROnPlateau wird separat behandelt, da es von Metriken abhängt
                scheduler = {"warmup": WarmupScheduler(optimizer, warmup_steps),
                             "main": base_scheduler}
        else:
            scheduler = base_scheduler

    # Beste Modellgewichte für Early Stopping speichern
    best_metric_value = 0.0
    best_model_weights = None
    patience_counter = 0
    last_improvement = 0.0

    # Training-Loop
    print(f"Starte Training für {num_epochs} Epochen mit Early Stopping (Patience: {early_stopping_patience}, Min Delta: {early_stopping_min_delta})...")
    print(f"Evaluation basierend auf Metrik: {eval_metric}")
    global_step = 0

    # Funktion zur Berechnung der Evaluationsmetrik
    def calculate_metric(y_true, y_pred):
        if eval_metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif eval_metric == "f1":
            return f1_score(y_true, y_pred, average='macro')
        elif eval_metric == "balanced_accuracy":
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score(y_true, y_pred)
        else:
            print(f"Unbekannte Metrik: {eval_metric}, verwende F1-Score")
            return f1_score(y_true, y_pred, average='macro')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

        for batch_idx, (sentences, labels) in enumerate(train_pbar):
            global_step += 1

            optimizer.zero_grad()

            outputs = model(sentences)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradientenclipping für stabileres Training
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)

            # Erst Optimizer-Schritt, dann Scheduler (wichtig für korrekte Reihenfolge!)
            optimizer.step()

            # Warmup-Scheduler Schritt (falls verwendet)
            if use_warmup and warmup_steps > 0 and global_step <= warmup_steps:
                if isinstance(scheduler, dict):
                    scheduler["warmup"].step()
                elif not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

            total_loss += loss.item()

            # Aktuelle Lernrate anzeigen
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})

        avg_loss = total_loss / len(train_loader)

        # Validierung nach jeder Epoche
        model.eval()
        val_predictions, val_labels = [], []
        val_loss = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
            for sentences, labels in val_pbar:
                outputs = model(sentences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_predictions.extend(preds.tolist())
                val_labels.extend(labels.tolist())

                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_f1 = f1_score(val_labels, val_predictions, average='macro')

        # Berechne die aktuelle Metrik für Early Stopping
        current_metric = calculate_metric(val_labels, val_predictions)

        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | '
              f'Val Accuracy: {val_accuracy:.4f} | Val F1: {val_f1:.4f} | {eval_metric}: {current_metric:.4f}')

        # Learning Rate Scheduler aktualisieren (falls aktiviert)
        if use_lr_scheduler and (not use_warmup or global_step > warmup_steps):
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) or \
               (isinstance(scheduler, dict) and "main" in scheduler):
                # ReduceLROnPlateau nimmt Metriken als Eingabe
                if isinstance(scheduler, dict):
                    scheduler["main"].step(current_metric)
                else:
                    scheduler.step(current_metric)
            elif isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) and \
                 not isinstance(scheduler, WarmupScheduler):
                # Andere Scheduler werden pro Epoche aufgerufen
                scheduler.step()

        # Ausführlicher Validierungsbericht (alle 5 Epochen oder am Ende)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1 or epoch < 3:
            print("\nValidation Report:")
            print(classification_report(val_labels, val_predictions, target_names=['nicht logisch', 'logisch']))

            # Detaillierte Fehleranalyse
            errors_0_as_1 = sum([1 for pred, true in zip(val_predictions, val_labels) if pred == 1 and true == 0])
            errors_1_as_0 = sum([1 for pred, true in zip(val_predictions, val_labels) if pred == 0 and true == 1])
            print(f"Fehleranalyse:")
            print(f"  'nicht logisch' als 'logisch' klassifiziert: {errors_0_as_1} ({errors_0_as_1/val_labels.count(0)*100:.1f}% aller 'nicht logisch')")
            print(f"  'logisch' als 'nicht logisch' klassifiziert: {errors_1_as_0} ({errors_1_as_0/val_labels.count(1)*100:.1f}% aller 'logisch')")

        # Early Stopping mit min_delta
        improvement = current_metric - best_metric_value

        if improvement > early_stopping_min_delta:
            best_metric_value = current_metric
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
            last_improvement = improvement
            print(f"Neues bestes Modell gespeichert mit {eval_metric}: {best_metric_value:.4f} (Verbesserung: {improvement:.4f})")

            # Speichere das beste Modell sofort
            model_save_path = os.path.join(PROJECT_ROOT, "models", "transformer_model_best.pth")
            torch.save(best_model_weights, model_save_path)
            print(f"Bestes Modell gespeichert in {model_save_path}")
        else:
            patience_counter += 1
            print(f"Keine signifikante Verbesserung (min_delta={early_stopping_min_delta}). "
                  f"Aktuelle Änderung: {improvement:.4f}, Beste bisher: {last_improvement:.4f}. "
                  f"Patience: {patience_counter}/{early_stopping_patience}")

        if use_early_stopping and patience_counter >= early_stopping_patience:
            print(f"Early Stopping nach Epoche {epoch+1}. Keine Verbesserung über min_delta={early_stopping_min_delta} "
                  f"für {early_stopping_patience} Epochen.")
            break

    # Lade das beste Modell für die Ausgabe
    print(f"Training abgeschlossen nach {epoch+1} Epochen.")
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"Bestes Modell mit {eval_metric}={best_metric_value:.4f} wiederhergestellt")

    # Finales Modell separat speichern
    model_save_path = os.path.join(PROJECT_ROOT, "models", "transformer_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Finales Modell gespeichert in {model_save_path}")

    # Finale Evaluation auf dem Validierungsdatensatz mit geladenen besten Modellgewichten
    print("\nFinale Evaluation auf dem Validierungsdatensatz:")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sentences, labels in val_loader:
            outputs = model(sentences)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Finale Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=['nicht logisch', 'logisch']))

    # Detaillierte Fehleranalyse für die beste Lösung
    errors_0_as_1 = sum([1 for pred, true in zip(all_preds, all_labels) if pred == 1 and true == 0])
    errors_1_as_0 = sum([1 for pred, true in zip(all_preds, all_labels) if pred == 0 and true == 1])
    total_0 = all_labels.count(0)
    total_1 = all_labels.count(1)

    print(f"Finale Fehleranalyse:")
    print(f"  'nicht logisch' als 'logisch' klassifiziert: {errors_0_as_1} ({errors_0_as_1/total_0*100:.1f}% der Klasse)")
    print(f"  'logisch' als 'nicht logisch' klassifiziert: {errors_1_as_0} ({errors_1_as_0/total_1*100:.1f}% der Klasse)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training des Transformer-Modells für Satzklassifikation")
    parser.add_argument('--rebuild-vocab', action='store_true',
                      help='Erzwingt den Neuaufbau des Vokabulars')

    args = parser.parse_args()

    train(force_rebuild_vocab=args.rebuild_vocab)
