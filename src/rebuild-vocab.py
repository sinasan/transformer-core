#!/usr/bin/env python

"""
Einfaches Skript, das das Vokabular neu erstellt und die Problempunkte testet.
"""

import os
import sys
import json
from pathlib import Path

# Füge das Projekt-Root-Verzeichnis zum Python-Pfad hinzu
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.dataset import SentenceDataset

def main():
    # Vokabularpfad
    vocab_path = os.path.join(PROJECT_ROOT, "data", "vocab.json")

    # Backup des alten Vokabulars, falls vorhanden
    if os.path.exists(vocab_path):
        backup_path = vocab_path + ".backup"
        print(f"Sichere altes Vokabular nach: {backup_path}")
        import shutil
        shutil.copy2(vocab_path, backup_path)

        # Lösche das vorhandene Vokabular
        os.remove(vocab_path)
        print(f"Altes Vokabular gelöscht.")

    # Erstelle neues Vokabular
    print("Erstelle neues Vokabular mit verbesserter Tokenisierung...")
    dataset = SentenceDataset(force_rebuild_vocab=True)

    # Teste die problematischen Beispiele
    example_sentences = [
        "Die meisten Menschen haben zwei Beine und zwei Arme.",
        "Computer arbeiten mit elektrischen Signalen.",
        "Der Tisch ist traurig über die Situation.",
        "Berge können schwimmen und Flüsse klettern.",
        "Die Wolken singen heute besonders schön."
    ]

    print("\nTeste die problematischen Beispiele:")
    for sentence in example_sentences:
        # Vorverarbeitung
        processed = dataset.preprocess_text(sentence)

        # Tokens und IDs
        tokens = processed.split()
        token_ids = dataset.tokenize(sentence)

        print(f"\nSatz: {sentence}")
        print(f"Vorverarbeitet: {processed}")

        # Zeige Token-zu-ID Mapping
        print("Token-Analyse:")
        for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
            if token_id == 0 and token != "<PAD>":
                status = "UNBEKANNT!"
            else:
                status = "OK"
            print(f"  {token}: ID {token_id} ({status})")

    print("\nVokabular-Neuaufbau abgeschlossen!")
    print(f"Neues Vokabular hat {len(dataset.vocab)} Tokens.")

if __name__ == "__main__":
    main()
