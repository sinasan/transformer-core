import openai
import pandas as pd
import time
import os
import argparse
import random
import sys
from dotenv import load_dotenv
from tqdm import tqdm
import json
load_dotenv("../.env")

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

# Verschiedene Themen für vielfältigere Sätze
THEMES = [
    "Alltag", "Technologie", "Natur", "Wissenschaft", "Philosophie",
    "Wirtschaft", "Sport", "Kultur", "Reisen", "Essen"
]

# Verschiedene Komplexitätsstufen
COMPLEXITY_LEVELS = ["einfach", "mittel", "komplex"]

# Sätze generieren mit variablen Themen und Komplexität
def generate_sentences(prompt, num_sentences=50, model="gpt-4-turbo-preview"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        sentences = response.choices[0].message.content.strip().split("\n")
        sentences = [s.strip("-•1234567890. \t") for s in sentences if len(s.strip()) > 0]
        return sentences[:num_sentences]
    except Exception as e:
        print(f"Fehler bei der API-Anfrage: {e}")
        time.sleep(5)  # Kurze Pause vor erneutem Versuch
        return []

# Satzvalidierung (logisch oder nicht)
def validate_sentence(sentence, model="gpt-4-turbo-preview"):
    try:
        prompt = f'Ist der folgende Satz logisch sinnvoll? Antworte ausschließlich mit "ja" oder "nein":\nSatz: "{sentence}"'
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().lower()
        return 1 if "ja" in result else 0
    except Exception as e:
        print(f"Fehler bei der Validierung: {e}")
        time.sleep(5)
        # Im Fehlerfall nutzen wir eine Fallback-Strategie: Ein sehr unlogischer Satz erhält 0
        if "der Himmel ist grün" in sentence or "Katzen können fliegen" in sentence:
            return 0
        return 1  # Sonst nehmen wir an, es ist logisch

def generate_specific_prompt(theme, complexity, is_logical=True):
    logical_state = "logische, sinnvolle" if is_logical else "grammatikalisch korrekte, aber logisch unsinnige"

    prompts = {
        "einfach": f"Generiere 20 {logical_state} deutsche Sätze zum Thema {theme}. Die Sätze sollten kurz und einfach sein.",
        "mittel": f"Generiere 20 {logical_state} deutsche Sätze zum Thema {theme}. Verwende mittlere Satzlänge mit etwas Variation.",
        "komplex": f"Generiere 20 {logical_state} deutsche Sätze zum Thema {theme}. Die Sätze sollten komplex sein mit Nebensätzen und anspruchsvollerem Vokabular."
    }

    return prompts[complexity]

def save_progress(sentences, labels, filepath="../data/sentences_progress.csv"):
    df = pd.DataFrame({
        "sentence": sentences,
        "label": labels
    })
    df.to_csv(filepath, index=False)
    print(f"Fortschritt gespeichert in {filepath}")

def print_examples(df):
    """Druckt einige Beispielsätze aus dem Datensatz"""
    print("\nBeispiele für logische Sätze:")
    logical_examples = df[df['label'] == 1].sample(min(5, sum(df['label'] == 1)))
    for _, row in logical_examples.iterrows():
        print(f"- {row['sentence']}")

    print("\nBeispiele für unlogische Sätze:")
    illogical_examples = df[df['label'] == 0].sample(min(5, sum(df['label'] == 0)))
    for _, row in illogical_examples.iterrows():
        print(f"- {row['sentence']}")

def print_stats(sentences, labels):
    """Druckt Statistiken zum Datensatz"""
    print("\nStatistiken des Datensatzes:")
    print(f"Gesamtzahl der Sätze: {len(sentences)}")
    print(f"Logische Sätze: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"Unlogische Sätze: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Generiere Beispielsätze für das Transformer-Training")
    parser.add_argument('--num_per_type', type=int, help='Anzahl zusätzlicher Sätze pro Kategorie (ERFORDERLICH)')
    parser.add_argument('--input', type=str, default="../data/sentences.csv", help='Eingabedatei (wird erweitert, falls vorhanden)')
    parser.add_argument('--output', type=str, default="../data/sentences.csv", help='Ausgabedatei')
    parser.add_argument('--model', type=str, default="gpt-4-turbo-preview", help='OpenAI Modell')
    parser.add_argument('--delay', type=float, default=0.5, help='Verzögerung zwischen API-Anfragen')
    parser.add_argument('--stats', action='store_true', help='Nur Statistiken der Eingabedatei ausgeben, ohne neue Sätze zu generieren')

    if len(sys.argv) == 1:
        parser.print_help()
        print("\nBeispiele:")
        print("  python generate_sentences.py --num_per_type 100    # Fügt 100 zusätzliche Sätze pro Kategorie hinzu")
        print("  python generate_sentences.py --stats               # Zeigt Statistiken der aktuellen Datei")
        print("  python generate_sentences.py --input existing.csv --output new.csv --num_per_type 200")
        sys.exit(1)

    args = parser.parse_args()

    # Überprüfen, ob die Eingabedatei existiert
    sentences, labels = [], []
    if os.path.exists(args.input):
        df = pd.read_csv(args.input)
        sentences = df["sentence"].tolist()
        labels = df["label"].tolist()
        print(f"Vorhandene Datei geladen: {args.input} mit {len(sentences)} Sätzen")

        # Aktuelle Statistik ausgeben
        print_stats(sentences, labels)
        print_examples(df)

        # Wenn nur Statistiken angefordert wurden, beenden wir hier
        if args.stats:
            sys.exit(0)
    else:
        print(f"Keine vorhandene Datei gefunden: {args.input}. Erzeuge neue Datei.")

    if args.num_per_type is None:
        print("FEHLER: --num_per_type muss angegeben werden!")
        parser.print_help()
        sys.exit(1)

    # Zählen, wie viele logische und unlogische Sätze wir bereits haben
    logical_count = sum(labels)
    illogical_count = len(labels) - logical_count

    # Ziel-Anzahl berechnen (bestehende Anzahl + Zielerhöhung pro Kategorie)
    TARGET_LOGICAL = logical_count + args.num_per_type
    TARGET_ILLOGICAL = illogical_count + args.num_per_type

    print(f"Aktuelle Anzahl: {logical_count} logische, {illogical_count} unlogische Sätze")
    print(f"Ziel: {TARGET_LOGICAL} logische, {TARGET_ILLOGICAL} unlogische Sätze")
    print(f"Es werden {args.num_per_type} neue Sätze pro Kategorie erzeugt.")

    # Für bessere Übersicht des Fortschritts
    pbar_logic = tqdm(total=TARGET_LOGICAL, initial=logical_count, desc="Logische Sätze")
    pbar_illogic = tqdm(total=TARGET_ILLOGICAL, initial=illogical_count, desc="Unlogische Sätze")

    # Zwischenspeicherung einrichten
    progress_file = "../data/sentences_progress.csv"
    save_interval = 20  # Speichern wir alle 20 neuen Sätze
    last_save = len(sentences)  # Zeitpunkt der letzten Speicherung

    # Prompts mit verschiedenen Themen und Komplexitäten generieren
    combinations = [(theme, level) for theme in THEMES for level in COMPLEXITY_LEVELS]
    random.shuffle(combinations)  # Zufällige Reihenfolge für bessere Verteilung

    # Liste, um bereits gesehene Sätze zu speichern und Duplikate zu vermeiden
    existing_sentences = set(sentences)

    for theme, complexity in combinations:
        # Falls wir genügend Sätze beider Kategorien haben, brechen wir ab
        if logical_count >= TARGET_LOGICAL and illogical_count >= TARGET_ILLOGICAL:
            break

        # Generieren logische Sätze, wenn wir noch mehr benötigen
        if logical_count < TARGET_LOGICAL:
            logical_prompt = generate_specific_prompt(theme, complexity, is_logical=True)
            logical_sentences = generate_sentences(logical_prompt,
                                                  num_sentences=min(20, TARGET_LOGICAL - logical_count),
                                                  model=args.model)

            for sentence in logical_sentences:
                # Duplikatprüfung
                if sentence in existing_sentences:
                    continue

                # Validieren und hinzufügen
                label = validate_sentence(sentence, model=args.model)
                sentences.append(sentence)
                labels.append(label)
                existing_sentences.add(sentence)

                if label == 1:
                    logical_count += 1
                    pbar_logic.update(1)
                else:
                    illogical_count += 1
                    pbar_illogic.update(1)

                time.sleep(args.delay)

                # Regelmäßiges Speichern
                if len(sentences) - last_save >= save_interval:
                    save_progress(sentences, labels, progress_file)
                    last_save = len(sentences)

        # Generieren unlogische Sätze, wenn wir noch mehr benötigen
        if illogical_count < TARGET_ILLOGICAL:
            illogical_prompt = generate_specific_prompt(theme, complexity, is_logical=False)
            illogical_sentences = generate_sentences(illogical_prompt,
                                                    num_sentences=min(20, TARGET_ILLOGICAL - illogical_count),
                                                    model=args.model)

            for sentence in illogical_sentences:
                # Duplikatprüfung
                if sentence in existing_sentences:
                    continue

                # Validieren und hinzufügen
                label = validate_sentence(sentence, model=args.model)
                sentences.append(sentence)
                labels.append(label)
                existing_sentences.add(sentence)

                if label == 0:
                    illogical_count += 1
                    pbar_illogic.update(1)
                else:
                    logical_count += 1
                    pbar_logic.update(1)

                time.sleep(args.delay)

                # Regelmäßiges Speichern
                if len(sentences) - last_save >= save_interval:
                    save_progress(sentences, labels, progress_file)
                    last_save = len(sentences)

    pbar_logic.close()
    pbar_illogic.close()

    # Endgültiges DataFrame erstellen und speichern
    df = pd.DataFrame({
        "sentence": sentences,
        "label": labels
    })

    # Statistiken drucken
    print_stats(sentences, labels)

    # Speichern
    df.to_csv(args.output, index=False)
    print(f"Datensatz gespeichert als '{args.output}'.")

    # Einige Beispiele anzeigen
    print_examples(df)

    # Aufräumen
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"Fortschrittsdatei {progress_file} gelöscht.")

if __name__ == "__main__":
    main()
