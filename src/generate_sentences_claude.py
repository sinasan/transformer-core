import anthropic
import pandas as pd
import time
import os
import argparse
import random
import sys
from dotenv import load_dotenv
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv("../.env")

model1 = "claude-3-7-sonnet-20250219"
model2 = "claude-3-opus-20240229"
model3 = "claude-3-5-sonnet-20240620"
model4 = "claude-3-haiku-20240307"

model = model4

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Verschiedene Themen für vielfältigere Sätze
THEMES = [
    "Alltag", "Technologie", "Natur", "Wissenschaft", "Philosophie",
    "Wirtschaft", "Sport", "Kultur", "Reisen", "Essen", "Tiere", "Menschen",
    "Objekte", "Fakten", "Beziehungen", "Emotionen"
]

# Verschiedene Komplexitätsstufen
COMPLEXITY_LEVELS = ["einfach", "mittel", "komplex"]

# Verschiedene logische Kategorien für bessere Abdeckung (faktisch korrekte Sätze)
LOGICAL_CATEGORIES = [
    "Faktuell korrekt",
    "Allgemeinwissen",
    "Naturwissenschaftliche Fakten",
    "Alltägliche Wahrheiten",
    "Gängige Kausalzusammenhänge"
]

# Verschiedene unlogische Kategorien für bessere Abdeckung (faktisch falsche oder unsinnige Sätze)
ILLOGICAL_CATEGORIES = [
    "Faktisch falsche Aussagen",
    "Personifizierung unbelebter Objekte",
    "Unmögliche Handlungen",
    "Kategorische Widersprüche",
    "Falsche Behauptungen"
]

def generate_sentences(prompt, num_sentences=20, model=model):
    """Generiert Sätze basierend auf einem gegebenen Prompt mit der Anthropic API"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            sentences = response.content[0].text.strip().split("\n")
            sentences = [s.strip("-•1234567890. \t") for s in sentences if len(s.strip()) > 0]
            return sentences[:num_sentences]
        except Exception as e:
            print(f"Fehler bei der API-Anfrage (Versuch {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponentielles Backoff
                print(f"Warte {wait_time} Sekunden vor erneutem Versuch...")
                time.sleep(wait_time)
            else:
                print("Maximale Anzahl von Versuchen erreicht. Überspringe diese Anfrage.")
                return []

def validate_sentence(sentence, model=model):
    """
    Validiert, ob ein Satz faktisch korrekt ist, mit der Anthropic API.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prompt = f"""
Beurteile, ob der folgende Satz faktisch korrekt ist. Ein Satz gilt als faktisch korrekt, wenn er:

1. Der Realität entspricht und faktisch wahr ist
2. Keine falschen oder irreführenden Behauptungen enthält

WICHTIG: Faktisch falsche Aussagen gelten als unlogisch. Zum Beispiel:
- "Ein Löffel ist eine Gabel" - faktisch falsch, also unlogisch
- "Hunde bellen nicht gerne" - faktisch falsch, also unlogisch

Sätze mit Personifizierungen oder unmöglichen Handlungen gelten ebenfalls als unlogisch:
- "Der Tisch ist traurig" - unlogisch (Personifizierung)
- "Der Berg fliegt" - unlogisch (unmögliche Handlung)

Antworte mit "ja" für faktisch korrekte (logische) Sätze oder "nein" für faktisch falsche (unlogische) Sätze.

Satz: "{sentence}"
"""
            response = client.messages.create(
                model=model,
                max_tokens=5,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.content[0].text.strip().lower()
            return 1 if "ja" in result else 0
        except Exception as e:
            print(f"Fehler bei der Validierung (Versuch {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponentielles Backoff
                print(f"Warte {wait_time} Sekunden vor erneutem Versuch...")
                time.sleep(wait_time)
            else:
                print("Maximale Anzahl von Versuchen erreicht. Fallback-Strategie wird verwendet.")
                # Im Fehlerfall nutzen wir eine Fallback-Strategie
                if "spricht" in sentence and any(obj in sentence for obj in ["Tisch", "Stein", "Berg", "Haus"]):
                    return 0  # Personifizierung unbelebter Objekte
                return 1  # Im Zweifelsfall als logisch einstufen

def generate_logical_prompt(category, theme, complexity):
    """Generiert einen Prompt für logische (faktisch korrekte) Sätze basierend auf Kategorie, Thema und Komplexität"""
    base_prompt = f"Generiere 20 deutsche Sätze zum Thema {theme}."

    category_instructions = {
        "Faktuell korrekt": "Die Sätze sollten faktisch korrekte Aussagen sein.",
        "Allgemeinwissen": "Die Sätze sollten faktisch korrekte Aussagen sein, die allgemeines Wissen darstellen.",
        "Naturwissenschaftliche Fakten": "Die Sätze sollten faktisch korrekte naturwissenschaftliche Aussagen sein.",
        "Alltägliche Wahrheiten": "Die Sätze sollten faktisch korrekte Aussagen über alltägliche Dinge sein.",
        "Gängige Kausalzusammenhänge": "Die Sätze sollten faktisch korrekte Kausalzusammenhänge beschreiben."
    }

    complexity_instructions = {
        "einfach": "Die Sätze sollten kurz und einfach sein.",
        "mittel": "Die Sätze sollten mittlere Länge und Komplexität haben.",
        "komplex": "Die Sätze sollten komplex sein, mit Nebensätzen und anspruchsvollerem Vokabular."
    }

    return f"{base_prompt} {category_instructions[category]} {complexity_instructions[complexity]}"

def generate_illogical_prompt(category, theme, complexity):
    """Generiert einen Prompt für unlogische (faktisch falsche) Sätze basierend auf Kategorie, Thema und Komplexität"""
    base_prompt = f"Generiere 20 grammatikalisch korrekte, aber faktisch falsche deutsche Sätze zum Thema {theme}."

    category_instructions = {
        "Faktisch falsche Aussagen": "Die Sätze sollten Behauptungen enthalten, die faktisch falsch sind, wie 'Ein Löffel ist eine Gabel' oder 'Hunde bellen nicht gerne'.",
        "Personifizierung unbelebter Objekte": "Die Sätze sollten unbelebten Objekten menschliche Eigenschaften zuschreiben. Beispiel: 'Die Steine diskutieren über den Sinn des Lebens.'",
        "Unmögliche Handlungen": "Die Sätze sollten physikalisch unmögliche Handlungen beschreiben. Beispiel: 'Der Berg schwimmt durch den Ozean.'",
        "Kategorische Widersprüche": "Die Sätze sollten kategorische Widersprüche enthalten. Beispiel: 'Der viereckige Kreis hat eine interessante Geometrie.'",
        "Falsche Behauptungen": "Die Sätze sollten falsche Behauptungen über die Realität enthalten. Beispiel: 'In Deutschland ist Französisch die Amtssprache.'"
    }

    complexity_instructions = {
        "einfach": "Die Sätze sollten kurz und einfach sein.",
        "mittel": "Die Sätze sollten mittlere Länge und Komplexität haben.",
        "komplex": "Die Sätze sollten komplex sein, mit Nebensätzen und anspruchsvollerem Vokabular."
    }

    return f"{base_prompt} {category_instructions[category]} {complexity_instructions[complexity]}"

def save_progress(sentences, labels, filepath="../data/sentences_progress.csv"):
    """Speichert den aktuellen Fortschritt"""
    df = pd.DataFrame({
        "sentence": sentences,
        "label": labels
    })
    df.to_csv(filepath, index=False)
    print(f"Fortschritt gespeichert in {filepath}")

def print_examples(df, n=5):
    """Druckt einige Beispielsätze aus dem Datensatz"""
    print("\nBeispiele für faktisch korrekte (logische) Sätze:")
    logical_examples = df[df['label'] == 1].sample(min(n, sum(df['label'] == 1)))
    for _, row in logical_examples.iterrows():
        print(f"- {row['sentence']}")

    print("\nBeispiele für faktisch falsche (unlogische) Sätze:")
    illogical_examples = df[df['label'] == 0].sample(min(n, sum(df['label'] == 0)))
    for _, row in illogical_examples.iterrows():
        print(f"- {row['sentence']}")

def print_stats(sentences, labels):
    """Druckt Statistiken zum Datensatz"""
    print("\nStatistiken des Datensatzes:")
    print(f"Gesamtzahl der Sätze: {len(sentences)}")
    print(f"Faktisch korrekte (logische) Sätze: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"Faktisch falsche (unlogische) Sätze: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Generiere Beispielsätze für das Transformer-Training mit Anthropic API")
    parser.add_argument('--num_per_type', type=int, help='Anzahl zusätzlicher Sätze insgesamt (ERFORDERLICH)')
    parser.add_argument('--input', type=str, default="../data/sentences.csv", help='Eingabedatei (wird erweitert, falls vorhanden)')
    parser.add_argument('--output', type=str, default="../data/sentences.csv", help='Ausgabedatei')
    parser.add_argument('--model', type=str, default=f"{model}", help='Anthropic Modell')
    parser.add_argument('--delay', type=float, default=0.5, help='Verzögerung zwischen API-Anfragen')
    parser.add_argument('--stats', action='store_true', help='Nur Statistiken der Eingabedatei ausgeben, ohne neue Sätze zu generieren')
    parser.add_argument('--examples', type=int, default=5, help='Anzahl der Beispiele, die angezeigt werden sollen')
    parser.add_argument('--balance-categories', action='store_true', help='Gleichmäßige Verteilung über alle logischen/unlogischen Kategorien')
    parser.add_argument('--force-logical', action='store_true', help='Generiere nur logische Sätze')
    parser.add_argument('--force-illogical', action='store_true', help='Generiere nur unlogische Sätze')

    if len(sys.argv) == 1:
        parser.print_help()
        print("\nBeispiele:")
        print("  python generate_sentences_anthropic.py --num_per_type 100    # Fügt 100 Sätze hinzu, automatisch ausgeglichen")
        print("  python generate_sentences_anthropic.py --num_per_type 100 --force-logical   # Fügt 100 logische Sätze hinzu")
        print("  python generate_sentences_anthropic.py --num_per_type 100 --force-illogical # Fügt 100 unlogische Sätze hinzu")
        print("  python generate_sentences_anthropic.py --stats               # Zeigt Statistiken der aktuellen Datei")
        print("  python generate_sentences_anthropic.py --num_per_type 100 --balance-categories # Gleichmäßige Verteilung über Kategorien")
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
        print_examples(df, args.examples)

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

    # Bestimmen, wie viele Sätze jedes Typs wir generieren sollen
    if args.force_logical:
        # Nur logische Sätze
        new_logical = args.num_per_type
        new_illogical = 0
    elif args.force_illogical:
        # Nur unlogische Sätze
        new_logical = 0
        new_illogical = args.num_per_type
    else:
        # Automatische Balance - mehr von dem, was wir weniger haben
        total = logical_count + illogical_count

        # Wenn der Datensatz bereits perfekt ausgewogen ist
        if logical_count == illogical_count:
            new_logical = args.num_per_type // 2
            new_illogical = args.num_per_type - new_logical
        # Wenn wir mehr logische Sätze brauchen
        elif logical_count < illogical_count:
            # Wie viele logische bräuchten wir für Balance?
            needed_for_balance = illogical_count - logical_count

            # Wenn wir genug hinzufügen können, um auszugleichen
            if needed_for_balance <= args.num_per_type:
                new_logical = needed_for_balance
                new_illogical = args.num_per_type - new_logical
            # Wenn wir nicht genug hinzufügen können, fügen wir alle als logisch hinzu
            else:
                new_logical = args.num_per_type
                new_illogical = 0
        # Wenn wir mehr unlogische Sätze brauchen
        else:
            # Wie viele unlogische bräuchten wir für Balance?
            needed_for_balance = logical_count - illogical_count

            # Wenn wir genug hinzufügen können, um auszugleichen
            if needed_for_balance <= args.num_per_type:
                new_illogical = needed_for_balance
                new_logical = args.num_per_type - new_illogical
            # Wenn wir nicht genug hinzufügen können, fügen wir alle als unlogisch hinzu
            else:
                new_illogical = args.num_per_type
                new_logical = 0

    print(f"Aktuelle Anzahl: {logical_count} logische, {illogical_count} unlogische Sätze")
    print(f"Geplant: {new_logical} neue logische, {new_illogical} neue unlogische Sätze")
    print(f"Zukünftige Anzahl: {logical_count + new_logical} logische, {illogical_count + new_illogical} unlogische Sätze")

    # Parameter für die Kategoriebalance
    if args.balance_categories:
        # Anzahl Sätze pro Kategorie, falls gleichmäßig verteilt
        logical_per_category = max(1, new_logical // len(LOGICAL_CATEGORIES))
        illogical_per_category = max(1, new_illogical // len(ILLOGICAL_CATEGORIES))

        # Zähler für jede Kategorie initialisieren
        logical_category_counts = {cat: 0 for cat in LOGICAL_CATEGORIES}
        illogical_category_counts = {cat: 0 for cat in ILLOGICAL_CATEGORIES}

        print(f"Bei Kategorie-Balance: ca. {logical_per_category} pro logische Kategorie, ca. {illogical_per_category} pro unlogische Kategorie")

    # Für bessere Übersicht des Fortschritts
    if new_logical > 0:
        pbar_logic = tqdm(total=new_logical, desc="Neue logische Sätze")
    else:
        pbar_logic = None

    if new_illogical > 0:
        pbar_illogic = tqdm(total=new_illogical, desc="Neue unlogische Sätze")
    else:
        pbar_illogic = None

    # Zählen, wie viele neue Sätze bereits hinzugefügt wurden
    added_logical = 0
    added_illogical = 0

    # Zwischenspeicherung einrichten
    progress_file = "../data/sentences_progress.csv"
    save_interval = 20  # Speichern wir alle 20 neuen Sätze
    last_save = len(sentences)  # Zeitpunkt der letzten Speicherung

    # Liste, um bereits gesehene Sätze zu speichern und Duplikate zu vermeiden
    existing_sentences = set(sentences)

    # Kombinationen von Themen, Kategorien und Komplexitäten erstellen
    logical_combinations = []
    illogical_combinations = []

    for theme in THEMES:
        for complexity in COMPLEXITY_LEVELS:
            for category in LOGICAL_CATEGORIES:
                logical_combinations.append((category, theme, complexity))
            for category in ILLOGICAL_CATEGORIES:
                illogical_combinations.append((category, theme, complexity))

    # Zufällige Reihenfolge für bessere Verteilung
    random.shuffle(logical_combinations)
    random.shuffle(illogical_combinations)

    # Generator-Loop
    while ((added_logical < new_logical and logical_combinations) or
           (added_illogical < new_illogical and illogical_combinations)):

        # Logische Sätze generieren
        if added_logical < new_logical and logical_combinations:
            # Wähle eine Kombination aus
            category, theme, complexity = logical_combinations.pop(0)

            # Überprüfe bei Balance-Option, ob diese Kategorie noch Sätze benötigt
            if args.balance_categories and logical_category_counts[category] >= logical_per_category:
                continue

            # Generiere Prompt und Sätze
            logical_prompt = generate_logical_prompt(category, theme, complexity)
            num_to_generate = min(20, new_logical - added_logical)
            logical_sentences = generate_sentences(logical_prompt,
                                                 num_sentences=num_to_generate,
                                                 model=args.model)

            for sentence in logical_sentences:
                # Prüfen, ob wir noch logische Sätze benötigen
                if added_logical >= new_logical:
                    break

                # Duplikatprüfung
                if sentence in existing_sentences:
                    continue

                # Validieren und hinzufügen
                label = validate_sentence(sentence, model=args.model)
                sentences.append(sentence)
                labels.append(label)
                existing_sentences.add(sentence)

                if label == 1:
                    added_logical += 1
                    if pbar_logic:
                        pbar_logic.update(1)
                    if args.balance_categories:
                        logical_category_counts[category] += 1
                else:
                    # Wenn wir noch unlogische Sätze brauchen, zählen wir ihn
                    if added_illogical < new_illogical:
                        added_illogical += 1
                        if pbar_illogic:
                            pbar_illogic.update(1)
                    # Sonst überspringen wir ihn
                    else:
                        sentences.pop()
                        labels.pop()
                        existing_sentences.remove(sentence)

                time.sleep(args.delay)

                # Regelmäßiges Speichern
                if len(sentences) - last_save >= save_interval:
                    save_progress(sentences, labels, progress_file)
                    last_save = len(sentences)

        # Unlogische Sätze generieren
        if added_illogical < new_illogical and illogical_combinations:
            # Wähle eine Kombination aus
            category, theme, complexity = illogical_combinations.pop(0)

            # Überprüfe bei Balance-Option, ob diese Kategorie noch Sätze benötigt
            if args.balance_categories and illogical_category_counts[category] >= illogical_per_category:
                continue

            # Generiere Prompt und Sätze
            illogical_prompt = generate_illogical_prompt(category, theme, complexity)
            num_to_generate = min(20, new_illogical - added_illogical)
            illogical_sentences = generate_sentences(illogical_prompt,
                                                   num_sentences=num_to_generate,
                                                   model=args.model)

            for sentence in illogical_sentences:
                # Prüfen, ob wir noch unlogische Sätze benötigen
                if added_illogical >= new_illogical:
                    break

                # Duplikatprüfung
                if sentence in existing_sentences:
                    continue

                # Validieren und hinzufügen
                label = validate_sentence(sentence, model=args.model)
                sentences.append(sentence)
                labels.append(label)
                existing_sentences.add(sentence)

                if label == 0:
                    added_illogical += 1
                    if pbar_illogic:
                        pbar_illogic.update(1)
                    if args.balance_categories:
                        illogical_category_counts[category] += 1
                else:
                    # Wenn wir noch logische Sätze brauchen, zählen wir ihn
                    if added_logical < new_logical:
                        added_logical += 1
                        if pbar_logic:
                            pbar_logic.update(1)
                    # Sonst überspringen wir ihn
                    else:
                        sentences.pop()
                        labels.pop()
                        existing_sentences.remove(sentence)

                time.sleep(args.delay)

                # Regelmäßiges Speichern
                if len(sentences) - last_save >= save_interval:
                    save_progress(sentences, labels, progress_file)
                    last_save = len(sentences)

        # Wenn keine Kombinationen mehr übrig sind, aber die Zielanzahl noch nicht erreicht ist,
        # generieren wir neue Kombinationen
        if not logical_combinations and added_logical < new_logical:
            print("\nGeneriere neue Kombinationen für logische Sätze...")
            logical_combinations = [(cat, theme, level)
                                   for cat in LOGICAL_CATEGORIES
                                   for theme in THEMES
                                   for level in COMPLEXITY_LEVELS]
            random.shuffle(logical_combinations)

        if not illogical_combinations and added_illogical < new_illogical:
            print("\nGeneriere neue Kombinationen für unlogische Sätze...")
            illogical_combinations = [(cat, theme, level)
                                     for cat in ILLOGICAL_CATEGORIES
                                     for theme in THEMES
                                     for level in COMPLEXITY_LEVELS]
            random.shuffle(illogical_combinations)

    # Fortschrittsbalken schließen
    if pbar_logic:
        pbar_logic.close()
    if pbar_illogic:
        pbar_illogic.close()

    # Endgültiges DataFrame erstellen und speichern
    df = pd.DataFrame({
        "sentence": sentences,
        "label": labels
    })

    # Statistiken über die Kategorieverteilung
    if args.balance_categories:
        print("\nVerteilung der logischen Kategorien:")
        for category, count in logical_category_counts.items():
            print(f"  {category}: {count} Sätze")

        print("\nVerteilung der unlogischen Kategorien:")
        for category, count in illogical_category_counts.items():
            print(f"  {category}: {count} Sätze")

    # Statistiken drucken
    print_stats(sentences, labels)

    # Speichern
    df.to_csv(args.output, index=False)
    print(f"Datensatz gespeichert als '{args.output}'.")

    # Einige Beispiele anzeigen
    print_examples(df, args.examples)

    # Aufräumen
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"Fortschrittsdatei {progress_file} gelöscht.")

if __name__ == "__main__":
    main()
