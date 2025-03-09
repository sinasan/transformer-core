import openai 
import pandas as pd
import time
import os
from dotenv import load_dotenv
load_dotenv("../.env")

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

# Sätze generieren
def generate_sentences(prompt, num_sentences=50):
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    sentences = response.choices[0].message.content.strip().split("\n")
    sentences = [s.strip("-•1234567890. \t") for s in sentences if len(s.strip()) > 0]
    return sentences[:num_sentences]

# Satzvalidierung (logisch oder nicht)
def validate_sentence(sentence):
    prompt = f'Ist der folgende Satz logisch sinnvoll? Antworte ausschließlich mit "ja" oder "nein":\nSatz: "{sentence}"'
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    result = response.choices[0].message.content.strip().lower()
    return 1 if result == "ja" else 0

def main():
    NUM_SENTENCES_PER_CATEGORY = 50

    positive_prompt = "Generiere 50 logische, sinnvolle deutsche Sätze über alltägliche Dinge."
    negative_prompt = "Generiere 50 grammatikalisch korrekte, aber logisch unsinnige deutsche Sätze."

    print("Generiere positive Beispiele...")
    positive_sentences = generate_sentences(positive_prompt, NUM_SENTENCES_PER_CATEGORY)

    print("Generiere negative Beispiele...")
    negative_sentences = generate_sentences(negative_prompt, NUM_SENTENCES_PER_CATEGORY)

    sentences, labels = [], []

    print("Validiere positive Beispiele...")
    for sentence in positive_sentences:
        label = validate_sentence(sentence)
        sentences.append(sentence)
        labels.append(label)
        time.sleep(0.3)

    print("Validiere negative Beispiele...")
    for sentence in negative_sentences:
        label = validate_sentence(sentence)
        sentences.append(sentence)
        labels.append(label)
        time.sleep(0.3)

    df = pd.DataFrame({
        "sentence": sentences,
        "label": labels
    })

    df.to_csv("../data/sentences.csv", index=False)
    print("Datensatz gespeichert als 'transformer_sentences.csv'.")

if __name__ == "__main__":
    main()

