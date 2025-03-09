# transformer-core

Wir bauen ein einfaches, aber vollständiges Transformer-basiertes Modell, das entscheiden kann, ob ein gegebener Satz logisch Sinn ergibt oder nicht.

Beispielhafte Eingabe:
* „Elefanten essen gerne Äpfel" → Sinnvoll
* „Elefanten essen Schnitzel" → Weniger sinnvoll

## Enthält

1. **Erstellung eines Datensatzes**  
   Eine kleine Menge an Trainingsdaten (Sätze mit Labels sinnvoll/nicht sinnvoll).

2. **Eigenes Embedding erstellen**  
   Aufbau eines simplen Wort-Embeddings (z.B. auf Basis eines kleinen, zufällig initialisierten Lookup-Tables).

3. **Aufbau eines einfachen Transformer-Encoders**  
   Ein kleiner Transformer (Self-Attention + Feed-Forward Layers), um die Satzlogik zu lernen.

4. **Training und Inferenz**  
   Trainieren und evaluieren auf Beispiel-Daten.
