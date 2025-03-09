# transformer-core

We are building a simple but complete transformer-based model that can determine whether a given sentence makes logical sense or not.

Example inputs:
* "Elephants like to eat apples" → Makes sense
* "Elephants eat schnitzel" → Less sensible

## Contents

1. **Creating a Dataset**  
   A small amount of training data (sentences with labels sensible/not sensible).

2. **Creating Custom Embeddings**  
   Building a simple word embedding (e.g., based on a small, randomly initialized lookup table).

3. **Building a Simple Transformer Encoder**  
   A small transformer (Self-Attention + Feed-Forward Layers) to learn sentence logic.

4. **Training and Inference**  
   Training and evaluating on example data.

## Technologies & Frameworks

* Python with PyTorch (transparent, easy to understand)
* No external embeddings like GloVe or FastText (we deliberately create our own small embedding).
