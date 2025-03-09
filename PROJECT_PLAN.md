# Project Plan: Custom Embedding + Transformer Model for Logical Sentence Recognition

## Project Goal
Implement a small embedding and transformer model from scratch in PyTorch that evaluates simple sentences based on whether they are logically meaningful or not.

## Technologies
* Python (3.10+)
* PyTorch
* NumPy
* Optional: Docker (containerization, reproducible environment)

## Roadmap

### Step 1: Project Preparation & Setup
- [1.1] Create a clean Python environment (`venv`, `conda`, or Docker container)
- [1.2] Install required packages:
  - `torch`, `torchvision`
  - `numpy`
  - `pandas` (optional, for convenient data handling)

### Step 2: Create and Prepare Dataset
- [2.1] Create a small example dataset:
  - About 50-100 short sentences (e.g., 50 meaningful, 50 meaningless)
  - Clearly defined format (e.g., CSV with columns: sentence, label)
- [2.2] Tokenize sentences:
  - Create a simple tokenizer (whitespace tokenizer or simplest tokenization)
- [2.3] Generate vocabulary:
  - Create a small dictionary (index ↔ word mapping)
- [2.4] Dataset split:
  - Training set (~80%), test set (~20%)

### Step 3: Create Custom Embedding Model
- [3.1] Create an embedding layer (PyTorch Embedding Layer)
  - Set parameters (e.g., `embedding_dimension = 32`)
- [3.2] Validate embedding output:
  - Test if words are correctly converted to embeddings (shape: `[batch_size, seq_len, embedding_dim]`)

### Step 4: Implement Transformer Model
- [4.1] Implement a transformer encoder layer (self-attention):
  - Multi-head attention layer
  - Position-wise feedforward network
  - Layer normalization and residual connections
- [4.2] Combine layers into transformer encoder:
  - Create a transformer class (`nn.Module`) with multiple encoder layers
- [4.3] Output layer for classification:
  - Pooling (e.g., averaging or [CLS]-token)
  - Fully-connected layer (`Linear`) for binary classification (logical/illogical)
- [4.4] Validate model output:
  - Ensure the output has the correct dimension and produces values (`[batch_size, num_classes]`)

### Step 5: Train the Model
- [5.1] Define loss function & optimizer:
  - `CrossEntropyLoss`
  - `Adam` optimizer
- [5.2] Implement training loop:
  - Forward pass
  - Backpropagation
  - Log training loss and accuracy
- [5.3] Evaluate on test data:
  - Calculate accuracy and loss

### Step 6: Analyze & Interpret Results
- [6.1] Examine results:
  - Analyze errors and correct classifications
- [6.2] Visualize results:
  - Confusion matrix
  - Progress of loss & accuracy over epochs

### Step 7: Optional - Deployment & Extensions
- [7.1] Containerization with Docker (create Dockerfile)
- [7.2] Create a simple REST endpoint with FastAPI to use the model
- [7.3] Document extension possibilities:
  - Expand the dataset
  - Tune the transformer architecture
  - Incorporate position encodings, additional regularization techniques, etc.

## Recommended Timeline

| Step | Estimated Effort | Recommended Schedule |
|------|-----------------|---------------------|
| Step 1 | ~30 min | Day 1 |
| Step 2 | ~2 hours | Day 1 |
| Step 3 | ~1 hour | Day 2 |
| Step 4 | ~3 hours | Day 2-3 |
| Step 5 | ~2 hours | Day 3 |
| Step 6 | ~1 hour | Day 4 |
| Step 7 | Optional | Day 5 |

This schedule is generous and allows for a good understanding of each step.
