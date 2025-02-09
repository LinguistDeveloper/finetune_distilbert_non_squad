# Fine-tune DistilBERT on non-SQuAD data

This project demonstrates the process of fine-tuning a DistilBERT-based model (from Hugging Face's Transformers library) for a custom Question Answering (QA) task. The project adapts a sample of the SQuAD dataset (with answer positions manually removed) and inserts answer positions programmatically from the answers; it also uses the TriviaQA dataset to simulate the more complex structure of many (non-SQuAD) datasets, and extracts the required data. If you are new to fine-tuning DistilBERT for QA, SQuAD is the easiest dataset to start with because it has a clean structure, pre-existing models, and official support in libraries like Hugging Face. However, adapting other datasets (like TriviaQA) can still be done with extra preprocessing steps, as is demonstrated here.

## Project Overview

### Objectives:

- Fine-tune a DistilBERT model for Question Answering.

- Adapt a sample of SQuAD to stand in for content with multiple questions and answers about a context passage minus the pre-inserted positions.

- Adapt a sample of TriviaQA to exemplify handling of content from non-SQuAD datasets with a more complex structure.

### Datasets:

- Sources: SQuAD dataset sample, modified by removing answer positions, and TriviaQA dataset.

- Format: The datasets consist of context paragraphs, questions, and answers with a structure that differs from that of SQuAD in complexity and/or nomenclature, and with answer positions removed and then reinserted programmatically prior to model training.

### Custom Modifications:

TriviaQA Structure: TriviaQA's complex structure has multiple QA sets per context and varying answer lengths, requiring a custom approach for data preprocessing.

Answer Positioning: Answer positions from SQuAD were removed initially, then reinserted to reflect TriviaQA's structure.

---
## Methodology

### Steps:

#### Dataset Preprocessing:

- Removed answer positions from the SQuAD dataset.

- Loaded the SQuAD sample dataset.

- Wrote a script to reintegrate the answer positions back into the context data.

- Built a custom Dataset object with SQuAD labels for data extracted from the TriviaQA node structure.

#### Tokenization:

- Tokenized the context, question, and answer texts using DistilBERT's tokenizer.

- Applied padding and truncation to ensure uniform sequence lengths.

#### Model Definition:

- Used the pre-trained DistilBERT model.

- Adapted the model for Question Answering by adding a QA head.

#### Fine-Tuning:

- Fine-tuned the model using the following hyperparameters:

  - Optimizer: AdamW with a learning rate of 2e-5.

  - Batch size: 16.

  - Epochs: 3.

  - Weight Decay: 0.01.

- Used cross-entropy loss for optimization.

- Evaluated performance on a held-out validation set.

#### Evaluation:

- Assessed model performance using:

  - Exact Match Score (EM)

  - F1 Score

- Compared results with other QA models to determine the effectiveness of the custom dataset.
