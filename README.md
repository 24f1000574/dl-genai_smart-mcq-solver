# Smart MCQ Solver Challenge

A deep learning project developed for the **Smart MCQ Solver Challenge**, where the goal is to automatically predict the correct answer for multiple-choice questions using Natural Language Processing (NLP) and Transformer-based models.

This project explores multiple approaches, starting from traditional NLP techniques and progressing to modern Transformer architectures. All experiments were tracked using **Weights & Biases (W&B)**, and the final models were evaluated using the **MAP@3** metric.

---

## Project Overview

The objective of this project is to build a system capable of understanding a question along with its five answer choices and selecting the most appropriate answer.

To better understand the strengths and limitations of different approaches, I implemented and compared several models, ranging from classical methods to state-of-the-art Transformer models.

---

## Models Implemented

### 1. ANN (Built from Scratch)

A custom neural network developed using PyTorch.

Features:

- Custom vocabulary creation
- Embedding layer
- Mean pooling
- Fully connected matching network
- GELU activation
- Layer Normalization
- Dropout for regularization

This model serves as a strong baseline without relying on pretrained language models.

---

### 2. DistilBERT

A lightweight Transformer model fine-tuned for the multiple-choice classification task.

Advantages:

- Faster training
- Lower memory usage
- Strong contextual understanding
- Excellent validation performance

---

### 3. DeBERTa-v3

A larger Transformer model with disentangled attention.

Advantages:

- Better contextual representation
- Improved semantic understanding
- High accuracy on complex questions

---

## Project Workflow

```
Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Duplicate Removal
      │
      ▼
Train / Validation Split
      │
      ▼
Feature Engineering
      │
      ▼
TF-IDF & Word2Vec Baselines
      │
      ▼
Transformer Embeddings
      │
      ▼
Context Retrieval (RAG)
      │
      ▼
Model Training
      │
      ▼
Validation (MAP@3)
      │
      ▼
Model Comparison
      │
      ▼
Ensembling
      │
      ▼
Final Submission
```

---

## Techniques Used

- Data preprocessing
- Duplicate question removal
- TF-IDF baseline
- Word2Vec embeddings
- Transformer embeddings
- Retrieval-Augmented Generation (RAG)
- Fine-tuning pretrained models
- Gradient Accumulation
- Mixed Precision Training
- Cosine Learning Rate Scheduler
- Gradient Clipping
- Model Ensembling

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- Gensim
- Weights & Biases (W&B)
- Kaggle Notebook

---

## Dataset

The competition dataset contains:

- Training questions
- Five answer choices (A–E)
- Correct answer labels

Each sample consists of one question and five possible answers.

---

## Evaluation Metric

The project is evaluated using **MAP@3 (Mean Average Precision @ 3)**.

Instead of considering only the top prediction, MAP@3 rewards the model when the correct answer appears within the top three ranked predictions.

---

## Experiment Tracking

All training experiments were logged using **Weights & Biases (W&B)**.

Tracked metrics include:

- Training Loss
- Validation MAP@3
- Learning Rate
- Model Configuration
- Training History

This made it easier to compare models and reproduce results.

---

## Results

| Model | Validation MAP@3 |
|--------|-----------------:|
| ANN | 0.9768 |
| DeBERTa-v3 | 0.9029 |
| DistilBERT | 0.9974 |

> DistilBERT achieved the best validation performance among the implemented models.

---

## Repository Structure

```
├── notebooks/
│   ├── training.ipynb
│   └── inference.ipynb
│
├── models/
│   ├── ann/
│   ├── deberta/
│   └── distilbert/
│
├── reports/
│
├── figures/
│
├── requirements.txt
│
└── README.md
```

---

## Key Learnings

During this project I gained practical experience with:

- Building deep learning models from scratch
- Fine-tuning pretrained Transformer models
- NLP preprocessing techniques
- Multiple-choice question modeling
- Experiment tracking using W&B
- Hyperparameter tuning
- Retrieval-Augmented Generation (RAG)
- Model evaluation using MAP@3
- Comparing classical NLP methods with modern Transformers

---

## Future Improvements

Some possible extensions for this project include:

- Cross-validation
- Larger Transformer models
- Better retrieval strategies
- Knowledge distillation
- Model quantization
- Advanced ensemble techniques

---

## Acknowledgements

- IIT Madras BS Degree Program
- Kaggle
- Hugging Face
- PyTorch
- Weights & Biases

---

## Author

**Sagnik Sen**

BS in Data Science and Applications  
Indian Institute of Technology Madras

