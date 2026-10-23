# 📝 Transformer Text Summarization

A proof-of-concept abstractive text summarization project built from scratch using **TensorFlow** and the **Transformer Encoder–Decoder architecture**.

The project demonstrates the implementation of Transformers, attention mechanisms, tokenization, training, and text generation using a small custom dataset.

## 🛠️ Technologies Used

- 🐍 Python
- 🧠 TensorFlow
- 🔤 SentencePiece
- 📊 NumPy, Pandas, Scikit-learn, NLTK

## 🏗️ Model Architecture

- Encoder Layers: 2
- Decoder Layers: 2
- Model Dimension: 128
- Attention Heads: 4
- Feed-Forward Dimension: 512

## 📂 Dataset

- **File:** `transformer_summarization_dataset.csv`
- **Input:** `paragraph`
- **Target:** `summary`
- **Size:** Approximately 960 examples

## ⚙️ How to Run

Install dependencies:

```bash
pip install tensorflow numpy pandas sentencepiece scikit-learn nltk
```

Run the project:

```bash
python summarization.py
```

Enter a paragraph and type `END` on a new line to generate its summary. Type `EXIT` to stop.

## 📊 Evaluation

The model is evaluated using BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores.

## 🎯 Project Purpose

This is an experimental proof of concept demonstrating the ability to build a Transformer from scratch. A larger, more diverse, and higher-quality dataset would be needed to improve summary accuracy and generalization.

## 👨‍💻 Author

**Dhruv Surana**