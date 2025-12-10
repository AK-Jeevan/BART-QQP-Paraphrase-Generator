# BART QQP Paraphrase Generator

This repository contains a complete implementation for fine-tuning Facebook's **BART (Bidirectional and Auto-Regressive Transformer)** model on the **Quora Question Pairs (QQP)** dataset to perform **paraphrase generation**.

The model learns to generate alternative rewordings of a question by mapping:
Question 1 → Question 2 (paraphrase).

---

## 🚀 Project Overview

- **Task**: Paraphrase Generation (Sequence-to-Sequence)
- **Model**: `facebook/bart-base`
- **Dataset**: Quora Question Pairs (QQP)
- **Frameworks**: Hugging Face Transformers & Datasets
- **Training Method**: Supervised fine-tuning with teacher forcing

---

## 🧠 What This Model Learns

The model is trained only on **duplicate question pairs** (`is_duplicate = 1`) so that it learns:
- Semantic preservation
- Lexical variation
- Question rephrasing

---

## 📦 Installation

pip install transformers datasets torch accelerate

## 📁 Dataset Loading

The dataset is automatically downloaded from Hugging Face:

from datasets import load_dataset
quora = load_dataset("quora")


Duplicate-only filtering is applied before training.

## ✅ Evaluation

Evaluation is automatically performed after training using the validation split of QQP.

## ✨ Inference

input_text = "How can I learn Python quickly?"

## ✅ Model Output:

What is the fastest way to learn Python?

## 📊 Key Features

Automatic duplicate question filtering

Proper label masking using -100

Dynamic padding with DataCollatorForSeq2Seq

Beam search for high-quality text generation

Mixed precision training for better GPU performance

## 🧪 Use Cases

Paraphrase generation for NLP datasets

Question augmentation for QA systems

Educational NLP projects

Data augmentation for chatbots

## 📌 Future Improvements

Add BLEU / ROUGE evaluation

Add Gradio / Streamlit demo

Export trained model to Hugging Face Hub

Add dataset size control for low-resource GPUs

## 📜 License

This project is released under the MIT License.

Feel free to modify, experiment, and improve this project!
