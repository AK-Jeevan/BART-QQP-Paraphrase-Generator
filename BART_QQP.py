# ================================================================================
# BART Paraphrase Generation on Quora Question Pairs (QQP)
# ================================================================================
# This script fine-tunes Facebook's BART model to generate paraphrases of questions.
# It uses the Quora dataset and trains BART to learn question paraphrasing by mapping
# question Q1 -> question Q2 (paraphrase).
#
# Task: Paraphrase generation (seq2seq)
# Model: BART (Bidirectional Auto-Regressive Transformer)
# Dataset: Quora Question Pairs (QQP) - only duplicate pairs
# Goal: Train BART to generate alternative phrasings of the same question
#
# Key Steps:
#   1. Load the Quora dataset from HuggingFace
#   2. Filter to keep only duplicate question pairs (is_duplicate == 1)
#   3. Tokenize Q1 as input and Q2 as target paraphrase
#   4. Fine-tune BART with early stopping and evaluation
#   5. Generate new paraphrases for unseen questions
#
# ================================================================================

# Load libraries
from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
import numpy as np

# Load the Quora dataset
quora = load_dataset("quora")

# Keep only duplicate question pairs (true paraphrases)
def keep_duplicates(example):
    return example["is_duplicate"] == 1

train_dups = quora["train"].filter(keep_duplicates)
val_dups = quora["validation"].filter(keep_duplicates)

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Preprocessing function: map q1 -> input, q2 -> target paraphrase
max_input_len = 128
max_target_len = 128

def preprocess(example):
    q1, q2 = example["questions"]
    model_inputs = tokenizer(
        q1,
        max_length=max_input_len,
        truncation=True,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            q2,
            max_length=max_target_len,
            truncation=True,
            padding="max_length"
        )

    # Replace padding token IDs in labels with -100 so loss ignores them
    label_ids = labels["input_ids"]
    label_ids = [token_id if token_id != tokenizer.pad_token_id else -100 for token_id in label_ids]

    model_inputs["labels"] = label_ids
    return model_inputs

# Apply preprocessing
train_data = train_dups.map(preprocess, remove_columns=train_dups.column_names)
val_data = val_dups.map(preprocess, remove_columns=val_dups.column_names)

# Data collator: handles dynamic padding for seq2seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training configuration
training_args = TrainingArguments(
    output_dir="./bart-paraphrase",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_steps=100,
    fp16=True
)


# Trainer: ties dataset, model, training args, and collator together
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(eval_results)

# Generate paraphrases for new text
input_text = "How can I learn Python quickly?"
inputs = tokenizer([input_text], return_tensors="pt")
generated_ids = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_beams=5,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
    early_stopping=True
)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
