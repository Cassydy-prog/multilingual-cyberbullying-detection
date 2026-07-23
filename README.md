# Multilingual Cyberbullying Detection
---

## 1. Project Objective

The objective of this project is to detect cyberbullying in texts written in different languages.

The task is formulated as a binary classification problem:

- `0`: non-cyberbullying;
- `1`: cyberbullying or abusive content.

We compare two approaches:

1. **TF-IDF + Linear SVM**
2. **XLM-RoBERTa fine-tuned for binary classification**

---

## 2. Research Question

> Can a multilingual Transformer model detect cyberbullying more effectively across several languages than a traditional TF-IDF and Linear SVM model?

We also want to know whether the models perform equally well across languages or whether some languages are more difficult to classify.

---

## 3. Dataset

The dataset contains social media texts written in several languages, including English, French, Arabic, and Turkish.
We cleaned out the dataset so we could have two classes 

#### Class 0: Non-cyberbullying

- `neutral`
- `nonbully`

#### Class 1: Cyberbullying or abusive content

- `bully`
- `hate`
- `threat`
- `troll`
- `offense`
- `defame`
- `race`
- `religion`
- `sex`
- `sexual`

The categories `fake` and `political` are considered ambiguous because they do not always represent cyberbullying. They should be excluded or analyzed separately.

---

## 4. Data Preparation

The raw dataset is kept unchanged.

A single preprocessing pipeline is used to create three processed datasets:

- training set;
- validation set;
- test set.

The data preparation steps are:

```text
Raw dataset
    ↓
Remove missing texts and invalid labels
    ↓
Remove duplicates
    ↓
Extract language and category
    ↓
Create the binary label
    ↓
Clean the text while preserving multilingual characters
    ↓
Split into train, validation, and test sets
