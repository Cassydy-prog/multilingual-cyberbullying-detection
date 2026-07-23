# Multilingual Cyberbullying Detection

## Project Overview

This Natural Language Processing project aims to develop and evaluate several models capable of automatically detecting cyberbullying in texts written in different languages.

The task is formulated as a **binary classification** problem:

- `0`: no cyberbullying;
- `1`: cyberbullying or abusive content.

The project compares a traditional NLP approach with a multilingual Transformer-based approach:

- **TF-IDF + Linear SVM**;
- **XLM-RoBERTa fine-tuned for binary classification**.

---

## Research Question

Social media platforms make communication easier, but they are also used to spread insults, threats, hate speech, and different forms of online harassment.

Automatically detecting this type of content is especially challenging in a multilingual setting. Languages do not all have the same amount of training data, and expressions of harassment may vary depending on the language, culture, and context.

The main research question of this project is:

> To what extent does a multilingual Transformer model such as XLM-RoBERTa outperform a traditional TF-IDF and Linear SVM approach for cyberbullying detection?

The project will also investigate whether model performance remains consistent across the different languages represented in the dataset.

---

## Objectives

The main objectives of the project are:

1. Explore and clean a multilingual cyberbullying dataset.
2. Convert the original categories into binary labels.
3. Study the class distribution within each language.
4. Build a baseline using TF-IDF and Linear SVM.
5. Fine-tune XLM-RoBERTa for binary classification.
6. Compare the performance of the selected models.
7. Evaluate the results globally and separately for each language.
8. Analyze model errors and limitations.

---

## Dataset

The project uses the **[exact dataset name]** dataset, which contains texts written in several languages.

Each observation includes the following fields:

| Column | Description |
|---|---|
| `text` | Text content of the post |
| `label_original` | Original label containing the language and category |
| `language` | Language of the text |
| `category` | Original content category |
| `label_binary` | Label used for binary classification |

### Languages Included

The dataset contains several languages, including:

- English;
- French;
- Arabic;
- Hindi;
- Bengali;
- [other languages to be added].

### Label Transformation

The original categories are converted into two classes.

#### Class 0: No cyberbullying

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
- [other selected categories]

Some ambiguous categories, such as `fake` or `political`, may be excluded from training when they do not clearly correspond to cyberbullying.

---

## Project Pipeline

The general project pipeline is organized as follows:

```text
Raw dataset
    ↓
Data exploration
    ↓
Text cleaning and duplicate removal
    ↓
Language and category extraction
    ↓
Binary label transformation
    ↓
Class distribution analysis by language
    ↓
Train / validation / test split
    ↓
Model training
    ↓
Global evaluation
    ↓
Evaluation by language
    ↓
Error analysis
