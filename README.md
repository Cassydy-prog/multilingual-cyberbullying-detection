# Multilingual Cyberbullying Detection

## Simple Presentation Guide

This document provides the team with one clear story to follow during the presentation.

The goal is to avoid presenting disconnected code fragments and to explain the project as one complete NLP pipeline.

---

## 1. Project Objective

The objective of this project is to detect cyberbullying in texts written in different languages.

The task is formulated as a binary classification problem:

- `0`: non-cyberbullying
- `1`: cyberbullying or abusive content

We compare two approaches:

- **TF-IDF + Linear SVM**
- **XLM-RoBERTa fine-tuned for binary classification**

---

## 2. Research Question

> Can a multilingual Transformer model detect cyberbullying more effectively across several languages than a traditional TF-IDF and Linear SVM model?

We also want to determine whether the models perform equally well across languages or whether some languages are more difficult to classify.

---

## 3. Dataset

The dataset contains social media texts written in several languages, including:

- English
- French
- Arabic
- Turkish

The original labels contain two types of information:

- the language
- the content category

### Binary Label Transformation

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

The categories `fake` and `political` are considered ambiguous because they do not always represent cyberbullying.

They should therefore be excluded or analyzed separately.

---

## 4. Data Preparation

The raw dataset is kept unchanged.

A single preprocessing pipeline is used to create three processed datasets:

- training set
- validation set
- test set

### Data Preparation Pipeline

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
```

The split should preserve:

- the proportion of the two classes
- the distribution of languages

The same training, validation, and test files must be used by every team member.

---

## 5. Multilingual Text Cleaning

The cleaning process removes:

- URLs
- user mentions
- unnecessary spaces

It preserves characters from all writing systems, including:

- Arabic characters
- Hindi characters
- Bengali characters
- accented French characters

The original text is also kept in the dataset.

For **TF-IDF**, a cleaned text column is used.

For **XLM-RoBERTa**, cleaning remains limited because punctuation, emojis, hashtags, and capitalization may contain useful information.

---

## 6. Model 1: TF-IDF + Linear SVM

TF-IDF converts each text into a numerical vector based on the importance of its words or character sequences.

Linear SVM then learns a decision boundary between:

- class `0`: non-cyberbullying
- class `1`: cyberbullying

### Why This Model?

- Fast to train
- Computationally efficient
- Strong baseline for text classification
- Suitable for high-dimensional sparse data
- Easy to compare with a Transformer model

The TF-IDF vectorizer is fitted only on the training set.

The validation and test sets are transformed using the vocabulary learned from the training set.

```text
Training texts
    ↓
TF-IDF fit and transformation
    ↓
Linear SVM training
    ↓
Binary prediction
```

---

## 7. Model 2: XLM-RoBERTa

XLM-RoBERTa is a multilingual Transformer model.

The selected pretrained model is:

```text
FacebookAI/xlm-roberta-base
```

It is fine-tuned end-to-end for binary classification.

### Why This Model?

- Designed for multilingual text
- Uses contextual word representations
- Can process several languages with one model
- May transfer knowledge between languages
- May better understand indirect or context-dependent attacks

Unlike TF-IDF, XLM-RoBERTa does not only identify statistically important words.

It represents words according to their context in the sentence.

```text
Original text
    ↓
XLM-RoBERTa tokenizer
    ↓
Contextual representations
    ↓
Classification layer
    ↓
Prediction: class 0 or class 1
```

---

## 8. Training and Evaluation Workflow

```text
Training set
    ↓
Train the model

Validation set
    ↓
Compare settings and detect overfitting

Test set
    ↓
Final evaluation
```

The training set is used to fit the models.

The validation set is used to:

- compare model configurations
- select hyperparameters
- detect overfitting

The test set is used only for the final evaluation.

It must not be used to select model settings.

### Saving the Trained Models

After training, the model must be saved so that another team member can evaluate it without running the training again.

#### Files to Save for TF-IDF + Linear SVM

- the fitted TF-IDF vectorizer
- the trained Linear SVM model
- or the complete fitted pipeline

#### Files to Save for XLM-RoBERTa

- the fine-tuned model
- the tokenizer
- the label mapping
- the training configuration

---

## 9. Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion matrix

### Accuracy

Accuracy represents the percentage of all examples that were correctly classified.

### Precision

Precision measures how many messages predicted as cyberbullying were actually cyberbullying.

### Recall

Recall measures how many real cyberbullying messages were correctly detected.

### F1-score

The F1-score combines precision and recall into one score.

### Macro F1-score

Macro F1 calculates the F1-score for each class and then gives both classes the same importance.

The main metric in this project is **Macro F1-score** because the dataset is not perfectly balanced.

### Important Errors

- **False positive:** a normal message is incorrectly classified as cyberbullying.
- **False negative:** a cyberbullying message is not detected.

False negatives are especially important because they represent harmful messages that the system fails to detect.

---

## 10. Current Model Results

For a valid comparison, both models must be evaluated on the same test set.

The current test set contains:

- 603 non-cyberbullying examples
- 1,026 cyberbullying examples
- 1,629 examples in total

---

### 10.1 TF-IDF + Linear SVM Results

#### Global Results

| Metric | Score |
|---|---:|
| Test accuracy | 0.8275 |
| Precision for cyberbullying | 0.8729 |
| Recall for cyberbullying | 0.8499 |
| F1-score for cyberbullying | 0.8612 |
| Macro F1-score | 0.8167 |

#### Results by Class

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Non-cyberbullying | 0.7556 | 0.7894 | 0.7721 | 603 |
| Cyberbullying | 0.8729 | 0.8499 | 0.8612 | 1,026 |

#### Simple Interpretation

The TF-IDF and Linear SVM model correctly classified about **82.75%** of the test examples.

It performed better on the cyberbullying class than on the non-cyberbullying class.

The F1-score was:

- **86.12%** for cyberbullying
- **77.21%** for non-cyberbullying

This difference may partly be explained by class imbalance because the cyberbullying class contains more examples than the non-cyberbullying class.

However, the difference may also come from:

- ambiguous labels
- normal messages containing aggressive words
- messages requiring more contextual understanding
- differences between languages
- limitations of the TF-IDF representation

The TF-IDF and Linear SVM model is therefore a strong baseline, but there is still room for improvement.

---

### 10.2 XLM-RoBERTa Results

#### Global Results

| Metric | Score |
|---|---:|
| Test accuracy | 0.8660 |
| Macro precision | 0.8563 |
| Macro recall | 0.8559 |
| Macro F1-score | 0.8561 |
| Weighted F1-score | 0.8660 |

#### Cyberbullying Class Results

| Metric | Score |
|---|---:|
| Cyberbullying precision | 0.8930 |
| Cyberbullying recall | 0.8947 |
| Cyberbullying F1-score | 0.8939 |

#### Simple Interpretation

The XLM-RoBERTa model correctly classified about **86.60%** of the test examples.

Its Macro F1-score of **85.61%** shows that the model achieved relatively balanced performance across the two classes.

For the cyberbullying class, the model obtained:

- **89.30% precision**
- **89.47% recall**
- **89.39% F1-score**

The high precision means that most messages predicted as cyberbullying were actually cyberbullying.

The high recall means that the model detected most of the real cyberbullying messages in the test set.

The precision and recall scores are also very close, which indicates a good balance between avoiding false alerts and detecting harmful content.

---

### 10.3 Comparison Between the Models

| Metric | TF-IDF + Linear SVM | XLM-RoBERTa | Improvement |
|---|---:|---:|---:|
| Test accuracy | 0.8275 | 0.8660 | +0.0385 |
| Macro F1-score | 0.8167 | 0.8561 | +0.0394 |
| Cyberbullying precision | 0.8729 | 0.8930 | +0.0201 |
| Cyberbullying recall | 0.8499 | 0.8947 | +0.0448 |
| Cyberbullying F1-score | 0.8612 | 0.8939 | +0.0327 |

#### Comparison Interpretation

XLM-RoBERTa outperformed TF-IDF + Linear SVM on all the main evaluation metrics.

Its test accuracy increased from **82.75% to 86.60%**, representing an improvement of **3.85 percentage points**.

The Macro F1-score increased from **81.67% to 85.61%**, showing that XLM-RoBERTa achieved better overall performance across both classes.

The largest improvement was observed in cyberbullying recall, which increased from **84.99% to 89.47%**.

This means that XLM-RoBERTa missed fewer cyberbullying messages than TF-IDF + Linear SVM.

These results suggest that contextual and multilingual representations help the model better understand abusive content than a traditional statistical text representation.

However, XLM-RoBERTa requires more computational resources and more training time than TF-IDF + Linear SVM.

## 11. Multilingual Evaluation

A global score is not enough for this project.

The final comparison should include results for each language.

| Model | Language | Accuracy | Precision | Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| TF-IDF + Linear SVM | English | To be added | To be added | To be added | To be added |
| TF-IDF + Linear SVM | French | To be added | To be added | To be added | To be added |
| TF-IDF + Linear SVM | Arabic | To be added | To be added | To be added | To be added |
| TF-IDF + Linear SVM | Hindi | To be added | To be added | To be added | To be added |
| TF-IDF + Linear SVM | Bengali | To be added | To be added | To be added | To be added |
| XLM-RoBERTa | English | To be added | To be added | To be added | To be added |
| XLM-RoBERTa | French | To be added | To be added | To be added | To be added |
| XLM-RoBERTa | Arabic | To be added | To be added | To be added | To be added |
| XLM-RoBERTa | Hindi | To be added | To be added | To be added | To be added |
| XLM-RoBERTa | Bengali | To be added | To be added | To be added | To be added |

This analysis will show whether a model performs well globally but poorly on a specific language.

It will also help determine whether XLM-RoBERTa transfers knowledge more effectively between languages.

---

## 12. Expected Model Comparison

| Criterion | TF-IDF + Linear SVM | XLM-RoBERTa |
|---|---|---|
| Model family | Classical machine learning | Transformer |
| Text representation | Statistical word importance | Contextual representation |
| Training time | Low | High |
| Computational cost | Low | High |
| GPU required | No | Recommended |
| Context understanding | Limited | Stronger |
| Native multilingual capability | No | Yes |
| Cross-language transfer | Very limited | Possible |
| Interpretability | Easier | More difficult |
| Management of explicit insults | Strong | Strong |
| Management of indirect attacks | Limited | Potentially stronger |

Our hypothesis is that XLM-RoBERTa may perform better on multilingual and context-dependent examples.

However, TF-IDF + Linear SVM may remain competitive on explicit abusive messages while requiring fewer computational resources.

---

## 13. Limitations

The project has several limitations:

- The original categories are not identical across all languages.
- Some categories represent abusive content rather than cyberbullying in the strict sense.
- A single message does not show whether harassment is repeated.
- Some languages may contain fewer examples than others.
- The two binary classes are not perfectly balanced.
- The model may learn language patterns instead of cyberbullying patterns.
- Some labels may be ambiguous.
- Normal messages may contain words associated with abusive content.
- Cultural differences may affect how cyberbullying is expressed.
- TF-IDF has limited contextual understanding.
- XLM-RoBERTa requires more training time and computing resources.
- The model results may depend on the quality of the original annotations.

For this reason, the project can be described more precisely as:

> **Multilingual detection of cyberbullying and abusive online content.**

---

## 14. Simple Presentation Order for Five People

### Speaker 1 — Context and Objective

Present:

- the cyberbullying problem
- the project objective
- the binary classification task
- the research question
- the two selected approaches

### Speaker 2 — Dataset and Labels

Present:

- the source and content of the dataset
- the languages
- the original labels
- the separation between language and category
- the binary label transformation
- the class distribution
- the ambiguous categories

### Speaker 3 — Preprocessing and Pipeline

Present:

- the raw data
- missing-value management
- duplicate removal
- multilingual text cleaning
- the preservation of Arabic, Hindi, Bengali, and accented characters
- train, validation, and test creation
- the complete pipeline diagram

### Speaker 4 — Models

Present:

- TF-IDF
- Linear SVM
- XLM-RoBERTa
- why both models were selected
- the main differences between the two approaches
- model training and saving

### Speaker 5 — Metrics, Results, and Conclusion

Present:

- the evaluation metrics
- the current TF-IDF results
- the results by class
- the effect of class imbalance
- the future XLM-RoBERTa comparison
- multilingual evaluation
- limitations
- the conclusion

---

## 15. Presentation Guidelines

The presentation should not contain many disconnected code cells.

The team should mainly show:

1. one slide explaining the problem and research question
2. one table describing the dataset
3. one diagram of the complete preprocessing pipeline
4. one slide explaining TF-IDF + Linear SVM
5. one slide explaining XLM-RoBERTa
6. one comparison table between the two models
7. one results table
8. one confusion matrix
9. one multilingual results table
10. one final conclusion

Code should only be shown when it helps explain an important technical decision.

The audience does not need to see every implementation detail.

---

## 16. Current Project Status

- [x] Define the project objective
- [x] Select the multilingual dataset
- [x] Separate the language and category labels
- [x] Transform the original labels into binary classes
- [x] Identify ambiguous categories
- [x] Create multilingual text cleaning
- [x] Preserve non-Latin writing systems
- [x] Build the TF-IDF + Linear SVM baseline
- [x] Evaluate the baseline on the test set
- [x] Interpret the TF-IDF + Linear SVM results
- [ ] Finalize the common training, validation, and test files
- [ ] Push the latest notebooks and files to GitHub
- [ ] Save and share the trained TF-IDF model
- [ ] Finalize the XLM-RoBERTa fine-tuning
- [ ] Save and share the fine-tuned XLM-RoBERTa model
- [ ] Evaluate both models on the three datasets
- [ ] Evaluate both models by language
- [ ] Compare TF-IDF + Linear SVM with XLM-RoBERTa
- [ ] Conduct an error analysis
- [ ] Complete the final presentation

---

## 17. Repository Organization

```text
multilingual-cyberbullying-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── multilingual_cyberbullying.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_tfidf_linear_svm.ipynb
│   ├── 04_xlm_roberta.ipynb
│   └── 05_model_evaluation.ipynb
│
├── models/
│   ├── tfidf_linear_svm/
│   └── xlm_roberta/
│
├── results/
│   ├── metrics/
│   ├── predictions/
│   └── figures/
│
└── presentation/
```

---

## 18. Team Collaboration Rules

To avoid confusion, the team should follow the same organization:

- Use the same train, validation, and test files.
- Do not recreate a different split for each model.
- Keep the raw dataset unchanged.
- Push the latest notebooks to GitHub.
- Use one branch for each task.
- Save trained models before sharing them.
- Avoid modifying the same notebook at the same time.
- Store final metrics and figures in the `results` folder.
- Merge validated work into the main branch.

Suggested branches:

```text
feature/data-preprocessing
feature/tfidf-svm
feature/xlm-roberta
feature/evaluation
docs/presentation
```

---

## 19. Team Members

Project developed by:

- emi-ane
- sadokkks
- selkardi
- WIAMBELOUARD
- Cassydy-prog

This project was completed as part of the Natural Language Processing course at **aivancity**.
