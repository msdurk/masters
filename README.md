# Phishing Email Detection and Robustness Evaluation

This repository contains notebook-based experiments for binary phishing email detection. The project compares classical machine-learning models, recurrent neural networks, and transformer-based models using two email feature representations. It also evaluates how model predictions change under text perturbation attacks.

The classification target is binary:

- `0` = legitimate email
- `1` = phishing email

Training and validation data are derived from the PhishFuzzer / Machine Wars email dataset. External test data are derived from the Kaggle Phishing Email Dataset.

No raw, cleaned, or redistributed datasets are stored in this repository. The repository contains model notebooks and files/instructions for downloading or preparing the required data locally.

## Project Structure

The repository is organized by model family under a top-level `models/` directory.

```text
.
├── README.md
├── LICENSE
├── models/
│   ├── deBERT/
│   │   ├── deBERT.ipynb
│   │   ├── debert_extended_features.ipynb
│   │   ├── no_spam_deBERT.ipynb
│   │   └── no_spam_deBERT_extended_features.ipynb
│   │
│   ├── distilBERT/
│   │   ├── DistilBERT.ipynb
│   │   ├── DistilBERT_extended_features.ipynb
│   │   ├── no_spam_DistilBERT.ipynb
│   │   └── no_spam_DistilBERT_extended_features.ipynb
│   │
│   ├── LogReg/
│   │   ├── Logreg.ipynb
│   │   ├── Logreg_extended_features.ipynb
│   │   ├── no_spam_Logreg.ipynb
│   │   └── no_spam_Logreg_extended_features.ipynb
│   │
│   ├── LSTM/
│   │   ├── LSTM.ipynb
│   │   ├── LSTM_extended_features.ipynb
│   │   ├── no_spam_LSTM.ipynb
│   │   └── no_spam_LSTM_extended_features.ipynb
│   │
│   ├── RF/
│   │   ├── RF.ipynb
│   │   ├── RF_extended_features.ipynb
│   │   ├── no_spam_RF.ipynb
│   │   └── no_spam_RF_extended_features.ipynb
│   │
│   └── SVM/
│       ├── SVM.ipynb
│       ├── SVM_extended_features.ipynb
│       ├── no_spam_SVM.ipynb
│       └── no_spam_SVM_extended_features.ipynb
```

The repository may also include data download or preparation files. These are included only to help users obtain and preprocess the datasets locally; they are not dataset files themselves.

## Notebook Naming Convention

The notebooks follow this naming pattern:

```text
[no_spam_]MODEL[_extended_features].ipynb
```

| Variant | Meaning |
|---|---|
| `MODEL.ipynb` | Baseline subject-body representation with the default spam-handling setup. |
| `MODEL_extended_features.ipynb` | Extended representation using sender, subject, body, and URL fields. |
| `no_spam_MODEL.ipynb` | Baseline subject-body representation with spam excluded from the phishing class. |
| `no_spam_MODEL_extended_features.ipynb` | Extended representation with spam excluded from the phishing class. |

The `no_spam` prefix indicates that spam examples are removed rather than treated as phishing-like examples. The `extended_features` suffix indicates that the notebook uses the richer tagged email representation instead of only subject and body text.

## Data Sources

This project uses separate data sources for model training/validation and external testing.

### Training and Validation Data

The training and validation data are derived from the PhishFuzzer / Machine Wars email dataset associated with the article *The Phish, The Spam, and The Valid: Generating Feature-Rich Emails for Benchmarking LLMs*.

Source repository:

```text
https://github.com/DataPhish/PhishFuzzer
```

The local file expected by the notebooks is:

```text
machinewars_filtered_emails.json
```

This file should be treated as a local filtered/preprocessed version of the PhishFuzzer training source. It is not included in this repository. Users should obtain the original data from the upstream repository and use the repository's download/preparation files or equivalent preprocessing steps before running the notebooks.

### External Test Data

The external test sets are derived from the Kaggle Phishing Email Dataset by Naser Abdullah Alam:

```text
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
```

The notebooks expect cleaned CSV versions of the Kaggle-derived files:

```text
CEAS_08_cleaned.csv
Nazario_cleaned.csv
Nigerian_Fraud_cleaned.csv
SpamAssasin_cleaned.csv
```

These files are used only for external evaluation and are not included in this repository. Users should download the Kaggle dataset directly from Kaggle and prepare the cleaned CSV files locally.

## Data Loading and Label Normalization

The notebooks normalize labels into a binary phishing-detection target.

For Machine Wars / PhishFuzzer-style training data:

- `phishing` is mapped to `1`
- `legitimate`, `valid`, `ham`, `benign`, and `safe` are mapped to `0`
- `spam` is either mapped to `1` or excluded, depending on whether the notebook is a `no_spam` variant

For external Kaggle-derived CSV test sets:

- phishing-like records are mapped to `1`
- legitimate or benign records are mapped to `0`
- spam is excluded in the `no_spam` configuration

The training data is split into training and validation sets using a stratified 80/20 split with a fixed random seed of `42`.

## Feature Representations

The project compares two feature representations.

### 1. Baseline Subject-Body Representation

The baseline representation builds each model input from only the email subject and body:

```python
def build_text_subject_body(row):
    subject = safe_str(row.get("subject", ""))
    body = safe_str(row.get("body", ""))
    return f"{subject}\n\n{body}".strip()
```

This representation tests how well the models classify emails using only the main textual content.

### 2. Extended Feature Representation

The extended representation adds explicit field tags for sender, subject, body, and URL information:

```python
def build_text_all_fields(row):
    sender = normalize_sender(row.get("sender", ""))
    subject = safe_str(row.get("subject", ""))
    body = safe_str(row.get("body", ""))
    url = safe_str(row.get("url", ""))

    return (
        f"[SENDER] {sender}\n"
        f"[SUBJECT] {subject}\n"
        f"[BODY] {body}\n"
        f"[URL] {url}"
    ).strip()
```

If a URL field is missing, the notebooks attempt to extract URLs from the email body using a regular expression. This representation tests whether sender metadata and URL cues improve phishing detection and robustness.

## Models

### Logistic Regression

The Logistic Regression notebooks use TF-IDF features followed by a balanced logistic regression classifier.

Typical TF-IDF configuration:

```python
TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)
```

Classifier configuration:

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=SEED,
)
```

### Support Vector Machine

The SVM notebooks use TF-IDF features with a linear support vector classifier.

```python
LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=5000,
)
```

### Random Forest

The Random Forest notebooks use TF-IDF features with a balanced random forest classifier.

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
```

### LSTM

The LSTM notebooks use Keras tokenization and padded sequences.

Main configuration:

```python
MAX_WORDS = 30000
MAX_LEN = 300
```

Typical architecture:

```python
Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
])
```

Training configuration:

```python
optimizer="adam"
loss="binary_crossentropy"
epochs=5
batch_size=32
EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
```

### DistilBERT

The DistilBERT notebooks fine-tune `distilbert-base-uncased` for sequence classification.

Main configuration:

```python
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
DISTILBERT_MAX_LENGTH = 512
DISTILBERT_BATCH_SIZE = 32
DISTILBERT_EPOCHS = 3
```

Training arguments include:

```python
learning_rate=2e-5
weight_decay=0.01
save_strategy="no"
report_to="none"
evaluation_strategy="epoch"  # or eval_strategy="epoch", depending on Transformers version
```

Some notebooks use a weighted trainer to account for class imbalance.

### DeBERTa

The DeBERTa notebooks fine-tune `microsoft/deberta-v3-base` for sequence classification.

Main configuration:

```python
DISTILBERT_MODEL_NAME = "microsoft/deberta-v3-base"
DISTILBERT_MAX_LENGTH = 512
DISTILBERT_BATCH_SIZE = 16
DISTILBERT_EPOCHS = 3
```

Training arguments include:

```python
learning_rate=1e-5
weight_decay=0.01
save_strategy="no"
report_to="none"
evaluation_strategy="epoch"  # or eval_strategy="epoch", depending on Transformers version
fp16=False
bf16=False
max_grad_norm=1.0
```

Some DeBERTa notebooks still use variable names such as `DISTILBERT_MODEL_NAME`; the underlying model checkpoint is DeBERTa.

## Evaluation Metrics

The notebooks report standard binary classification metrics:

- **Accuracy**: proportion of all examples classified correctly.
- **Precision**: proportion of predicted phishing emails that are actually phishing.
- **Recall**: proportion of actual phishing emails correctly detected.
- **F1-score**: harmonic mean of precision and recall.

The metric calculation follows this structure:

```python
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="binary",
    zero_division=0,
)
accuracy = accuracy_score(y_true, y_pred)
```

F1-score is especially important for this task because phishing detection requires balancing false positives and false negatives.

## Robustness and Perturbation Testing

The notebooks include robustness tests that modify email text and evaluate whether the model prediction changes. The perturbations include attacks such as:

- `benign_prefix`
- `benign_suffix`
- `contradiction`
- `synonym_attack`
- `keyword_deletion`
- `prefix_injection`
- `add_only_add3`
- `delete_only_del5`
- `hybrid_add3_delete5`

The robustness evaluation reports both normal attacked-set classification metrics and phishing-specific evasion metrics.

### Attack Metrics

For each attack, the notebooks compute classification metrics on the modified text. These metrics show how the model performs after the input emails have been perturbed.

### Phishing Evasion Metrics

For phishing emails that were originally classified correctly, the notebooks measure how often an attack causes the model to stop predicting phishing.

Important robustness metrics include:

- `attack_success_rate`: proportion of originally correct phishing examples that flip away from phishing after attack
- `robust_recall_on_orig_correct_phishing`: proportion of originally correct phishing examples that remain classified as phishing after attack
- `avg_prob_drop`: average decrease in predicted phishing probability after attack

For models that do not naturally output probabilities, probability-like scores may be approximated or calibrated depending on the notebook implementation.

## Running the Notebooks

The notebooks are designed for Google Colab-style execution.

### 1. Install dependencies

Most notebooks install their dependencies directly in the first cells.

For classical machine-learning and LSTM notebooks:

```python
!pip install -q pandas numpy scikit-learn tensorflow nltk
```

For transformer notebooks:

```python
!pip install -q pandas numpy scikit-learn transformers datasets accelerate torch nltk
```

### 2. Download or prepare data files

Use the repository's data download/preparation files, or manually download the original datasets from their upstream sources. The expected local files are:

```text
machinewars_filtered_emails.json
CEAS_08_cleaned.csv
Nazario_cleaned.csv
Nigerian_Fraud_cleaned.csv
SpamAssasin_cleaned.csv
```

These files should remain local and should not be committed to the repository unless the redistribution terms of the original sources are explicitly satisfied.

### 3. Upload data files when using Colab

The notebooks use Colab upload logic:

```python
from google.colab import files
uploaded = files.upload()
```

Upload the required JSON and CSV files listed above.

### 4. Run cells in order

Each notebook is self-contained. Run all cells sequentially to:

1. install dependencies
2. upload or load data
3. normalize labels
4. build the selected feature representation
5. train the model
6. evaluate validation and external test performance
7. run robustness tests where included

## Expected Outputs

The notebooks produce tables for:

- validation metrics
- external test-set metrics
- attacked-set metrics
- phishing evasion metrics

Some notebooks save result files to a results directory, for example:

```text
/content/results/{ACTIVE_MODEL_NAME}_val_basic_attack_results.csv
/content/results/{ACTIVE_MODEL_NAME}_{safe_name}_basic_attack_results.csv
/content/results/{ACTIVE_MODEL_NAME}_test_add_delete_hybrid_metrics.csv
/content/results/{ACTIVE_MODEL_NAME}_test_add_delete_hybrid_evasion.csv
```

## Reproducibility

The notebooks use a fixed seed:

```python
SEED = 42
```

The seed is applied to Python, NumPy, TensorFlow, and PyTorch where applicable. GPU availability may affect training speed and can introduce small numerical differences, especially for neural models.

## Suggested Requirements File

A minimal `requirements.txt` for the full repository is:

```text
pandas
numpy
scikit-learn
tensorflow
nltk
transformers
datasets
accelerate
torch
```

## Limitations

- The project is notebook-based rather than packaged as reusable Python modules.
- Dataset files are expected to be uploaded or generated locally at runtime; dataset files are not included in the repository.
- Some variable names are inherited across notebooks and may not perfectly match the model family.
- The robustness attacks are heuristic text perturbations and should not be interpreted as exhaustive adversarial robustness testing.
- Results depend on the exact dataset versions and preprocessing used during execution.

## License

The source code and notebooks in this repository are licensed under the MIT License. See the `LICENSE` file for details.

The datasets used by the experiments are not included in this repository and are not covered by the MIT License. Dataset users must follow the terms of the original data sources:

- Training/validation source: PhishFuzzer / Machine Wars data from the upstream GitHub repository.
- External test source: Kaggle Phishing Email Dataset.

The repository only contains models, notebooks, and files/instructions for downloading or preparing the data locally.
