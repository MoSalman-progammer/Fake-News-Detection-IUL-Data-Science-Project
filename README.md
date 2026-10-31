# Development of a Machine Learning-Based Fake News Text Detection System

**Project aim:** Identifying fake news articles using text patterns.

An academic Data Science project that explores whether a news article can be classified as **Fake** or **True** from its textual content. The project combines exploratory data analysis, text preprocessing, TF-IDF feature extraction, a Linear Support Vector Machine (LinearSVC), model evaluation, and a Streamlit interface for interactive predictions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Research Background and References](#research-background-and-references)
- [Technologies and Libraries](#technologies-and-libraries)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model and Evaluation](#model-and-evaluation)
- [Application Screenshots](#application-screenshots)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Validation](#validation)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

---

## Project Overview

Online platforms make it easy for information to spread quickly. However, misleading or fabricated articles can circulate alongside reliable reporting. This project investigates the use of machine learning and natural language processing (NLP) to identify textual patterns associated with fake and true news articles.

The trained classifier takes article text as input and returns one of two labels:

- **Fake** — the text is classified as belonging to the fake-news class represented in the training dataset.
- **True** — the text is classified as belonging to the true-news class represented in the training dataset.

The project is an educational text-classification system. It does not independently verify claims, investigate sources, or establish whether a statement is factually correct.

## Objectives

1. Load and inspect the supplied fake-news and true-news datasets.
2. Clean the data and prepare a combined dataset for analysis.
3. Explore class distribution, subject distribution, and textual patterns.
4. Apply TF-IDF to convert article text into numerical features.
5. Train a machine-learning classifier for binary text classification.
6. Evaluate the classifier using appropriate classification metrics.
7. Identify text features that influence the classifier's predictions.
8. Save the trained model pipeline for reuse.
9. Build a Streamlit application to demonstrate predictions on user-provided text.

## Dataset

The project uses two CSV files:

| File | Label | Original number of rows |
|---|---|---:|
| `Fake.csv` | Fake (0) | 23,481 |
| `True.csv` | True (1) | 21,417 |
| **Combined** | | **44,898** |

The original files contain article information such as title, text, subject, and date. After the project's cleaning and filtering steps, the working dataset contains **38,644 records**:

| Class | Label | Cleaned records |
|---|---:|---:|
| Fake | 0 | 17,453 |
| True | 1 | 21,191 |
| **Total** | | **38,644** |

The `subject` field was found to be strongly associated with the target label. It was therefore not used as a model input, because doing so could allow the classifier to exploit dataset-specific label clues rather than article text.

### Dataset files and distribution

The source CSV files are expected in:

```text
News_dataset/
├── Fake.csv
└── True.csv
```

**Dataset availability note:** The CSV files are large (together over 100 MB). They may not be included in a standard GitHub repository because GitHub blocks regular Git files larger than 100 MB. If they are not committed, obtain them from the dataset source and place them in `News_dataset/`. Git LFS or a documented external download location can be used if redistribution is permitted.

### Dataset provenance and citation

The project uses the commonly circulated Fake/True news CSV dataset. The research papers below are relevant background references for fake-news text classification and n-gram-based machine learning. They should be cited as research references, not described as the direct creator or uploader of the CSV files unless the dataset documentation explicitly establishes that relationship.

**Dataset download/source page:**
[Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Please retain the dataset provider's attribution and follow the licence/terms shown on the source page when downloading, using, or redistributing the data.

## Research Background and References

The project is informed by the following publications:

1. Ahmed, H., Traore, I., & Saad, S. (2018). “Detecting opinion spams and fake news using text classification.” *Security and Privacy*, 1(1), e9. Wiley.
   https://doi.org/10.1002/spy2.9

2. Ahmed, H., Traore, I., & Saad, S. (2017). “Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques.” In I. Traore, I. Woungang, & A. Awad (Eds.), *Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments*. Lecture Notes in Computer Science, vol. 10618, pp. 127–138. Springer, Cham.
   https://doi.org/10.1007/978-3-319-69155-8_9

These publications provide research context for using textual features and machine-learning methods to detect fake news. They are not a claim that this project reproduces the exact methods or experimental results of those papers.

## Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Data analysis and model development |
| Pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualisation |
| Scikit-learn | TF-IDF, model training, and evaluation |
| LinearSVC | Binary text classifier |
| Joblib | Saving and loading the trained pipeline |
| Streamlit | Interactive web application |

## Repository Structure

The project is organised approximately as follows:

```text
Fake-News-Detection-IUL-Data-Science-Project/
│
├── News_dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── notebook/
│   └── Fake_News_Detection.ipynb
│
├── plots/
│   ├── confusion_matrix.png
│   ├── dataset_distribution.png
│   ├── model_comparison.png
│   └── project_workflow.png
├── screenshots/
│   ├── 01_dataset_overview.png
│   ├── 02_data_cleaning.png
│   ├── 03_dataset_distribution.png
│   ├── 04_subject_analysis.png
│   ├── 05_wordcloud_fake.png
│   ├── 06_wordcloud_true.png
│   ├── 07_model_comparison.png
│   ├── 08_confusion_matrix.png
│   ├── 09_feature_importance_fake.png
│   ├── 10_feature_importance_true.png
│   ├── 11_streamlit_homepage.png
│   ├── 12_streamlit_fake_prediction.png
│   └── 13_streamlit_true_prediction.png
│
├── validation/
│   ├── validate_model.py
│   └── app_validation_samples.csv
│
├── app.py
├── fake_news_detection_svm.pkl
├── generate_plots.py
├── requirements.txt
├── README.md
└── .gitignore
```

The screenshot names above are the expected names for the screenshots documented in this README. If your actual filenames differ, update the Markdown image paths to match the files in your `screenshots/` folder. The `plots/` folder may contain generated plots separately from presentation screenshots.

## Methodology

### 1. Data loading

The two source CSV files are loaded and combined into one dataset. A target label is assigned to each class:

- `0` represents Fake.
- `1` represents True.

### 2. Data cleaning

The cleaning stage prepares the records for analysis and model development. It includes handling missing or empty article text and removing unsuitable or duplicate records according to the notebook's cleaning procedure.

The cleaned dataset used for the reported model contains 38,644 records.

### 3. Exploratory data analysis

The notebook examines the distribution of the two classes, article subjects, and common terms in the article text. Visualisations help identify imbalances and dataset-specific patterns.

### 4. Text preparation and feature extraction

Text is converted into numerical features using **Term Frequency–Inverse Document Frequency (TF-IDF)**. TF-IDF represents the importance of terms in a document relative to the collection of documents. The resulting feature vectors can be used by a conventional machine-learning classifier.

### 5. Model training

A **Linear Support Vector Classifier (LinearSVC)** is trained using the TF-IDF text features. The trained TF-IDF and classifier steps are saved together as a reusable model pipeline:

```text
fake_news_detection_svm.pkl
```

### 6. Prediction

The Streamlit application loads the saved pipeline, prepares the input text using the application's preprocessing steps, and returns a Fake or True prediction.

## Exploratory Data Analysis

The project includes visual analysis of the dataset.

### Dataset distribution

Shows the number of fake and true articles in the cleaned dataset.

![Dataset distribution](screenshots/03_dataset_distribution.png)

### Subject analysis

Explores how article subjects are distributed across the two labels. Subject was examined during analysis but excluded from model input because it can reveal dataset-specific label information.

![Subject analysis](screenshots/04_subject_analysis.png)

### Word clouds

Word clouds provide a visual summary of frequently occurring terms in each class. They are exploratory illustrations, not proof that a word is inherently associated with fake or true information.

**Fake news word cloud**

![Fake news word cloud](screenshots/05_wordcloud_fake.png)

**True news word cloud**

![True news word cloud](screenshots/06_wordcloud_true.png)

## Model and Evaluation

### Selected model

The final saved classifier is a TF-IDF + LinearSVC pipeline.

### Reported evaluation results

| Metric | Result |
|---|---:|
| Accuracy | 96.24% |
| Balanced accuracy | 97.11% |

### Confusion matrix

The reported confusion matrix is:

| Actual class | Predicted Fake | Predicted True |
|---|---:|---:|
| Fake | 6,545 | 56 |
| True | 812 | 15,682 |

![Confusion matrix](screenshots/08_confusion_matrix.png)

For the Fake class, the model correctly classified 6,545 records and classified 56 as True. For the True class, it correctly classified 15,682 records and classified 812 as Fake.

These results apply to the evaluation split used during development. They do not establish performance on independent news sources or future articles.

### Model comparison

If the model-comparison plot contains results from multiple trained models, it can be viewed here:

![Model comparison](screenshots/07_model_comparison.png)

Only include this image if the chart is populated with actual measured results. Do not report placeholder or example scores as experimental results.

### Feature analysis

The feature-importance visualisations show terms associated with the model's learned decision function. For a binary linear classifier, coefficient direction indicates which class a feature supports; it does not establish that the term is objectively true or false.

**Features associated with Fake predictions**

![Fake-associated features](screenshots/09_feature_importance_fake.png)

**Features associated with True predictions**

![True-associated features](screenshots/10_feature_importance_true.png)

## Application Screenshots

The Streamlit interface allows a user to enter news text and obtain a model prediction.

### Application homepage

![Streamlit application homepage](screenshots/11_streamlit_homepage.png)

### Example Fake prediction

![Example fake news prediction](screenshots/12_streamlit_fake_prediction.png)

### Example True prediction

![Example true news prediction](screenshots/13_streamlit_true_prediction.png)

The screenshots are demonstration examples. A prediction is the output of the trained classifier and should not be treated as independent fact-checking.

## Installation

### Requirements

- Python installed on the computer.
- Git, if cloning the repository.
- The project files.
- The trained model file `fake_news_detection_svm.pkl`.
- The source CSV files if you want to rerun the notebook or dataset-dependent scripts.

### 1. Clone the repository

```bash
git clone https://github.com/MoSalman-progammer/Fake-News-Detection-IUL-Data-Science-Project.git
```

### 2. Enter the project directory

```bash
cd Fake-News-Detection-IUL-Data-Science-Project
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Run the Streamlit application

From the project root directory:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in a browser to use the application.

### Run the notebook

Open:

```text
notebook/Fake_News_Detection.ipynb
```

Run the notebook cells in order. The source CSV files must be present in `News_dataset/` for the data-loading and training workflow.

### Run the validation script

From the project root:

```bash
python validation/validate_model.py
```

The validation script expects the model and dataset paths configured in the script to match the repository structure.

## Validation

The project includes a validation script and sample inputs under `validation/`. These are intended to check that the saved model can be loaded and used for predictions.

A successful sample prediction only confirms that the pipeline executes for that input. It does not establish that the prediction is factually correct.

## Limitations

- **Dataset dependence:** The model learns from the patterns and sources represented in its training data. Different publishers, topics, writing styles, and time periods may produce different results.
- **Potential dataset bias:** Formatting conventions and source-specific language can become predictive signals. High test-set performance does not automatically imply strong generalisation.
- **No claim verification:** The system analyses text patterns. It does not verify claims against trusted sources or external evidence.
- **Subject leakage risk:** Subject was strongly associated with the labels in this dataset, so it was excluded from the model features.
- **Date evaluation caveat:** A chronological evaluation was not used as the basis for the reported performance figures. Date parsing issues in the working notebook should be resolved before making claims about time-based generalisation.
- **Binary classification:** The model returns one of two labels, even when an article may be mixed, satirical, incomplete, or outside the dataset's domain.
- **Language and context:** Performance may be limited for languages, dialects, or contexts not adequately represented in the dataset.

## Future Work

Possible extensions include:

1. Evaluate the classifier on an independent dataset from different publishers.
2. Test on more recent articles and use a correctly parsed chronological split.
3. Investigate methods to reduce source and formatting bias.
4. Compare additional models using the same train/test split and evaluation procedure.
5. Add explainability to make model decisions easier to inspect.
6. Explore multilingual support.
7. Add links to trusted fact-checking sources as a separate verification aid.
8. Improve input validation and user guidance in the Streamlit application.

## Acknowledgements

The project draws on publicly available fake/real news data and published research on fake-news text classification. Please consult the dataset provider's page for the applicable data terms and attribution requirements.

## Author

**Mohammad Salman**
B.Tech – Computer Science and Engineering
Data Science and Artificial Intelligence
Integral University

## Disclaimer

This project was developed for academic and educational purposes. Its output is a machine-learning classification based on patterns learned from a particular dataset. It is not a definitive determination of whether a news article, publisher, or claim is authentic. Users should consult reliable sources and independent fact-checking evidence before making decisions based on news content.
'''