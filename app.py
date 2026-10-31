
import re
import joblib
import streamlit as st
from datetime import datetime
from io import BytesIO

# ============================================================
# 1. MODEL
# ============================================================

model = joblib.load("fake_news_detection_svm.pkl")

political_terms = [
    "trump", "obama", "hillary", "clinton", "republican",
    "democrat", "gop", "president", "senate", "congress"
]


# ============================================================
# 2. PREPROCESSING
# ============================================================

def remove_text_artifacts(text):
    text = str(text)
    text = re.sub(r"\bReuters\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(read more|featured image|image via)\b",
        " ", text, flags=re.IGNORECASE
    )
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_political_terms(text):
    text = str(text)

    for term in political_terms:
        text = re.sub(
            rf"\b{term}\b",
            " ",
            text,
            flags=re.IGNORECASE
        )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_article(title, article):
    combined_text = str(title) + " " + str(article)
    combined_text = remove_text_artifacts(combined_text)
    combined_text = remove_political_terms(combined_text)
    return combined_text


# ============================================================
# 3. SAMPLE ARTICLES
# ============================================================

sample_articles = {
    "Sample 1": {
        "title": "Government announces new public transport initiative",
        "article": """
        The government has announced a new public transport initiative
        aimed at improving transportation facilities across several
        major cities. Officials said the programme will focus on
        infrastructure development, improved services and public
        accessibility. The proposal will be reviewed by the relevant
        departments before implementation. Further details regarding
        funding and the implementation schedule are expected to be
        released in the coming weeks.
        """
    },

    "Sample 2": {
        "title": "Scientists report discovery of an unusual deep-sea species",
        "article": """
        Researchers have reported the discovery of an unusual marine
        species during an expedition in deep ocean waters. The research
        team collected observations and samples for further analysis.
        Scientists said the discovery could provide additional
        information about marine biodiversity and the ability of
        organisms to survive in extreme environments. The findings
        will be examined further before a detailed scientific report
        is published.
        """
    }
}


# ============================================================
# 4. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# 5. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
/* Compact Analyze and Clear buttons */
.st-key-analyze_article_button button,
.st-key-clear_article_button button {
    min-height: 2.3rem;
    padding: 0.35rem 0.75rem;
    font-size: 0.9rem;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 750;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #888888;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.result-card {
    padding: 22px;
    border-radius: 12px;
    margin-top: 15px;
}

.small-muted {
    color: #888888;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 6. SESSION STATE
# ============================================================

defaults = {
    "page": "home",
    "headline_input": "",
    "article_input": "",
    "prediction_result": None,
    "prediction_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 7. CALLBACK FUNCTIONS
# ============================================================

def load_sample(sample_name):
    sample = sample_articles[sample_name]

    st.session_state.headline_input = sample["title"]
    st.session_state.article_input = sample["article"].strip()
    st.session_state.prediction_result = None


def clear_inputs():
    st.session_state.headline_input = ""
    st.session_state.article_input = ""
    st.session_state.prediction_result = None

def article_history_key(title, article):
    """
    Generate a unique key for an article.
    Ignores capitalisation and repeated whitespace.
    """

    normalized_title = re.sub(
        r"\s+", " ", str(title)
    ).strip().casefold()

    normalized_article = re.sub(
        r"\s+", " ", str(article)
    ).strip().casefold()

    return normalized_title, normalized_article
def analyze_article():
    title = st.session_state.headline_input.strip()
    article = st.session_state.article_input.strip()

    if not title or not article:
        st.session_state.prediction_result = {
            "error": "Please enter both the headline and article."
        }
        return

    prepared_text = prepare_article(title, article)

    prediction = int(model.predict([prepared_text])[0])

    result = {
        "title": title,
        "article": article,
        "prediction": prediction,
        "label": "Real News" if prediction == 1 else "Fake News",
        "timestamp": datetime.now().strftime("%d %B %Y, %I:%M:%S %p"),
        "word_count": len((title + " " + article).split()),
        "character_count": len(title + " " + article),
    }

    st.session_state.prediction_result = result

    # Add only unique articles to session history

    new_key = article_history_key(title, article)

    history = st.session_state.prediction_history

    existing_index = next(
        (
            index for index, item in enumerate(history)
            if article_history_key(
                item["title"],
                item["article"]
            ) == new_key
        ),
        None
    )

    if existing_index is not None:

        # Update the existing entry instead of creating a duplicate
        existing_item = history.pop(existing_index)
        existing_item.update(result)

        # Move the existing article to the top
        history.insert(0, existing_item)

    else:

        # Add a genuinely new article
        history.insert(0, result)

    # Keep the latest 20 unique articles
    st.session_state.prediction_history = history[:20]


def generate_report(result):
    prediction_text = (
        "REAL NEWS"
        if result["prediction"] == 1
        else "FAKE NEWS"
    )

    report = f"""
FAKE NEWS DETECTION SYSTEM
PREDICTION REPORT
========================================

PREDICTION RESULT
----------------------------------------
Classification: {prediction_text}
Date and Time: {result["timestamp"]}

ARTICLE INFORMATION
----------------------------------------
Headline:
{result["title"]}

Article:
{result["article"]}

ARTICLE STATISTICS
----------------------------------------
Word Count: {result["word_count"]}
Character Count: {result["character_count"]}

MODEL INFORMATION
----------------------------------------
Algorithm: Linear Support Vector Machine
Feature Extraction: TF-IDF
Classification Type: Binary Text Classification

IMPORTANT DISCLAIMER
----------------------------------------
This result is generated by a machine learning
model trained on a specific news dataset.

The model identifies patterns learned from
training data. It does not independently
verify facts, sources or claims.

A Real News prediction does not guarantee
that the article is factually correct.

Always verify important information using
reliable and independent sources.

========================================
Generated by Fake News Detection System
"""

    return report


# ============================================================
# 8. NAVIGATION
# ============================================================

if st.session_state.page == "home":

    # --------------------------------------------------------
    # MAIN HEADER
    # --------------------------------------------------------

    col1, col2 = st.columns([5, 1])

    with col1:
        st.markdown(
            '<div class="main-title">Fake News Detection System</div>',
            unsafe_allow_html=True
        )

    with col2:
        if st.button("About Project"):
            st.session_state.page = "about"
            st.rerun()

    st.markdown(
        '<div class="subtitle">'
        'Analyze news articles using Natural Language Processing '
        'and Machine Learning.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()



    # --------------------------------------------------------
    # INPUT, ANALYTICS AND PREDICTION LAYOUT
    # --------------------------------------------------------

    st.subheader("Enter News Details")

    input_col, analytics_col = st.columns(
        [3.3, 1.2],
        gap="large"
    )

    # ========================================================
    # LEFT SIDE: INPUT, BUTTONS AND PREDICTION RESULT
    # ========================================================

    with input_col:

        # News headline
        title = st.text_input(
            "News Headline",
            key="headline_input",
            placeholder="Enter the news headline..."
        )

        # News article
        article = st.text_area(
            "News Article",
            key="article_input",
            height=250,
            placeholder="Paste the complete news article here..."
        )

        # ----------------------------------------------------
        # ACTION AND SAMPLE BUTTONS
        # ----------------------------------------------------

        btn_analyze, btn_clear, btn_sample1, btn_sample2 = st.columns(
            [1.3, 0.8, 1, 1],
            gap="small"
        )

        with btn_analyze:
            st.button(
                "Analyze",
                key="analyze_article_button",
                type="primary",
                use_container_width=True,
                on_click=analyze_article
            )

        with btn_clear:
            st.button(
                "Clear",
                key="clear_article_button",
                use_container_width=True,
                on_click=clear_inputs
            )

        with btn_sample1:
            st.button(
                "Sample 1",
                key="sample_0",
                use_container_width=True,
                on_click=load_sample,
                args=("Sample 1",)
            )

        with btn_sample2:
            st.button(
                "Sample 2",
                key="sample_1",
                use_container_width=True,
                on_click=load_sample,
                args=("Sample 2",)
            )

        st.caption(
            "Sample articles are for demonstration purposes."
        )


        # ----------------------------------------------------
        # PREDICTION RESULT - ALWAYS VISIBLE
        # ----------------------------------------------------

        result = st.session_state.prediction_result

        st.divider()
        st.subheader("Analysis Result")

        if result is None:

            st.info(
                "Your prediction will appear here after you analyse an article."
            )

        elif "error" in result:

            st.warning(result["error"])

        else:

            if result["prediction"] == 0:

                st.error("🚨 FAKE NEWS")

                st.write(
                    "The model classified this article as Fake News."
                )

            else:

                st.success("✅ REAL NEWS")

                st.write(
                    "The model classified this article as Real News."
                )

            st.caption(
                f"Analysis completed: {result['timestamp']}"
            )

            st.info(
                "This result is based on patterns learned from the "
                "training dataset. It is not independent verification "
                "of the article."
            )

        # ----------------------------------------------------
        # DOWNLOAD REPORT - ALWAYS VISIBLE
        # ----------------------------------------------------

        st.subheader("Download Prediction Report")

        if result is not None and "error" not in result:

            report = generate_report(result)

            st.download_button(
                label="⬇ Download Prediction Report",
                data=report.encode("utf-8"),
                file_name="fake_news_prediction_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        else:

            st.download_button(
                label="⬇ Download Prediction Report",
                data="",
                file_name="fake_news_prediction_report.txt",
                mime="text/plain",
                use_container_width=True,
                disabled=True
            )



    # ========================================================
    # RIGHT SIDE: LIVE ANALYTICS
    # ========================================================

    with analytics_col:

        clean_title = st.session_state.headline_input.strip()
        clean_article = st.session_state.article_input.strip()

        combined_input = " ".join(
            part for part in [clean_title, clean_article] if part
        )

        total_words = len(combined_input.split())
        headline_words = len(clean_title.split())
        article_words = len(clean_article.split())

        character_count = len(clean_title) + len(clean_article)

        reading_time = (total_words + 199) // 200

        st.markdown("### Live Analytics")

        with st.container(border=True):

            st.metric(
                "Total Words",
                f"{total_words:,}"
            )

            st.divider()

            st.metric(
                "Headline Words",
                f"{headline_words:,}"
            )

            st.metric(
                "Article Words",
                f"{article_words:,}"
            )

            st.divider()

            st.metric(
                "Characters",
                f"{character_count:,}"
            )

            st.metric(
                "Estimated Reading Time",
                f"{reading_time} min" if total_words else "—"
            )

        st.caption(
            "Statistics update automatically as you type."
        )

    st.divider()




    # --------------------------------------------------------
    # SESSION PREDICTION HISTORY
    # --------------------------------------------------------

    st.divider()
    st.subheader("Prediction History")

    # Remove existing duplicate entries from the current session
    unique_history = []
    seen_articles = set()

    for item in st.session_state.prediction_history:

        key = article_history_key(
            item["title"],
            item["article"]
        )

        if key not in seen_articles:
            seen_articles.add(key)
            unique_history.append(item)

    st.session_state.prediction_history = unique_history[:20]

    history = st.session_state.prediction_history

    if not history:
        st.caption(
            "Your predictions will appear here during this session."
        )

    else:
        st.caption(
            f"{len(history)} prediction(s) recorded in this session. "
            "History is cleared when the session state is reset."
        )

        for index, item in enumerate(history):

            label = item["label"]

            with st.expander(
                f"{index + 1}. {label} — {item['title'][:70]}"
            ):

                st.write(f"**Headline:** {item['title']}")
                st.write(f"**Prediction:** {label}")
                st.write(f"**Date:** {item['timestamp']}")

                st.write(
                    f"**Word Count:** {item['word_count']}"
                )

                if st.button(
                    "Load This Article",
                    key=f"history_load_{index}"
                ):
                    st.session_state.headline_input = item["title"]
                    st.session_state.article_input = item["article"]
                    st.session_state.prediction_result = None
                    st.rerun()

        if st.button("Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()

    st.divider()

    st.caption(
        "Fake News Detection System | Machine Learning Project"
    )


# ============================================================
# 9. ABOUT PAGE
# ============================================================

elif st.session_state.page == "about":

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("About the Project")

    st.write(
        "A machine learning project that uses Natural Language "
        "Processing (NLP) to classify news articles as Fake News "
        "or Real News."
    )

    st.divider()

    # --------------------------------------------------------
    # 1. PROJECT OVERVIEW
    # --------------------------------------------------------

    st.header("1. Project Overview")

    st.write("""
    The objective of this project is to develop a text classification
    system that identifies patterns commonly associated with fake
    and real news articles.

    The system accepts a news headline and article body, processes
    the text, converts it into numerical features using TF-IDF,
    and applies a trained Linear Support Vector Machine classifier.
    """)

    st.subheader("Project Objectives")

    st.markdown("""
    - Process and prepare a news article dataset.
    - Apply Natural Language Processing techniques.
    - Convert text into numerical features using TF-IDF.
    - Train and compare multiple machine learning algorithms.
    - Evaluate the final model using classification metrics.
    - Develop a web application for making predictions.
    """)

    st.divider()

    # --------------------------------------------------------
    # 2. DATA SCIENCE WORKFLOW
    # --------------------------------------------------------

    st.header("2. Complete Data Science Workflow")

    st.write("""
    A Data Science project follows a systematic process, starting
    with understanding a problem and ending with deployment,
    monitoring and continuous improvement.
    """)

    workflow_steps = [
        (
            "1. Problem Definition",
            "Understand the problem, define objectives and identify the expected output.",
            "Our objective is to classify news articles into Fake News or Real News."
        ),
        (
            "2. Data Collection",
            "Gather relevant data from appropriate sources.",
            "We collected Fake.csv and True.csv containing news headlines, article text, subjects and dates."
        ),
        (
            "3. Data Wrangling",
            "Convert raw data into a usable format by handling missing values, duplicates, incorrect types and inconsistencies.",
            "We combined both datasets, assigned labels, removed empty article bodies and duplicate article text, and processed dates."
        ),
        (
            "4. Data Exploration (EDA)",
            "Examine data using statistical summaries, graphs and distributions to understand patterns and unusual values.",
            "We examined class distributions, article dates, subject categories, duplicate text and differences between Fake and Real articles."
        ),
        (
            "5. Feature Selection and Engineering",
            "Choose useful input features and transform data into a format suitable for machine learning.",
            "We combined headlines and article bodies, used TF-IDF to extract text features, and experimented with removing selected political terms."
        ),
        (
            "6. Model Selection",
            "Choose an appropriate machine learning algorithm based on the problem, data and requirements.",
            "We compared Logistic Regression, Multinomial Naive Bayes and Linear SVM."
        ),
        (
            "7. Model Training",
            "Feed training data to the selected algorithm so it can learn patterns and relationships.",
            "We trained classifiers using TF-IDF features and articles dated before 2017."
        ),
        (
            "8. Model Evaluation",
            "Evaluate model performance using suitable metrics and unseen test data.",
            "We used accuracy, balanced accuracy, precision, recall, F1-score and a confusion matrix. Articles from 2017 were used for chronological testing."
        ),
        (
            "9. Deployment",
            "Make the trained model available through an application or service.",
            "We saved the trained Linear SVM pipeline using Joblib and developed a Streamlit application."
        ),
        (
            "10. Monitoring and Improvement",
            "Identify weaknesses and improve the system using new data and experiments.",
            "We investigated source-related patterns, removed selected text artifacts, tested political-term ablation and identified limitations in generalization."
        )
    ]

    for heading, explanation, example in workflow_steps:
        with st.expander(heading):
            st.write(explanation)
            st.markdown("**Application in this project:**")
            st.write(example)

    st.info(
        "The Data Science workflow is iterative. Findings during "
        "evaluation can lead to further data cleaning, feature "
        "engineering or model training."
    )

    st.divider()

    # --------------------------------------------------------
    # 3. PROJECT ARCHITECTURE
    # --------------------------------------------------------

    st.header("3. Project Architecture")

    st.write("""
    The project consists of two connected workflows: model development
    and deployment. The diagrams below show both processes.
    """)

    st.subheader("A. Model Training Workflow")

    training_architecture = """
    digraph Training {

        graph [
            rankdir=LR,
            bgcolor="transparent",
            nodesep=0.12,
            ranksep=0.18,
            margin=0.05,
            pad=0.08,
            splines=polyline
        ];

        node [
            shape=box,
            style="rounded,filled",
            fontname="Arial",
            fontsize=9,
            margin="0.08,0.06",
            color="#777777",
            width=1.25,
            height=0.55,
            fixedsize=false
        ];

        edge [
            color="#888888",
            arrowsize=0.65
        ];

        A [label="CSV Dataset", fillcolor="#DCEBFA"];
        B [label="Clean + EDA", fillcolor="#DCEBFA"];
        C [label="Text Prep", fillcolor="#DCEBFA"];
        D [label="TF-IDF", fillcolor="#DCEBFA"];
        E [label="Train + Evaluate", fillcolor="#DCEBFA"];
        F [label="Saved SVM", fillcolor="#DFF2DF"];

        A -> B -> C -> D -> E -> F;
    }
    """

    st.graphviz_chart(
        training_architecture,
        use_container_width=True
    )

    st.subheader("B. Application Prediction Workflow")

    prediction_architecture = """
    digraph Prediction {

        graph [
            rankdir=LR,
            bgcolor="transparent",
            nodesep=0.12,
            ranksep=0.18,
            margin=0.05,
            pad=0.08,
            splines=polyline
        ];

        node [
            shape=box,
            style="rounded,filled",
            fontname="Arial",
            fontsize=9,
            margin="0.08,0.06",
            color="#777777",
            width=1.25,
            height=0.55,
            fixedsize=false
        ];

        edge [
            color="#888888",
            arrowsize=0.65
        ];

        A [label="User Input", fillcolor="#FCE8D5"];
        B [label="Validation", fillcolor="#FCE8D5"];
        C [label="Text Prep", fillcolor="#FCE8D5"];
        D [label="TF-IDF", fillcolor="#DCEBFA"];
        E [label="Linear SVM", fillcolor="#DCEBFA"];
        F [label="Fake / Real", fillcolor="#DFF2DF"];

        A -> B -> C -> D -> E -> F;
    }
    """

    st.graphviz_chart(
        prediction_architecture,
        use_container_width=True
    )

    st.caption(
        "The trained SVM model and TF-IDF vectorizer are saved and "
        "loaded by the Streamlit application for predictions."
    )

    st.divider()

    # --------------------------------------------------------
    # 4. PREDICTION PIPELINE
    # --------------------------------------------------------

    st.header("4. Prediction Pipeline")

    st.write("""
    The prediction pipeline describes the operations performed
    whenever a user submits a news article through the application.
    """)

    prediction_steps = [
        (
            "1. User Input",
            "The user enters a news headline and article body."
        ),
        (
            "2. Text Combination",
            "The headline and article body are combined into a single text input."
        ),
        (
            "3. Text Preprocessing",
            "Selected artifacts, URLs and predefined political terms are removed."
        ),
        (
            "4. TF-IDF Vectorization",
            "The saved TF-IDF vectorizer converts processed text into numerical features using the vocabulary learned during training."
        ),
        (
            "5. Linear SVM Classification",
            "The trained Linear Support Vector Machine processes the numerical feature vector and predicts a class label."
        ),
        (
            "6. Prediction Output",
            "The application displays Fake News or Real News based on the predicted label."
        )
    ]

    for heading, explanation in prediction_steps:
        with st.expander(heading):
            st.write(explanation)

    st.subheader("Prediction Logic")

    st.code("""
    Headline + Article
           |
           v
    Text Preprocessing
           |
           v
    TF-IDF Vectorization
           |
           v
    Trained Linear SVM
           |
           v
      Predicted Label
           |
       +---+---+
       |       |
       v       v
     Fake     Real
     News     News
    """, language="text")

    st.caption(
        "The application uses the TF-IDF vocabulary and Linear SVM "
        "parameters learned during training. It does not retrain "
        "the model for each submitted article."
    )

    st.divider()

    # --------------------------------------------------------
    # 5. DATASET
    # --------------------------------------------------------

    st.header("5. Dataset")

    st.write("""
    The project uses two CSV files: Fake.csv and True.csv.
    Both contain news headlines, article text, subjects and dates.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Original Articles", "44,898")
    col2.metric("Cleaned Articles", "38,644")
    col3.metric("Dataset Columns", "5")

    st.image(
        "plots/dataset_distribution.png",
        caption="Distribution of Fake and Real articles in the cleaned dataset.",
        width=600
    )

    st.markdown("""
    **Original dataset:**
    - Fake articles: 23,481
    - Real articles: 21,417

    **After preprocessing:**
    - Fake articles: 17,453
    - Real articles: 21,191
    - Total: 38,644
    """)

    st.caption(
        "The cleaned dataset was created after removing empty "
        "article bodies and duplicate article text."
    )

    st.divider()

    # --------------------------------------------------------
    # 6. DATA PREPROCESSING
    # --------------------------------------------------------

    st.header("6. Data Preprocessing")

    st.write("""
    Data preprocessing was performed to prepare articles for
    machine learning.
    """)

    st.markdown("""
    1. **Combining text:** Headline and article body were combined.
    2. **Removing empty articles:** Articles without body text were removed.
    3. **Removing duplicates:** Repeated article text was removed.
    4. **Removing text artifacts:** Selected source and formatting artifacts and URLs were removed.
    5. **Political-term ablation:** Selected political terms were removed during a separate experiment to study their influence on predictions.
    """)

    st.warning(
        "Political-term removal was an experimental feature-ablation "
        "step, not a universal rule for news classification."
    )

    st.divider()

    # --------------------------------------------------------
    # 7. MACHINE LEARNING EXPERIMENTS
    # --------------------------------------------------------

    st.header("7. Machine Learning Experiments")

    st.write("""
    Three classification configurations were tested using the same
    political-term-ablated text and chronological train-test split.
    """)

    st.image(
        "plots/model_comparison.png",
        caption="Accuracy and balanced accuracy of the tested model configurations.",
        width=650
    )

    st.markdown("""
    | Model | Accuracy | Balanced Accuracy |
    |---|---:|---:|
    | Logistic Regression | 95.46% | 96.26% |
    | Multinomial Naive Bayes | 90.40% | 92.19% |
    | Linear SVM | 96.24% | 97.11% |
    """)

    st.write("""
    Linear SVM achieved the highest accuracy and balanced accuracy
    among these tested configurations. It was selected as the final
    classifier.

    Note that the Logistic Regression and Linear SVM configurations
    used balanced class weights, while the Naive Bayes configuration
    used its default class-prior handling. Therefore, this is a
    comparison of tested configurations rather than a perfectly
    isolated comparison of algorithms.
    """)

    st.divider()

    # --------------------------------------------------------
    # 8. FINAL MODEL EVALUATION
    # --------------------------------------------------------

    st.header("8. Final Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "96.24%")
    col2.metric("Balanced Accuracy", "97.11%")
    col3.metric("Test Articles", "23,095")

    st.image(
        "plots/confusion_matrix.png",
        caption="Confusion matrix of the final Linear SVM model.",
        width=550
    )

    st.subheader("Confusion Matrix")

    st.markdown("""
    | Actual / Predicted | Fake News | Real News |
    |---|---:|---:|
    | Fake News | 6,545 | 56 |
    | Real News | 812 | 15,682 |

    - **True Negative (6,545):** Fake articles correctly classified as Fake.
    - **False Positive (56):** Fake articles incorrectly classified as Real.
    - **False Negative (812):** Real articles incorrectly classified as Fake.
    - **True Positive (15,682):** Real articles correctly classified as Real.
    """)

    st.write("""
    The evaluation used a chronological split:

    - Training data: Articles dated before 2017.
    - Testing data: Articles dated in 2017.

    This evaluates the model on a later period than its training data.
    """)

    st.divider()

    # --------------------------------------------------------
    # 9. EVALUATION METRIC EXPLANATIONS
    # --------------------------------------------------------

    st.header("9. Understanding Evaluation Metrics")

    metric_explanations = [
        (
            "Accuracy",
            "The proportion of all predictions that are correct.",
            "Accuracy = (Correct Predictions / Total Predictions)"
        ),
        (
            "Precision",
            "Of all articles predicted as a particular class, how many actually belong to that class.",
            "Precision = TP / (TP + FP)"
        ),
        (
            "Recall",
            "Of all actual articles belonging to a class, how many the model correctly identified.",
            "Recall = TP / (TP + FN)"
        ),
        (
            "F1-score",
            "The harmonic mean of precision and recall. It balances the two measures.",
            "F1 = 2 × (Precision × Recall) / (Precision + Recall)"
        ),
        (
            "Balanced Accuracy",
            "The average recall across the classes. It is useful when class distributions are unequal.",
            "Balanced Accuracy = (Recall of Fake + Recall of Real) / 2"
        ),
        (
            "Confusion Matrix",
            "A table that compares actual classes with predicted classes and shows the types of correct and incorrect predictions.",
            "It displays True Positives, True Negatives, False Positives and False Negatives."
        )
    ]

    for name, explanation, formula in metric_explanations:
        with st.expander(name):
            st.write(explanation)
            st.code(formula)

    st.info(
        "Accuracy alone can be misleading when classes are imbalanced. "
        "Balanced accuracy and per-class precision and recall provide "
        "additional information."
    )

    st.divider()

    # --------------------------------------------------------
    # 10. ERROR ANALYSIS
    # --------------------------------------------------------

    st.header("10. Error Analysis")

    st.write("""
    Error analysis examines incorrect predictions to understand
    where and how the model fails.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Test Errors", "868")
    col2.metric("Fake → Real Errors", "56")
    col3.metric("Real → Fake Errors", "812")

    st.markdown("""
    **Observations from the final model:**

    - The model correctly classified 6,545 Fake articles.
    - It incorrectly classified 56 Fake articles as Real.
    - It incorrectly classified 812 Real articles as Fake.
    - It correctly classified 15,682 Real articles.

    The larger number of Real-to-Fake errors indicates that the
    model's errors are not evenly distributed between the two classes.
    """)

    st.subheader("What did the earlier error investigation reveal?")

    st.markdown("""
    During earlier experiments, many incorrectly classified Real
    articles came from the `worldnews` and `politicsNews` subject
    categories.

    These categories contained only Real-labelled examples in this
    dataset. This makes it difficult to determine whether the model
    learned a general Fake-versus-Real distinction or patterns
    associated with the particular topics and sources.

    Inspection of incorrectly classified articles also showed
    conventional news-reporting patterns, including location
    datelines and source-related writing styles.
    """)

    st.warning(
        "The subject-level observations above come from earlier "
        "experiments. They should not be interpreted as exact "
        "subject-level error counts for the final political-ablation model."
    )

    st.subheader("Why error analysis matters")

    st.markdown("""
    - It identifies which class is being misclassified more often.
    - It helps reveal weaknesses in the dataset.
    - It can expose source, topic or writing-style shortcuts.
    - It helps guide future data collection and model improvements.
    """)

    st.divider()

    # --------------------------------------------------------
    # 11. BIAS AND DATASET INVESTIGATION
    # --------------------------------------------------------

    st.header("11. Bias and Dataset Investigation")

    st.write("""
    A model may achieve high accuracy by learning shortcuts or
    dataset-specific associations instead of general patterns.
    We conducted several experiments to investigate this issue.
    """)

    with st.expander("1. Subject and Label Association"):

        st.write("""
        The dataset had strong associations between subject categories
        and class labels. Fake and Real articles were stored under
        different subject categories.
        """)

        st.markdown("""
        Examples:

        - `News`, `left-news` and `Government News` were associated with Fake labels.
        - `politicsNews` and `worldnews` were associated with Real labels.
        """)

        st.warning(
            "A subject-only decision tree achieved 100% accuracy on "
            "a random split. This demonstrates metadata leakage and "
            "strong label association, not genuine fact verification."
        )

    with st.expander("2. Reuters and Source-Related Patterns"):

        st.write("""
        Earlier investigation found a strong association between
        the presence of the word Reuters and the Real label.

        - 99.83% of Real articles contained Reuters.
        - 98.80% of Fake articles did not contain Reuters.
        """)

        st.write("""
        This suggested that the model could learn source-related
        patterns rather than evaluating the truth of an article.
        """)

    with st.expander("3. Text Artifact Removal Experiment"):

        st.write("""
        We removed selected source and formatting artifacts,
        including Reuters, Read More, Featured Image, Image Via
        and URLs.
        """)

        st.markdown("""
        In an earlier balanced chronological experiment:

        | Version | Accuracy |
        |---|---:|
        | Before artifact removal | 98.22% |
        | After artifact removal | 96.03% |

        The number of Real articles incorrectly classified as Fake
        increased from 386 to 864.
        """)

        st.write("""
        This showed that source and formatting clues contributed
        to the earlier model's performance. Their removal did not
        eliminate all dataset-related patterns.
        """)

    with st.expander("4. Political-Term Ablation Experiment"):

        st.write("""
        We removed selected political terms, including Trump,
        Obama, Hillary, Clinton, Republican, Democrat, GOP,
        President, Senate and Congress.
        """)

        st.markdown("""
        | Experiment | Accuracy | Balanced Accuracy |
        |---|---:|---:|
        | Cleaned balanced SVM | 96.03% | 96.98% |
        | Political-term ablation | 96.24% | 97.11% |
        """)

        st.write("""
        Removing these selected terms did not reduce performance
        in this experiment. It slightly improved the measured
        accuracy and balanced accuracy.

        This does not prove that political bias was eliminated.
        The model can still learn from other correlated vocabulary,
        topics and writing styles.
        """)

    st.subheader("Main conclusion")

    st.info("""
    High accuracy does not necessarily mean that a model has learned
    to identify factual truth. Dataset leakage, source associations,
    topic differences and writing styles can influence predictions.
    These limitations must be considered when interpreting results.
    """)

    st.divider()

    # --------------------------------------------------------
    # 12. TECHNOLOGIES USED
    # --------------------------------------------------------

    st.header("12. Technologies Used")

    st.markdown("""
    | Technology | Purpose |
    |---|---|
    | Python | Programming language |
    | Pandas | Dataset processing |
    | NumPy | Numerical operations |
    | Scikit-learn | Machine learning |
    | TF-IDF | Text feature extraction |
    | Linear SVM | News classification |
    | Matplotlib | Data visualization |
    | Streamlit | Web application |
    | Joblib | Saving and loading the trained model |
    """)

    st.divider()

    # --------------------------------------------------------
    # 13. LIMITATIONS
    # --------------------------------------------------------

    st.header("13. Limitations")

    st.warning("""
    This application is a machine learning text classifier, not an
    independent fact-checking system.
    """)

    st.markdown("""
    - The model learns patterns from the dataset's articles and labels.
    - The dataset contains strong source, topic and writing-style associations with labels.
    - High test accuracy does not guarantee reliable performance on modern news articles or unfamiliar publishers.
    - The chronological test uses articles from 2017, so it does not establish performance on current news.
    - A Real News prediction does not guarantee that every claim in an article is factually correct.
    - The model has not been established as reliable across different languages or news domains.
    """)

    st.divider()

    # --------------------------------------------------------
    # 14. FUTURE SCOPE
    # --------------------------------------------------------

    st.header("14. Future Scope")

    st.markdown("""
    - Use newer and more diverse news datasets.
    - Evaluate the model on completely unseen publishers.
    - Add reliable source and publication information.
    - Explore transformer-based NLP models.
    - Improve performance on articles from different topics and sources.
    - Develop more detailed prediction explanations.
    - Test the system on current articles and different writing styles.
    """)

    st.divider()

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.header("Disclaimer")

    st.caption("""
    This project was developed for educational and academic purposes.
    Predictions should not be treated as proof that an article is
    true or false. Always verify important information using
    reliable and independent sources.
    """)

    st.divider()

    st.caption(
        "Fake News Detection System | Machine Learning Project"
    )
