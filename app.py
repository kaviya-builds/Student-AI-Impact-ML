
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Impact on Students",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Impact on Students")
st.write(
    "Explore student AI usage patterns and predict "
    "their recorded impact on grades using machine learning."
)

st.sidebar.title("Dashboard Settings")


# ==================================================
# 2. DATASET UPLOAD
# ==================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload Student Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Please upload AI_Student_Life_India_2026.csv "
        "using the sidebar."
    )
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read the CSV file: {error}")
    st.stop()


# ==================================================
# 3. DATA VALIDATION
# ==================================================

required_columns = [
    "StudentID",
    "Age",
    "Gender",
    "EducationLevel",
    "City",
    "AIToolUsed",
    "DailyUsageHours",
    "Purpose",
    "ImpactonGrades",
    "SatisfactionLevel"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"The dataset is missing these columns: {missing_columns}"
    )
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()


# ==================================================
# 4. DATASET OVERVIEW
# ==================================================

st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Students", len(df))
col2.metric("Total Columns", len(df.columns))
col3.metric("Missing Values", int(df.isnull().sum().sum()))
col4.metric("Duplicate Rows", int(df.duplicated().sum()))

with st.expander("View Dataset"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("View Dataset Information"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })
    st.dataframe(info_df, use_container_width=True)


# ==================================================
# 5. DATA CLEANING
# ==================================================

st.header("2. Data Cleaning")

original_count = len(df)

df = df.drop_duplicates().copy()

# Remove records without a target label
df = df.dropna(subset=["ImpactonGrades"]).copy()

cleaned_count = len(df)

col1, col2 = st.columns(2)

col1.metric("Original Records", original_count)
col2.metric("Records After Cleaning", cleaned_count)

st.write(
    f"Duplicate rows removed: {original_count - len(df.dropna(subset=['ImpactonGrades']))}"
)

if len(df) < 10:
    st.error("Not enough records to train machine learning models.")
    st.stop()

if df["ImpactonGrades"].nunique() < 2:
    st.error("ImpactonGrades must contain at least two different classes.")
    st.stop()


# ==================================================
# 6. EXPLORATORY DATA ANALYSIS
# ==================================================

st.header("3. Exploratory Data Analysis")

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.subheader("AI Tools Used")

    fig, ax = plt.subplots(figsize=(8, 4))

    df["AIToolUsed"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("AI Tool")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of AI Tools")

    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with eda_col2:
    st.subheader("Daily AI Usage Hours")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x="DailyUsageHours",
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title("Daily AI Usage Distribution")
    ax.set_xlabel("Daily Usage Hours")

    st.pyplot(fig)
    plt.close(fig)


eda_col3, eda_col4 = st.columns(2)

with eda_col3:
    st.subheader("Education Level")

    fig, ax = plt.subplots(figsize=(8, 4))

    df["EducationLevel"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Students by Education Level")
    ax.set_xlabel("Education Level")
    ax.set_ylabel("Number of Students")

    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with eda_col4:
    st.subheader("Purpose of AI Usage")

    fig, ax = plt.subplots(figsize=(8, 4))

    df["Purpose"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Purpose of Using AI")
    ax.set_xlabel("Purpose")
    ax.set_ylabel("Number of Students")

    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


eda_col5, eda_col6 = st.columns(2)

with eda_col5:
    st.subheader("Impact on Grades")

    fig, ax = plt.subplots(figsize=(8, 4))

    df["ImpactonGrades"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Impact on Grades Distribution")
    ax.set_xlabel("Grade Impact")
    ax.set_ylabel("Number of Students")

    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with eda_col6:
    st.subheader("Satisfaction Level")

    fig, ax = plt.subplots(figsize=(8, 4))

    df["SatisfactionLevel"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Student Satisfaction Distribution")
    ax.set_xlabel("Satisfaction Level")
    ax.set_ylabel("Number of Students")

    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# 7. SUPERVISED MACHINE LEARNING
# ==================================================

st.header("4. Supervised Machine Learning")

st.write(
    "The classification models learn from existing student "
    "records and predict the ImpactonGrades category."
)

TARGET = "ImpactonGrades"

# Input features
X = df.drop(
    columns=[
        "StudentID",
        "City",
        "SatisfactionLevel",
        "ImpactonGrades"
    ],
    errors="ignore"
)

y = df[TARGET]


# ==================================================
# 8. FEATURE PREPROCESSING
# ==================================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# ==================================================
# 9. TRAIN-TEST SPLIT
# ==================================================

class_counts = y.value_counts()

stratify_target = (
    y if len(class_counts) > 1
    and class_counts.min() >= 2
    else None
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=stratify_target
)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()


# ==================================================
# 10. DEFINE SEVEN CLASSIFICATION MODELS
# ==================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "Naive Bayes": GaussianNB()
}

if LIGHTGBM_AVAILABLE:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )


# ==================================================
# 11. TRAIN AND EVALUATE MODELS
# ==================================================

results = []
trained_models = {}

for name, model in models.items():

    try:
        model.fit(X_train_processed, y_train)

        predictions = model.predict(X_test_processed)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        trained_models[name] = model

    except Exception as error:
        st.warning(f"{name} could not be trained: {error}")


if not results:
    st.error("No machine learning models could be trained.")
    st.stop()

results_df = pd.DataFrame(results)


# ==================================================
# 12. OVERALL MODEL COMPARISON
# ==================================================

st.header("5. Model Performance Comparison")

st.subheader("Performance Metrics")

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}"
    }),
    use_container_width=True
)


# ==================================================
# 13. MODEL COMPARISON GRAPH
# ==================================================

st.subheader("Classification Model Comparison Graph")

comparison = results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
]

fig, ax = plt.subplots(figsize=(12, 5))

comparison.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Comparison of Machine Learning Models")
ax.set_xlabel("Machine Learning Models")
ax.set_ylabel("Performance Score")
ax.set_ylim(0, 1)

plt.xticks(rotation=30)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


# ==================================================
# 14. MODEL SELECTION
# ==================================================

st.header("6. Individual Model Visualization")

selected_model = st.sidebar.selectbox(
    "Choose Classification Model",
    list(trained_models.keys())
)

st.subheader(selected_model)

selected_model_object = trained_models[selected_model]


# ==================================================
# 15. LOGISTIC REGRESSION
# ==================================================

if selected_model == "Logistic Regression":

    st.write("Feature Coefficient Importance")

    importance = np.abs(
        selected_model_object.coef_
    ).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(
        "Importance",
        ascending=False
    ).head(15)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title("Logistic Regression Feature Importance")

    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# 16. DECISION TREE
# ==================================================

elif selected_model == "Decision Tree":

    st.write("Decision Tree Structure")

    fig, ax = plt.subplots(figsize=(20, 10))

    plot_tree(
        selected_model_object,
        feature_names=feature_names,
        class_names=[
            str(value)
            for value in selected_model_object.classes_
        ],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=ax
    )

    ax.set_title("Decision Tree Visualization")

    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# 17. RANDOM FOREST
# ==================================================

elif selected_model == "Random Forest":

    st.write("Random Forest Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": selected_model_object.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    ).head(15)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title("Random Forest Feature Importance")

    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "Number of decision trees:",
        len(selected_model_object.estimators_)
    )


# ==================================================
# 18. KNN
# ==================================================

elif selected_model == "KNN":

    st.write("Nearest Neighbors Visualization")

    distances, indices = selected_model_object.kneighbors(
        X_test_processed[0:1]
    )

    neighbor_classes = y_train.iloc[
        indices[0]
    ].values

    neighbor_df = pd.DataFrame({
        "Neighbor": [
            f"Neighbor {i + 1}"
            for i in range(len(indices[0]))
        ],
        "Distance": distances[0],
        "Class": neighbor_classes
    })

    st.write("Nearest training students:")

    st.dataframe(
        neighbor_df,
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=neighbor_df,
        x="Neighbor",
        y="Distance",
        ax=ax
    )

    ax.set_title("Distance to Nearest Neighbors")
    ax.set_ylabel("Distance")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "KNN predicts a class using the labels of nearby training samples. "
        "A smaller distance indicates greater similarity."
    )


# ==================================================
# 19. SVM
# ==================================================

elif selected_model == "SVM":

    st.write("Support Vector Machine Visualization")

    support_counts = selected_model_object.n_support_

    support_df = pd.DataFrame({
        "Class": [
            str(value)
            for value in selected_model_object.classes_
        ],
        "Support Vectors": support_counts
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=support_df,
        x="Class",
        y="Support Vectors",
        ax=ax
    )

    ax.set_title("Support Vectors by Class")

    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "Total support vectors:",
        int(support_counts.sum())
    )

    st.caption(
        "Support vectors are training samples that help define "
        "the SVM decision boundary."
    )


# ==================================================
# 20. NAIVE BAYES
# ==================================================

elif selected_model == "Naive Bayes":

    st.write("Naive Bayes Prediction Probability")

    probabilities = selected_model_object.predict_proba(
        X_test_processed
    )

    probability_df = pd.DataFrame(
        probabilities,
        columns=[
            str(value)
            for value in selected_model_object.classes_
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    probability_df.plot(
        kind="hist",
        bins=20,
        alpha=0.5,
        ax=ax
    )

    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Prediction Probability")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# 21. LIGHTGBM
# ==================================================

elif selected_model == "LightGBM":

    st.write("LightGBM Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": selected_model_object.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    ).head(15)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title("LightGBM Feature Importance")

    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# 22. PROJECT SUMMARY
# ==================================================

st.divider()

st.header("7. Project Summary")

st.write("""
### Learning Approach: Supervised Machine Learning

This project uses seven classification algorithms to predict
the recorded impact of AI usage on student grades.

### Prediction Target

ImpactonGrades

### Algorithms

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors
5. Support Vector Machine
6. Naive Bayes
7. LightGBM

The models are evaluated using Accuracy, Precision, Recall
and F1-score.

The predictions show patterns in the available dataset.
They do not prove that AI usage causes changes in grades.
""")

st.success("AI Impact on Students Dashboard Loaded Successfully!")

st.caption(
    "Machine Learning Academic Project | Streamlit Dashboard"
)
