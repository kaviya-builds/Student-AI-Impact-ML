
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI Impact on Students",
    page_icon="🤖"
)

st.title("AI Impact on Students", icon="🤖")

# =====================================
# CSV FILE UPLOAD - ORIGINAL STYLE
# =====================================

st.subheader("Upload CSV file")

uploaded_file = st.file_uploader(
    "Choose a csv file",
    type="csv"
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    required_columns = [
        "ImpactonGrades",
        "StudentID",
        "City",
        "SatisfactionLevel"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        st.stop()

    # =====================================
    # DATASET PREVIEW
    # =====================================

    st.header("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Students", len(data))

    with col2:
        st.metric("Total Columns", len(data.columns))

    with col3:
        st.metric("Missing Values", int(data.isnull().sum().sum()))

    with st.expander("View Dataset"):
        selected_columns = st.multiselect(
            "Select columns to display:",
            options=data.columns.tolist(),
            default=data.columns.tolist()
        )
        st.write(data[selected_columns].head(10))

    with st.expander("View Dataset Information"):
        info_df = pd.DataFrame({
            "Column": data.columns,
            "Data Type": data.dtypes.astype(str).values,
            "Missing Values": data.isnull().sum().values
        })
        st.dataframe(info_df)

    # =====================================
    # DATA CLEANING
    # =====================================

    st.header("2. Data Cleaning")

    original_records = len(data)
    duplicate_count = int(data.duplicated().sum())

    clean_data = data.drop_duplicates().copy()

    missing_target = int(
        clean_data["ImpactonGrades"].isnull().sum()
    )

    clean_data = clean_data.dropna(
        subset=["ImpactonGrades"]
    ).copy()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Original Records", original_records)

    with col2:
        st.metric("Records After Cleaning", len(clean_data))

    st.write("Duplicate rows removed:", duplicate_count)
    st.write("Rows with missing target removed:", missing_target)

    # =====================================
    # EXPLORATORY DATA ANALYSIS
    # =====================================

    st.header("3. Exploratory Data Analysis")

    eda_columns = [
        "AIToolUsed",
        "DailyUsageHours",
        "EducationLevel",
        "Purpose",
        "ImpactonGrades",
        "SatisfactionLevel"
    ]

    for column in eda_columns:

        if column not in clean_data.columns:
            continue

        st.subheader(column.replace("_", " "))

        fig, ax = plt.subplots(figsize=(9, 4))

        if pd.api.types.is_numeric_dtype(clean_data[column]):

            sns.histplot(
                clean_data[column],
                kde=True,
                ax=ax
            )

        else:

            sns.countplot(
                data=clean_data,
                x=column,
                order=clean_data[column].value_counts().index,
                ax=ax
            )

            plt.setp(
                ax.get_xticklabels(),
                rotation=30,
                ha="right"
            )

        ax.set_title(column)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # =====================================
    # SUPERVISED MACHINE LEARNING
    # =====================================

    st.divider()

    st.header("4. Supervised Machine Learning")

    st.write(
        "The classification models learn from existing student "
        "records and predict the ImpactonGrades category."
    )

    # Separate features and target

    X = clean_data.drop(
        columns=[
            "StudentID",
            "City",
            "SatisfactionLevel",
            "ImpactonGrades"
        ],
        errors="ignore"
    )

    y = clean_data["ImpactonGrades"].astype(str)

    if y.nunique() < 2:
        st.error("The target column must contain at least two classes.")
        st.stop()

    # Encode target

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_counts = pd.Series(y_encoded).value_counts()

    if class_counts.min() < 2:
        st.error(
            "Each target class needs at least two records "
            "for stratified splitting."
        )
        st.stop()

    # Train-test split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # =====================================
    # DATA PREPROCESSING
    # =====================================

    numeric_columns = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns)
        ],
        sparse_threshold=0
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    st.success("Classification data prepared successfully!")

    st.write("Training records:", len(X_train))
    st.write("Testing records:", len(X_test))
    st.write("Number of features:", X_train_processed.shape[1])
    st.write("Target classes:", list(label_encoder.classes_))

    # =====================================
    # MACHINE LEARNING MODELS
    # =====================================

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),

        "Decision Tree": DecisionTreeClassifier(
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

    # Optional LightGBM

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=100,
            random_state=42,
            verbosity=-1
        )

    except ImportError:
        pass

    # =====================================
    # TRAIN ALL MODELS
    # =====================================

    st.subheader("Training Machine Learning Models")

    trained_models = {}
    predictions = {}
    results = {}
    errors = {}

    for name, model in models.items():

        try:
            model.fit(X_train_processed, y_train)

            y_pred = model.predict(X_test_processed)

            trained_models[name] = model
            predictions[name] = y_pred

            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(
                    y_test, y_pred,
                    average="macro",
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_test, y_pred,
                    average="macro",
                    zero_division=0
                ),
                "F1-score": f1_score(
                    y_test, y_pred,
                    average="macro",
                    zero_division=0
                )
            }

        except Exception as error:
            errors[name] = str(error)

    if not trained_models:
        st.error("No models could be trained.")
        st.stop()

    st.success(
        f"{len(trained_models)} models trained successfully!"
    )

    if errors:
        with st.expander("Models with errors"):
            for name, error in errors.items():
                st.write(f"{name}: {error}")

    # =====================================
    # MODEL PERFORMANCE COMPARISON
    # =====================================

    st.divider()

    st.subheader("Model Performance Comparison")

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"

    st.dataframe(
        results_df.style.format("{:.2%}"),
        use_container_width=True
    )

    st.subheader("Model Comparison Graph")

    st.bar_chart(results_df)

    # =====================================
    # SIDEBAR MODEL SELECTION
    # =====================================

    st.sidebar.subheader("Machine Learning Models")

    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(trained_models.keys())
    )

    # =====================================
    # SELECTED MODEL VISUALIZATION
    # =====================================

    st.divider()

    st.subheader(
        selected_model_name + " Visualization"
    )

    selected_model = trained_models[selected_model_name]

    feature_names = preprocessor.get_feature_names_out()

    if selected_model_name == "Logistic Regression":

        importance = np.mean(
            np.abs(selected_model.coef_),
            axis=0
        )

        importance_df = pd.Series(
            importance,
            index=feature_names
        ).sort_values(ascending=False).head(10)

        st.write("Top 10 Feature Coefficients")
        st.bar_chart(importance_df)

    elif selected_model_name == "Decision Tree":

        fig, ax = plt.subplots(figsize=(18, 9))

        plot_tree(
            selected_model,
            max_depth=3,
            feature_names=feature_names,
            class_names=label_encoder.classes_,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

    elif selected_model_name == "Random Forest":

        importance_df = pd.Series(
            selected_model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False).head(10)

        st.write("Top 10 Feature Importances")
        st.bar_chart(importance_df)

    elif selected_model_name == "KNN":

        distances, indices = selected_model.kneighbors(
            X_test_processed[:1]
        )

        neighbor_df = pd.DataFrame({
            "Neighbor": range(1, len(indices[0]) + 1),
            "Distance": distances[0],
            "Class": label_encoder.inverse_transform(
                y_train[indices[0]]
            )
        })

        st.write("Nearest Neighbors")
        st.dataframe(neighbor_df)

        st.bar_chart(
            neighbor_df.set_index("Neighbor")["Distance"]
        )

    elif selected_model_name == "SVM":

        support_df = pd.DataFrame({
            "Class": label_encoder.classes_,
            "Support Vectors": selected_model.n_support_
        })

        st.write("Support Vectors per Class")
        st.bar_chart(support_df.set_index("Class"))

    elif selected_model_name == "Naive Bayes":

        probabilities = selected_model.predict_proba(
            X_test_processed
        )

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.hist(
            probabilities.max(axis=1),
            bins=15,
            edgecolor="black"
        )

        ax.set_title("Predicted Probability Distribution")
        ax.set_xlabel("Maximum Predicted Probability")
        ax.set_ylabel("Number of Students")

        st.pyplot(fig)
        plt.close(fig)

    elif selected_model_name == "LightGBM":

        importance_df = pd.Series(
            selected_model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False).head(10)

        st.write("Top 10 Feature Importances")
        st.bar_chart(importance_df)

    # =====================================
    # FOOTER
    # =====================================

    st.divider()

    st.caption(
        "AI Impact on Students | Machine Learning Project"
    )

    st.caption(
        "Predictions show patterns in the recorded dataset. "
        "They do not prove that AI usage causes changes in grades."
    )

else:
    st.info("Upload a csv file to get started.")
