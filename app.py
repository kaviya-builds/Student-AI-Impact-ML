import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
    f1_score,
    confusion_matrix,
    classification_report
)

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Impact on Students",
    page_icon="🤖",
    layout="wide"
)

st.title("🎓 AI Impact on Students")

st.write(
    "Explore student AI usage patterns and predict their "
    "recorded impact on grades using machine learning."
)


# --------------------------------------------------
# 2. CSV FILE UPLOAD
# --------------------------------------------------

st.subheader("Upload CSV file")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv"
)


if uploaded_file is not None:

    # Read CSV
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read the CSV file: {e}")
        st.stop()

    if data.empty:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    # --------------------------------------------------
    # 3. DATASET OVERVIEW
    # --------------------------------------------------

    st.header("1. Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", len(data))
    col2.metric("Total Columns", len(data.columns))
    col3.metric("Missing Values", int(data.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(data.duplicated().sum()))

    with st.expander("View Dataset"):
        selected_columns = st.multiselect(
            "Select columns to display",
            data.columns.tolist(),
            default=data.columns.tolist()
        )

        if selected_columns:
            st.dataframe(
                data[selected_columns].head(10),
                use_container_width=True
            )
        else:
            st.info("Select at least one column.")

    with st.expander("View Dataset Information"):
        info_df = pd.DataFrame({
            "Column": data.columns,
            "Data Type": data.dtypes.astype(str).values,
            "Missing Values": data.isnull().sum().values,
            "Unique Values": data.nunique().values
        })

        st.dataframe(info_df, use_container_width=True)


    # --------------------------------------------------
    # 4. DATA CLEANING
    # --------------------------------------------------

    st.header("2. Data Cleaning")

    original_records = len(data)

    cleaned_data = data.drop_duplicates().copy()
    cleaned_data = cleaned_data.dropna(subset=["ImpactonGrades"])

    cleaned_records = len(cleaned_data)

    col1, col2, col3 = st.columns(3)

    col1.metric("Original Records", original_records)
    col2.metric("Records After Cleaning", cleaned_records)
    col3.metric(
        "Records Removed",
        original_records - cleaned_records
    )

    st.write(
        "Duplicate rows and rows without the target value "
        "ImpactonGrades are removed for model training."
    )


    # --------------------------------------------------
    # 5. EXPLORATORY DATA ANALYSIS
    # --------------------------------------------------

    st.header("3. Exploratory Data Analysis")

    # AI Tools Used
    if "AIToolUsed" in cleaned_data.columns:

        st.subheader("AI Tools Used")

        fig, ax = plt.subplots(figsize=(9, 4))

        sns.countplot(
            data=cleaned_data,
            x="AIToolUsed",
            order=cleaned_data["AIToolUsed"].value_counts().index,
            ax=ax
        )

        ax.set_xlabel("AI Tools")
        ax.set_ylabel("Number of Students")
        ax.tick_params(axis="x", rotation=30)

        st.pyplot(fig)
        plt.close(fig)


    # Daily Usage Hours
    if "DailyUsageHours" in cleaned_data.columns:

        st.subheader("Daily AI Usage Hours")

        fig, ax = plt.subplots(figsize=(9, 4))

        sns.histplot(
            cleaned_data["DailyUsageHours"].dropna(),
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Daily Usage Hours")
        ax.set_ylabel("Number of Students")

        st.pyplot(fig)
        plt.close(fig)


    # Education Level
    if "EducationLevel" in cleaned_data.columns:

        st.subheader("Education Level")

        fig, ax = plt.subplots(figsize=(9, 4))

        sns.countplot(
            data=cleaned_data,
            x="EducationLevel",
            order=cleaned_data["EducationLevel"].value_counts().index,
            ax=ax
        )

        ax.set_xlabel("Education Level")
        ax.set_ylabel("Number of Students")
        ax.tick_params(axis="x", rotation=30)

        st.pyplot(fig)
        plt.close(fig)


    # Purpose of AI Usage
    if "Purpose" in cleaned_data.columns:

        st.subheader("Purpose of AI Usage")

        fig, ax = plt.subplots(figsize=(9, 4))

        sns.countplot(
            data=cleaned_data,
            x="Purpose",
            order=cleaned_data["Purpose"].value_counts().index,
            ax=ax
        )

        ax.set_xlabel("Purpose")
        ax.set_ylabel("Number of Students")
        ax.tick_params(axis="x", rotation=30)

        st.pyplot(fig)
        plt.close(fig)


    # Impact on Grades
    st.subheader("Impact on Grades")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=cleaned_data,
        x="ImpactonGrades",
        order=cleaned_data["ImpactonGrades"].value_counts().index,
        ax=ax
    )

    ax.set_xlabel("Impact on Grades")
    ax.set_ylabel("Number of Students")
    ax.tick_params(axis="x", rotation=20)

    st.pyplot(fig)
    plt.close(fig)


    # Satisfaction Level
    if "SatisfactionLevel" in cleaned_data.columns:

        st.subheader("Satisfaction Level")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.countplot(
            data=cleaned_data,
            x="SatisfactionLevel",
            order=cleaned_data["SatisfactionLevel"].value_counts().index,
            ax=ax
        )

        ax.set_xlabel("Satisfaction Level")
        ax.set_ylabel("Number of Students")
        ax.tick_params(axis="x", rotation=20)

        st.pyplot(fig)
        plt.close(fig)


    # --------------------------------------------------
    # 6. SUPERVISED MACHINE LEARNING
    # --------------------------------------------------

    st.header("4. Supervised Machine Learning")

    st.write(
        "The classification models learn from existing student "
        "records and predict the ImpactonGrades category."
    )

    target = "ImpactonGrades"

    if target not in cleaned_data.columns:
        st.error(
            "The uploaded CSV must contain the "
            "'ImpactonGrades' target column."
        )
        st.stop()

    if cleaned_data[target].nunique() < 2:
        st.error(
            "At least two different ImpactonGrades categories "
            "are required for classification."
        )
        st.stop()

    # Features and target
    drop_columns = [
        "ImpactonGrades",
        "StudentID",
        "City",
        "SatisfactionLevel"
    ]

    X = cleaned_data.drop(
        columns=[
            col for col in drop_columns
            if col in cleaned_data.columns
        ]
    )

    y = cleaned_data[target].astype(str)

    if X.empty:
        st.error("No feature columns are available for training.")
        st.stop()

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Identify numerical and categorical columns
    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # Preprocessing for numerical features
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric_columns:
        transformers.append(
            ("num", numeric_transformer, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("cat", categorical_transformer, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0
    )

    # Train and test split
    class_counts = pd.Series(y_encoded).value_counts()

    if class_counts.min() < 2:
        st.error(
            "Each ImpactonGrades category needs at least "
            "two records for a stratified train/test split."
        )
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    # --------------------------------------------------
    # 7. MACHINE LEARNING MODELS
    # --------------------------------------------------

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

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

    trained_models = {}
    results = []

    with st.spinner("Training machine learning models..."):

        for model_name, model in models.items():

            try:
                pipeline = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model)
                ])

                pipeline.fit(X_train, y_train)

                predictions = pipeline.predict(X_test)

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                trained_models[model_name] = pipeline

                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1
                })

            except Exception as e:
                st.warning(
                    f"{model_name} could not be trained: {e}"
                )

    if not trained_models:
        st.error("None of the classification models could be trained.")
        st.stop()

    results_df = pd.DataFrame(results)

    # --------------------------------------------------
    # 8. MODEL COMPARISON
    # --------------------------------------------------

    st.subheader("Overall Model Comparison")

    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}"
        }),
        use_container_width=True
    )

    # Comparison chart
    st.subheader("Model Performance Comparison")

    comparison = results_df.set_index("Model")

    fig, ax = plt.subplots(figsize=(12, 5))

    comparison.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Performance Score")
    ax.set_xlabel("Machine Learning Models")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=25)

    st.pyplot(fig)
    plt.close(fig)


    # --------------------------------------------------
    # 9. SIDEBAR MODEL SELECTION
    # --------------------------------------------------

    st.sidebar.title("🤖 Model Selection")

    st.sidebar.write(
        "Choose a machine learning model to view its results."
    )

    selected_model = st.sidebar.selectbox(
        "Choose a Model",
        list(trained_models.keys())
    )

    st.sidebar.divider()

    st.sidebar.write("Available Models")

    for model_name in trained_models:
        st.sidebar.write("✓ " + model_name)


    # --------------------------------------------------
    # 10. SELECTED MODEL VISUALIZATION
    # --------------------------------------------------

    st.header("5. Selected Model Analysis")

    st.subheader(selected_model)

    selected_pipeline = trained_models[selected_model]

    selected_predictions = selected_pipeline.predict(X_test)

    selected_accuracy = accuracy_score(
        y_test,
        selected_predictions
    )

    st.metric(
        "Selected Model Accuracy",
        f"{selected_accuracy:.2%}"
    )

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        selected_predictions
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Actual Values")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)


    # Classification Report
    with st.expander("View Classification Report"):

        report = classification_report(
            y_test,
            selected_predictions,
            labels=np.arange(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            zero_division=0,
            output_dict=True
        )

        st.dataframe(
            pd.DataFrame(report).transpose(),
            use_container_width=True
        )


    # --------------------------------------------------
    # 11. MODEL-SPECIFIC VISUALIZATIONS
    # --------------------------------------------------

    classifier = selected_pipeline.named_steps["classifier"]

    feature_names = selected_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    if selected_model == "Logistic Regression":

        st.subheader("Logistic Regression Coefficients")

        coefficients = classifier.coef_

        if coefficients.shape[0] == 1:
            importance = np.abs(coefficients[0])
        else:
            importance = np.mean(
                np.abs(coefficients),
                axis=0
            )

        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(
            by="Importance",
            ascending=False
        ).head(15)

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=feature_importance,
            x="Importance",
            y="Feature",
            ax=ax
        )

        ax.set_title("Top Logistic Regression Coefficients")

        st.pyplot(fig)
        plt.close(fig)


    elif selected_model == "Decision Tree":

        st.subheader("Decision Tree Visualization")

        fig, ax = plt.subplots(figsize=(18, 8))

        plot_tree(
            classifier,
            feature_names=feature_names,
            class_names=label_encoder.classes_,
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=7,
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)


    elif selected_model == "Random Forest":

        st.subheader("Random Forest Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": classifier.feature_importances_
        }).sort_values(
            by="Importance",
            ascending=False
        ).head(15)

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            ax=ax
        )

        ax.set_title("Top Random Forest Features")

        st.pyplot(fig)
        plt.close(fig)


    elif selected_model == "KNN":

        st.subheader("KNN Model Information")

        st.write(
            f"Number of nearest neighbors (K): "
            f"{classifier.n_neighbors}"
        )

        st.write(
            "KNN predicts a student's impact category by "
            "looking at the nearest training records."
        )

        st.write(
            f"Training records used: {len(X_train)}"
        )


    elif selected_model == "SVM":

        st.subheader("Support Vector Machine Information")

        st.write(
            f"Number of support vectors: "
            f"{classifier.n_support_.sum()}"
        )

        st.write(
            "SVM finds decision boundaries that separate "
            "the different ImpactonGrades categories."
        )


    elif selected_model == "Naive Bayes":

        st.subheader("Naive Bayes Prediction Probabilities")

        probabilities = selected_pipeline.predict_proba(X_test)

        fig, ax = plt.subplots(figsize=(9, 4))

        for i, class_name in enumerate(label_encoder.classes_):

            sns.histplot(
                probabilities[:, i],
                bins=20,
                kde=True,
                label=class_name,
                ax=ax,
                alpha=0.4
            )

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Number of Students")
        ax.legend(title="Impact on Grades")

        st.pyplot(fig)
        plt.close(fig)


    elif selected_model == "LightGBM":

        st.subheader("LightGBM Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": classifier.feature_importances_
        }).sort_values(
            by="Importance",
            ascending=False
        ).head(15)

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            ax=ax
        )

        ax.set_title("Top LightGBM Features")

        st.pyplot(fig)
        plt.close(fig)


    # --------------------------------------------------
    # 12. FOOTER
    # --------------------------------------------------

    st.divider()

    st.caption(
        "AI Impact on Students | Supervised Machine Learning "
        "Classification Project"
    )

    st.caption(
        "Model predictions show patterns in the uploaded data. "
        "They do not prove that AI usage causes changes in grades."
    )


else:

    st.info("Please upload a CSV file to get started.")

