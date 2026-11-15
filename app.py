
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title='AI Impace on Students',page_icon='🤖')
st.title('AI Impact on Students',icon='🤖')

#upload csv file
st.subheader('Upload CSV file')
Uploaded_file=st.file_uploader('Choose a csv file',type='csv')
if Uploaded_file is not None:
  data=pd.read_csv(Uploaded_file)
  st.subheader('Dataset Preview')

  selected_columns = st.multiselect(
      'Select columns to display:',
      options=data.columns.tolist(),
      default=data.columns.tolist()
  )

  st.write(data[selected_columns].head(10))
  data.dropna(axis=0,inplace=True)
  data.drop_duplicates(inplace=True)
  x=data.copy()

  x = x.drop(
  columns=['StudentID', 'SatisfactionLevel'],
  errors='ignore')
 #encode
  if 'Gender' in x.columns:
    x['Gender']=x['Gender'].map({'Male':0,'Female':1})

  # One-Hot Encode remaining categorical columns
  cat_cols = x.select_dtypes(include=['object', 'category']).columns

  x = pd.get_dummies(
        x,
        columns=cat_cols,
        drop_first=True,
        dtype=int
    )

  st.subheader('Encoded Dataset')

  st.write(x.head())
  scaler=StandardScaler()
  x_scaled=scaler.fit_transform(x)
  st.sidebar.subheader('Clustering Options')
  k=st.sidebar.slider('No.of.Cluster(k)',min_value=2,max_value=10,value=6)
  kmeans=KMeans(n_clusters=k,random_state=42, n_init=10)
  clusters=kmeans.fit_predict(x_scaled)
  data['cluster']=clusters
  st.subheader("Number of Students in Each Cluster")
  st.write(data['cluster'].value_counts().sort_index())
  pca=PCA(n_components=2)
  x_pca=pca.fit_transform(x_scaled)
  st.write("Explained variance ratio:", pca.explained_variance_ratio_)
  st.write("Total variance explained:", pca.explained_variance_ratio_.sum())
  data['PCA1']=x_pca[:,0]
  data['PCA2']=x_pca[:,1]
  st.subheader('Customer segments(PCA Projection)')
  plt.figure(figsize=(10,6))
  palette=sns.color_palette('Set2',n_colors=k)
  sns.scatterplot(x='PCA1',y='PCA2',hue='cluster',data=data,palette=palette,s=120,alpha=0.8)
  plt.title('Customer segments(PCA Projection)')
  plt.xlabel('PCA Component 1')
  plt.ylabel('PCA Component 2')
  plt.legend(title='cluster')
  st.pyplot(plt.gcf())
  plt.clf()
  num_col=data.select_dtypes(include=np.number).columns.drop(['StudentID','cluster','PCA1','PCA2'], errors='ignore')
  cluster_summary=data.groupby('cluster')[num_col].mean()
  cluster_summary['count']=data['cluster'].value_counts().sort_index()
  st.subheader('Cluster Summary')
  st.dataframe(cluster_summary.round(2))

  #optonal visualize cluster averages

  st.subheader("Cluster Feature Comparison")
  st.bar_chart(cluster_summary.drop(columns=['count']))
  
  # =====================================
  # STUDENT SATISFACTION CLASSIFICATION
  # =====================================

  st.divider()
  st.header("Student Satisfaction Prediction")

  # Create a separate dataset for classification
  classification_data = data.drop(
      columns=['cluster', 'PCA1', 'PCA2'],
      errors='ignore'
  ).copy()

  # Separate features and target
  X_class = classification_data.drop(
      columns=['StudentID', 'City', 'SatisfactionLevel'],
      errors='ignore'
  )

  y_class = classification_data['SatisfactionLevel']

  # Encode categorical input columns
  X_class = pd.get_dummies(
      X_class,
      drop_first=True,
      dtype=int
  )

  # Encode target labels
  target_encoder = LabelEncoder()
  y_encoded = target_encoder.fit_transform(y_class)

  # Split training and testing data
  X_train, X_test, y_train, y_test = train_test_split(
      X_class,
      y_encoded,
      test_size=0.2,
      random_state=42,
      stratify=y_encoded
  )

  # Scale features
  classification_scaler = StandardScaler()

  X_train_scaled = classification_scaler.fit_transform(X_train)
  X_test_scaled = classification_scaler.transform(X_test)

  st.success("Classification data prepared successfully!")

  st.write("Training records:", len(X_train))
  st.write("Testing records:", len(X_test))
  st.write("Number of features:", X_class.shape[1])
  st.write("Target classes:", list(target_encoder.classes_))
  
  # =====================================
  # TRAIN FOUR CLASSIFICATION MODELS
  # =====================================

  st.subheader("Training Machine Learning Models")

  models = {
      "Logistic Regression": LogisticRegression(max_iter=1000),
      "Decision Tree": DecisionTreeClassifier(random_state=42),
      "Random Forest": RandomForestClassifier(
          n_estimators=100,
          random_state=42
      ),
      "KNN": KNeighborsClassifier(n_neighbors=5)
  }

  # Train each model and store predictions
  trained_models = {}
  predictions = {}

  for name, model in models.items():

      model.fit(X_train_scaled, y_train)

      y_pred = model.predict(X_test_scaled)

      trained_models[name] = model
      predictions[name] = y_pred

  st.success("All four models trained successfully!")
  
  # =====================================
  # MODEL PERFORMANCE COMPARISON
  # =====================================

  st.divider()
  st.subheader("Model Performance Comparison")

  results = []

  for name, y_pred in predictions.items():

      results.append({
          "Model": name,
          "Accuracy": accuracy_score(y_test, y_pred),
          "Precision": precision_score(
              y_test, y_pred, average="macro", zero_division=0
          ),
          "Recall": recall_score(
              y_test, y_pred, average="macro", zero_division=0
          ),
          "F1-score": f1_score(
              y_test, y_pred, average="macro", zero_division=0
          )
      })

  results_df = pd.DataFrame(results)

  st.write("Performance of all four Machine Learning models:")

  st.dataframe(
      results_df.set_index("Model").style.format("{:.2%}"),
      use_container_width=True
  )

  # Model comparison graph

  st.subheader("Model Comparison Graph")

  chart_data = results_df.set_index("Model")

  st.bar_chart(
      chart_data[["Accuracy", "Precision", "Recall", "F1-score"]]
  )



else:
  st.info("Upload a csv file to get  stated")


