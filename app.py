
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
  columns=['StudentID', 'City', 'SatisfactionLevel'],
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
else:
  st.info("Upload a csv file to get  stated")


