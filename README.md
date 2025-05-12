# Clustering-Techniques-for-Data-Insights

## Project Overview 

**Project Title : Clustering-Techniques-for-Data-Insights(Unsupervised learning)**
The goal of this project is to group individuals by age and income to identify distinct clusters and identify high-income groups or specific age brackets for marketing or analysis.

## Objectives
1. **Cluster Identification**:
Group individuals into clusters based on shared characteristics (e.g., similar age and income levels).
2. **Target Audience Identification**:
Identify high-income groups or specific age brackets for marketing or analysis.
3. **Anomaly Detection**:
Find individuals who significantly differ from typical group patterns.

## Project Structure

### 1. Importing Libraries
sklearn (from scikit-learn) for machine learning tools
pandas for data manipulation
matplotlib for data visualization.
```python
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
%matplotlib inline
```
%matplotlib inline is a jupyter notebook command which is used to display plots directly in the notebook output cells
KMeans is a sklearn.cluster tool for clustering analysis.
MinMaxScaler is used for scaling features to a specific range, often required for clustering.

### 2. Loading the Dataset
The given dataset is loaded using pandas.read_csv(). This dataset contains data about the names, age and income
```python
df=pd.read_csv("Data5.csv")
df
```

### 3. Ploting the graph
Create a scatter plot to visualize the relationship between two variables: Age and Income.
```python
plt.scatter(df['Age'],df['Income'])
```
It visually shows how individuals in the dataset are distributed across these two dimensions. This can help identify any inherent patterns or trends before applying clustering.

### 4. Data processing
**4.1 Step-1**
```python
km=KMeans(n_clusters=3)
km
```
'n_clusters=3' initializes a K-Means clustering model with 3 clusters. This means the algorithm will attempt to partition the data into 3 distinct groups based on similarity./
**4.2 Step-2**
```python
y_predicted=km.fit_predict(df[['Age','Income']])
y_predicted
```
The fit_predict() method does two things:
Fit: It trains the KMeans model on the provided data (i.e df[['Age', 'Income']]). It finds the centroids of clusters by minimizing the within-cluster variance.
Predict: After fitting the model, it predicts which cluster each data point belongs to. The result is an array where each value corresponds to the cluster label assigned to each data point in df[['Age', 'Income']]./
**4.3 Step-3**
```python
df['cluster']=y_predicted
df
```
Adds the predicted cluster label 'y_predicted' as a new column in the DataFrame df./
**4.4 Step-4**
```python
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
```
This code creates three new DataFrames (df1, df2, and df3) by filtering the original DataFrame df based on the cluster labels./
**4.5 Step-5**
```python
plt.scatter(df1.Age,df1['Income'],color='Green')
plt.scatter(df2.Age,df2['Income'],color='Red')
plt.scatter(df3.Age,df3['Income'],color='Black')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
```
This code creates a scatter plot which visually represent the data points from three clusters (df1, df2, and df3) with Age on the x-axis and Income on the y-axis.
Each cluster group in the DataFrame is assigned a unique color to make it easier to distinguish between the groups./
**4.6 Step-6**
```python
scalar=MinMaxScaler()
scalar.fit(df[['Income']])
df['Income']=scalar.transform(df[['Income']])
scalar.fit(df[['Age']])
df['Age']=scalar.transform(df[['Age']])
df
```
This code uses the MinMaxScaler from scikit-learn to scale the Income and Age columns in the DataFrame df so that their values are transformed into a specific range (i.e between 0 and 1)
This makes the data comparable and ready for algorithm./
**4.7 Step-7 : Processing the data again for plotting the scatter plot with updated values(i.e the values we transformed in the range of 0 and 1 using MinMaxScaler )**
```python
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income']])
y_predicted
df['cluster']=y_predicted
df
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
```
**4.8 Step-8 : Plotting the graph again**
```python
plt.scatter(df1.Age,df1['Income'],color='Green')
plt.scatter(df2.Age,df2['Income'],color='Red')
plt.scatter(df3.Age,df3['Income'],color='Black')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
```
**4.9 Step-9**
```python
km.cluster_centers_
```
This attribute provides the co-ordinates of the centroids of the clusters identified by the algorithm

## Conclusion
The use of clustering techniques, such as KMeans, in this project has proven to be an effective way to uncover hidden patterns and segment data into meaningful groups. By applying clustering to key features like Age and Income, we successfully grouped data points into distinct clusters, allowing for a deeper understanding of the dataset's structure. By identifying natural groupings within data, this project highlights the potential of unsupervised learning to reveal trends, optimize resources, and drive informed decisions.

## Author - Aniket Pal
This project is part of my portfolio, showcasing the machine learning skills essential for data science roles.

-**LinkedIn**: [ www.linkedin.com/in/aniket-pal-098690204 ]
-**Email**: [ aniketspal04@gmail.com ]


