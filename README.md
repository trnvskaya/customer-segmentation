# Customer Segmentation Clustering App

This project provides an interactive web application to predict customer cluster based on sales data. The [dataset used](https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering?resource=download) for this task is from an open data source on Kaggle.

---
## **Project Overview**
Using the open-source Kaggle dataset, I downloaded and explored the data. During analysis, some data quality issues were identified, such as null values and unrealistic entries (e.g., customer age > 120 years).

For clustering customers, I selected the most relevant features, applied the **Elbow Method** to determine the optimal number of clusters, and used **KMeans** to segment the customers.

A **Streamlit app** was created to allow users to input customer information, predict their cluster, and **visualize** customer segments **in 2D using PCA**. The app is hosted on Streamlit Community Cloud and ready for interactive use.

---

## **Technologies Used**
- Data processing and exploratory data analysis: Python (NumPy, Pandas, Plotly)
- Data visualizations (Plotly)
- Clustering using KMeans algorithm (scikit-learn)
- Interactive web app (streamlit)
- Deployment: Streamlit Community Cloud
---

## **Project Structure**

```
customer_segmentation/
├── main.py                  # Streamlit web app source code
├── data/
│   ├── customer_segmentation.csv  # Raw Kaggle dataset
│   └── customers_clustered.csv    # Preprocessed and cleaned data
├── analysis.ipynb                 # Jupyter notebook with data preprocessing and analysis
├── scaler.pkl                     # Trained data scaler
├── kmeans_model.pkl               # Trained KMeans model
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation

```

## **Access Online**

The app is deployed on Streamlit Community Cloud and can be accessed here:

[https://customer-segmentation-clusters.streamlit.app/](https://customer-segmentation-clusters.streamlit.app/)

---


## **License**

This project is open-source and available under the MIT License.
