import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.decomposition import PCA


kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

data = pd.read_csv("data/customers_clustered.csv") 

st.title("Customer Segmentation App")
st.write("Enter customer details to predict their segment:")


age = st.number_input("Age", min_value=18, max_value=100, value=42)
income = st.number_input("Income", min_value=0, max_value=1000000, value=50000)
total_spending = st.number_input("Total Spending", min_value=0, max_value=100000, value=5000)
num_instore_purchases = st.number_input("Number of In-store Purchases", min_value=0, max_value=100, value=5)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=5)
num_web_visits = st.number_input("Number of Web Visits", min_value=0, max_value=1000, value=7)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=50)


input_data = pd.DataFrame({
    "Income": [income],
    "Age": [age],
    "Recency": [recency],
    "Total_spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_instore_purchases],
    "NumWebVisitsMonth": [num_web_visits],
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.write(f"The predicted customer segment is: **Cluster {cluster}**")


    cluster_info = {
        0: "Low income, low total spending, *infrequent buyers with lots of web visits*",
        1: "High income, high total spendings, high number of in-store purchases - *premium buyers, possibly loyal customers*",
        2: "High income, older in age than Cluster 1, but similar amount of total spending, high number of in-store purchases - possibly *loyal older premium customers*",
        3: "Average income, older age and moderate spendings with almost same number of in-store and web purchases - *balanced customers*",
        4: "Low income, but lowest recency too, *frequent buyers of cheaper products with lots of web visits, possibly new customers*",
        5: "Mid-to-high income, low recency and high total spendings - *loyal highly engaged customers*"
    }
    st.markdown(f"**Cluster {cluster} insight:** {cluster_info[cluster]}")


    features = ["Income","Age","Recency","Total_spending",
                "NumWebPurchases","NumStorePurchases","NumWebVisitsMonth"]

    X_scaled = scaler.transform(data[features])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    data["PCA1"], data["PCA2"] = X_pca[:, 0], X_pca[:, 1]


    input_pca = pca.transform(input_scaled)


    fig = px.scatter(
        data,
        x="PCA1", y="PCA2",
        color="Cluster",
        title="Customer Segments (PCA Projection)",
        opacity=0.6
    )


    fig.add_scatter(
        x=[input_pca[0, 0]],
        y=[input_pca[0, 1]],
        mode="markers",
        marker=dict(color="red", size=15, symbol="star"),
        name="New Customer"
    )

    st.plotly_chart(fig, use_container_width=True)
