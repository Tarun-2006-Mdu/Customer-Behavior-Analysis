from flask import Flask, render_template, request, redirect
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load and clean data
    df = pd.read_csv(filepath)

    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df[column].fillna(df[column].median(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)

    df.drop_duplicates(inplace=True)
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')

    # Create customer summary
    summary = df.groupby('Customer ID').agg({
        'Total Purchase Amount': ['sum', 'count'],
        'Product Category': 'nunique',
        'Customer Age': 'mean',
        'Churn': 'max'
    }).reset_index()
    summary.columns = ['Customer ID', 'total_purchase', 'purchase_frequency', 'unique_categories', 'avg_age', 'churn']

    # Clustering
    features = summary[['total_purchase', 'purchase_frequency', 'unique_categories', 'avg_age', 'churn']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    summary['cluster'] = kmeans.fit_predict(scaled)

    # Save summary
    summary.to_csv('customer_summary.csv', index=False)

    # Stats for dashboard
    total_customers = len(summary)
    avg_purchase = round(df['Total Purchase Amount'].mean(), 2)
    popular_category = df['Product Category'].mode()[0]
    churn_rate = f"{df['Churn'].mean() * 100:.2f}%"

    # Graph data (example)
    labels1 = summary['Customer ID'].astype(str).tolist()[:10]
    values1 = summary['total_purchase'].tolist()[:10]
    churned = int(df['Churn'].sum())
    active = int(len(df) - churned)

    return render_template('dashboard.html',
                           total_customers=total_customers,
                           avg_purchase=avg_purchase,
                           popular_category=popular_category,
                           churn_rate=churn_rate,
                           labels1=labels1,
                           values1=values1,
                           churned=churned,
                           active=active)

if __name__ == '__main__':
    app.run(debug=True)
