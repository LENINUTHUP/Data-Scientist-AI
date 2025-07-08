
# 🧠 AI Data Scientist – Automated Machine Learning Pipeline

**Version:** 3.1  
**Author:** Lenin Uthup  
**Last Updated:** July 2025

---

## 📌 Project Overview

**AI Data Scientist** is a powerful, interactive no-code web application that automates the entire machine learning lifecycle—from data upload to model deployment. It supports:

- 🔮 Time Series Forecasting  
- ✅ Predictive Modeling (Regression/Classification)  
- 🧩 Clustering Analysis  
- 📊 Exploratory Data Analysis (EDA)  
- 📥 One-click model export and reporting  

Designed for data professionals, students, and businesses, this tool simplifies complex data science tasks with a clean and user-friendly UI built using **Streamlit**.

---

## 🚀 Features

| 🔧 Module             | 💡 Description                                                                 |
|-----------------------|--------------------------------------------------------------------------------|
| **Data Upload**        | Upload `.csv` files and initialize your ML pipeline                            |
| **Preprocessing**      | Handle missing values, encode features, scale data, and detect outliers        |
| **EDA**                | Explore data distributions, correlations, and trends with charts               |
| **Modeling**           | Train and evaluate regression/classification models                            |
| **Forecasting**        | Predict future values using Prophet & ARIMA with component analysis            |
| **Clustering**         | Segment data using KMeans + Elbow Method                                       |
| **Evaluation**         | Show metrics like MAE, RMSE, Accuracy, F1-Score, ROC AUC                       |
| **Live Simulation**    | Adjust top features in real-time to simulate model predictions                 |
| **Export & Reports**   | Download processed data, models (`.pkl`), and reports (`.pdf`, `.csv`)         |

---

## 📦 Tech Stack

- **Frontend:** Streamlit  
- **ML Models:** scikit-learn, XGBoost, Prophet, ARIMA  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Plotly, Matplotlib, Seaborn  
- **Reporting:** FPDF  

---

## 🧰 Project Structure

```
📁 Data_Scientist_AI/
├── Data_Scientist_AI_3.1.ipynb       # Main Streamlit application
├── requirements.txt                  # All dependencies
├── export/                           # Exported models, data, and reports
├── assets/                           # Custom icons (optional)
└── README.md                         # This file
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/data-scientist-ai.git
cd data-scientist-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run Data_Scientist_AI_3.1.ipynb
```

> 💡 If running inside Jupyter/Colab, convert notebook to `.py`:

```bash
jupyter nbconvert --to script Data_Scientist_AI_3.1.ipynb
```

---

## 📥 Input Requirements

- File Format: `.csv`
- For Forecasting: `ds` = date column, `y` = target
- For Classification/Regression: at least one `target` column
- For Clustering: primarily numerical columns

---

## 📤 Output Files

| 📄 File Name                  | 📘 Description                                    |
|------------------------------|--------------------------------------------------|
| `forecast_data.csv`          | Predicted values with timestamp (for forecasting)|
| `processed_data.csv`         | Cleaned and transformed dataset                  |
| `model.pkl`                  | Trained machine learning model (Pickle format)   |
| `report.pdf`                 | Auto-generated PDF report with metrics & plots   |
| `labeled_cluster_data.csv`   | Cluster results with data points and labels      |

---

## 📈 Visualizations

- 📊 Correlation heatmaps, distribution plots  
- 📉 Forecast trend/seasonal plots (Prophet)  
- 🎯 ROC Curves & classification metrics  
- 📍 Cluster scatter plots with centroids  
- 📃 Gauge plots, pie charts, and bar comparisons  
- 📋 Executive-level PDF reports  

---

## 🧠 Supported ML Tasks

### ✅ Predictive Modeling
- Linear Regression  
- Logistic Regression  
- Random Forest  
- XGBoost  
- Decision Trees  
- Support Vector Machines

### 🔮 Time Series Forecasting
- Prophet (trend, seasonality, holidays)
- ARIMA

### 🧩 Clustering
- KMeans
- Elbow method for `k` optimization
- Cluster profiling

---

## ✨ Live Prediction Simulator

- Select top 5 most important features
- Use sliders to simulate different input values
- Observe prediction values update in real-time
- Visualize classification probabilities or regression outputs

---

## ✅ Use Cases

- 🛒 Retail sales forecasting  
- 📉 Revenue prediction  
- 🧍 Customer segmentation  
- 🧾 Churn classification  
- 🧑‍🏫 ML education and training  
- 🧠 Smart dashboards for business intelligence

---

## 🔐 Data Privacy

- All data stays local or in-session
- No cloud storage or external API calls
- Designed for secure offline use

---

## 👤 Author

**Lenin Uthup**  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/your-profile)

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

- [Streamlit](https://streamlit.io)  
- [scikit-learn](https://scikit-learn.org)  
- [Facebook Prophet](https://facebook.github.io/prophet)  
- [Plotly](https://plotly.com)  
- [FPDF](https://pyfpdf.readthedocs.io/en/latest)
