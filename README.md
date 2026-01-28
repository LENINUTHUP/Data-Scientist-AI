
#  AI Data Scientist – Automated Machine Learning Pipeline

**Version:** 3.1  
**Author:** Lenin Uthup  
**Last Updated:** July 2025

 [Tap to view the Live Website ](https://data-scientist-ai-pipeline.streamlit.app/)

---

##  Project Overview

**AI Data Scientist** is a powerful, interactive no-code web application that automates the entire machine learning lifecycle—from data upload to model deployment. It supports:

-  Time Series Forecasting  
-  Predictive Modeling (Regression/Classification)  
-  Clustering Analysis  
-  Exploratory Data Analysis (EDA)  
-  One-click model export and reporting  

Designed for data professionals, students, and businesses, this tool simplifies complex data science tasks with a clean and user-friendly UI built using **Streamlit**.

---

##  Features

|  Module             |  Description                                                                 |
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

##  Tech Stack

- **Frontend:** Streamlit  
- **ML Models:** scikit-learn, XGBoost, Prophet, ARIMA  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Plotly, Matplotlib, Seaborn  
- **Reporting:** FPDF  

---

##  Project Structure

```
 Data_Scientist_AI/
├── Data_Scientist_AI_3.1.ipynb       # Main Streamlit application
├── requirements.txt                  # All dependencies
├── export/                           # Exported models, data, and reports
├── assets/                           # Custom icons (optional)
└── README.md                         # This file
```

---

##  Installation & Setup

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

>  If running inside Jupyter/Colab, convert notebook to `.py`:

```bash
jupyter nbconvert --to script Data_Scientist_AI_3.1.ipynb
```

---

##  Input Requirements

- File Format: `.csv`
- For Forecasting: `ds` = date column, `y` = target
- For Classification/Regression: at least one `target` column
- For Clustering: primarily numerical columns

---

##  Output Files

|  File Name                  |  Description                                    |
|------------------------------|--------------------------------------------------|
| `forecast_data.csv`          | Predicted values with timestamp (for forecasting)|
| `processed_data.csv`         | Cleaned and transformed dataset                  |
| `model.pkl`                  | Trained machine learning model (Pickle format)   |
| `report.pdf`                 | Auto-generated PDF report with metrics & plots   |
| `labeled_cluster_data.csv`   | Cluster results with data points and labels      |

---

##  Visualizations

-  Correlation heatmaps, distribution plots  
-  Forecast trend/seasonal plots (Prophet)  
-  ROC Curves & classification metrics  
-  Cluster scatter plots with centroids  
-  Gauge plots, pie charts, and bar comparisons  
-  Executive-level PDF reports  

---

##  Supported ML Tasks

###  Predictive Modeling
- Linear Regression  
- Logistic Regression  
- Random Forest  
- XGBoost  
- Decision Trees  
- Support Vector Machines

###  Time Series Forecasting
- Prophet (trend, seasonality, holidays)
- ARIMA

###  Clustering
- KMeans
- Elbow method for `k` optimization
- Cluster profiling

---

##  Live Prediction Simulator

- Select top 5 most important features
- Use sliders to simulate different input values
- Observe prediction values update in real-time
- Visualize classification probabilities or regression outputs

---
## Demo Output
Data Upload
![Upload tab](https://github.com/user-attachments/assets/2fa387d1-585c-4dcf-af90-f08102c5a77b)

Task Selection
![Task Selection](https://github.com/user-attachments/assets/6b344c7b-bf86-4354-a30a-08aaf447cbbb)

Preprocessing
![3 Duplicate value](https://github.com/user-attachments/assets/996fa189-2992-4857-960d-d1bd4a89c8fe)
![Pre Processing](https://github.com/user-attachments/assets/293781f6-e7ca-4169-80fc-c10af0016354)
![Label Encoding](https://github.com/user-attachments/assets/a97ae98b-4f76-4e05-81e5-befcd04bf81d)

EDA
![EDA 1](https://github.com/user-attachments/assets/c864174b-9e16-4739-b789-6160fe69aa44)
![EDA 2](https://github.com/user-attachments/assets/db025059-6357-4ccb-9444-b18291cb0db5)
![EDA 3](https://github.com/user-attachments/assets/9697cc1a-1215-4c61-a81c-0a4eb55ad77c)

Feature Engineering
![Feature engineering](https://github.com/user-attachments/assets/0d2f4f77-9856-4745-b9dc-169c17524cd9)
![Feature Engineering 2](https://github.com/user-attachments/assets/dae24438-16fc-4ed4-8467-e741559583b2)

Model Recommendation
![Model Suggestion](https://github.com/user-attachments/assets/52f0ce72-82b9-4f2b-933d-0ee1e82a0adb)
![Model Suggestion 2](https://github.com/user-attachments/assets/729b0f84-43d5-4810-990b-393c670def69)

Model Selection
![Model selection](https://github.com/user-attachments/assets/17991838-722a-4664-bb02-1bfdd46121b6)
![Model Selection 1](https://github.com/user-attachments/assets/9b6634b7-9ba8-4a20-9366-e3c6066a68de)

Forecast
![Forecast](https://github.com/user-attachments/assets/94baf7e5-873f-42d2-a9a6-531f8ccbe566)
![Forecast 2](https://github.com/user-attachments/assets/dd4583de-d7bf-4295-85af-c628b8276d28)
![Forecast Components](https://github.com/user-attachments/assets/27d95a3b-0c6e-4119-b43d-e9a7c4f5d904)
![Automated Report](https://github.com/user-attachments/assets/4144ed79-fc56-4cfc-9c09-62be12aa78ca)

Live Prediction Simulator
![Live prediction simulator](https://github.com/user-attachments/assets/099bb69d-ae80-42fb-951a-b4227f9d59fd)

##  Use Cases

-  Retail sales forecasting  
-  Revenue prediction  
-  Customer segmentation  
-  Churn classification  
-  ML education and training  
-  Smart dashboards for business intelligence

---

##  Data Privacy

- All data stays local or in-session
- No cloud storage or external API calls
- Designed for secure offline use

---

##  Author

**Lenin Uthup**  
leninuthp@gmail.com

 [LinkedIn](https://www.linkedin.com/in/lenin-uthup/)

---

##  License

This project is licensed under the **MIT License**.

---

