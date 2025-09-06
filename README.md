# 📊 Customer Churn Prediction System  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)  
![Machine Learning](https://img.shields.io/badge/ML-ScikitLearn-green)  
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🚀 Overview  
This project predicts **Customer Churn** using machine learning models and provides an **interactive Streamlit web app**.  
It allows both **single-customer predictions** and **bulk predictions via CSV upload**, helping businesses identify customers likely to leave.  

---

## ✨ Features  
- 🔎 **Single Customer Prediction** – Enter details via sidebar and get real-time churn probability.  
- 📂 **Bulk Upload Prediction** – Upload CSV files and download predictions with churn probabilities.  
- 📊 **Exploratory Data Analysis (EDA)** – Visual insights into churn indicators.  
- 🤖 **ML Models** – Logistic Regression & Random Forest with evaluation metrics (Accuracy, Precision, Recall, ROC-AUC).  
- ⚡ **Deployed with Streamlit** – Lightweight, interactive, and user-friendly interface.  

---

## 📂 Project Structure  
├── churn_model.pkl # Trained ML model
├── feature_names.pkl # Feature names for prediction
├── app.py # Streamlit web app
├── requirements.txt # Python dependencies
├── data/ # Sample dataset
├── notebooks/ # Jupyter notebooks (EDA & Model training)
└── README.md # Project documentation



---

## ⚙️ Installation & Usage  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
2️⃣ Create & activate virtual environment
bash
Copy code
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Streamlit app
bash
Copy code
streamlit run app.py
📊 Sample Prediction
Single Input Example

json
Copy code
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "InternetService": "DSL",
  "Contract": "Month-to-month",
  "MonthlyCharges": 70.0,
  "TotalCharges": 1000.0
}
Output:
✅ Customer is likely to stay
📈 Probability: 63%

🛠️ Tech Stack
Python 🐍

Scikit-learn – ML algorithms

Pandas, Numpy – Data preprocessing

Matplotlib, Seaborn – Data visualization

Streamlit – Web application

📈 Model Performance
Accuracy: ~80%

Precision, Recall, ROC-AUC measured

Key features influencing churn: Contract type, Tenure, Internet service, Monthly charges

📌 Future Improvements
🔮 Deploy on Streamlit Cloud / Heroku for live access

📊 Add dashboard for churn insights

🧠 Explore deep learning models (LSTM/ANN) for prediction

👨‍💻 Author
Rohit Mahajan
📧 rohitmahajan123bca@gmail.com

