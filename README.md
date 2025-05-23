# Airline Passenger Satisfaction Prediction ✈️

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/streamlit-✨-orange.svg)](https://streamlit.io/)  
[![Status](https://img.shields.io/badge/status-Complete-brightgreen.svg)](https://github.com/Pedrom2002/Airline-Passenger-Satisfaction)

Machine Learning project to predict airline passenger satisfaction using XGBoost, including EDA, hyperparameter tuning, and a Streamlit deployment.

---

## 🚀 Table of Contents
- [Overview](#overview)  
- [Dataset](#dataset)  
- [Repository Structure](#repository-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [ML Pipeline](#ml-pipeline)  
- [EDA](#eda)  
- [Training & Evaluation](#training--evaluation)  
- [Web App](#web-app)  
- [Results](#results)  
- [Extras](#Extras)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

---

## 📖 Overview
This repository contains the complete pipeline to **predict passenger satisfaction** on commercial flights. It includes exploratory data analysis, modeling with XGBoost, result interpretation with SHAP, and a Streamlit web interface for inference.

## 📊 Dataset
Source: [Kaggle – Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)  
- **Records**: ~129,000 flights  
- **Features**: demographics (age, gender), class, punctuality, cleanliness, services (Wi-Fi, entertainment), and target (satisfied/unsatisfied).

Files in `data/`:
```bash
train.csv      # Training data
test.csv     # Test data for batch predictions
```

## 📁 Repository Structure
```bash
├── data/                   
│   ├── train.csv           
│   └── test.csv          
├── models/                 
│   └── xgb_model.pkl       
├── EDAnotebook.ipynb        
├── modelnotebook.ipynb   
├── app.py                  
├── requirements.txt        
└── README.md               
```

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Pedrom2002/Airline-Passenger-Satisfaction.git
   cd Airline-Passenger-Satisfaction
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

## 🎯 Usage 
- **Modeling**: run `modelnotebook.ipynb` to train and tune the model.  
- **App**: start the Streamlit app:
  ```bash
  streamlit run app.py
  ```
  Access it at `http://localhost:8501`.

## 🔄 ML Pipeline
1. Data loading and cleaning  
2. Feature encoding and scaling  
3. Train/test split  
4. XGBoost training + hyperparameter tuning (`RandomizedSearchCV`)  
5. Evaluation (accuracy, precision, recall, F1-score)  
6. Interpretation with SHAP
   

## 🏆 Training & Evaluation
- **Accuracy**: 96.5%  
- **F1-score**: 0.96  
- **Top features**: cleanliness, punctuality, comfort (via SHAP)  

## 🌐 Web App
- Individual prediction via form  
- CSV upload for batch inference  
- Variable importance charts  

## 📈 Results
- XGBoost outperforms Random Forest and Logistic Regression  
- Insights for service improvements (cleanliness and entertainment)  

## Extras

## Exploratory Data Analysis Highlights

- **Class-wise Satisfaction:**  
  Business class passengers report highest satisfaction (~69%), followed by Eco Plus (~25%), and Economy (~19%).

- **Gender Preferences:**  
  Females tend to rate seat comfort higher by 5.3%, while males value legroom more.

- **Key Feature Correlations:**  
  - Inflight Wi-Fi & Ease of Online Booking: 0.72  
  - Cleanliness & Food/Seat Comfort: 0.65–0.69  

- **Potential Multicollinearity:**  
  Seat comfort, cleanliness, and inflight entertainment show moderate correlation (~0.6–0.7), suggesting dimensionality reduction or feature combination might improve models.

---

## Statistical Analysis

- **Multifactorial Satisfaction**  
  Passenger satisfaction is driven by a combination of digital services (Wi-Fi, online check-in), physical comfort (seat, legroom), cleanliness, and punctuality.

- **Gender Differences**  
  - **Women** value seat comfort and cabin cleanliness more highly.  
  - **Men** prioritize legroom and are more likely to be “disloyal” customers.

- **Impact of Travel Class**  
  - **Business Class** enjoys the highest satisfaction, validating premium service investments.  
  - **Eco Plus** shows unexpectedly low satisfaction—especially among female travelers—indicating unmet expectations.  
  - **Economy** has the lowest satisfaction, reflecting basic service levels and discomfort.

- **Cleanliness as a Multiplier**  
  Cabin cleanliness amplifies the perceived quality of other services (food, entertainment, seating). Prioritizing visible hygiene standards can yield outsized gains in overall satisfaction.
- Significant difference in satisfaction by gender confirmed via t-test (p < 0.001).  
- ANOVA tests for satisfaction differences across flight classes support strong class impact.


## 📜 License
This project is licensed under the [MIT License](LICENSE).

## 📬 Contact
Pedro M. – [pedrom02.dev@gmail.com](mailto:pedrom02.dev@gmail.com)  
GitHub: [@Pedrom2002](https://github.com/Pedrom2002)




