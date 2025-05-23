# Airline Passenger Satisfaction Prediction Model ✈️📊

## Project Overview

This project aims to build a machine learning model to predict airline passenger satisfaction using the XGBoost algorithm. It covers the full ML pipeline—from Exploratory Data Analysis (EDA) and preprocessing, to model training, hyperparameter tuning, interpretability using SHAP, and deployment via an interactive Streamlit web application.

**Dataset:**  
[Airline Passenger Satisfaction (Kaggle)](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)  
120,000+ samples, 25 features capturing flight experience and passenger demographics.

---
**Key Highlights**:  
- End-to-end ML pipeline optimized for batch predictions  
- Model accuracy: **96.45%** on unseen data  
- SHAP-powered explainability for business decisions  
- GPU-accelerated training/inference  

## Features & Components

| Data Processing                         | Model Development                                      | Production & Deployment                         |
|---------------------------------------|-------------------------------------------------------|----------------------------------------------------|
| ✔️ Automated missing value imputation | ✔️ Comparison of multiple algorithms (XGBoost, RF, LR)| ✔️ Streamlit web app with real-time predictions   |
| ✔️ Categorical encoding & scaling     | ✔️ Hyperparameter tuning (RandomizedSearchCV)         | ✔️ Batch prediction support                       |
| ✔️ Feature engineering & selection    | ✔️ GPU-accelerated training                            | ✔️ SHAP-based explainability dashboard           |

---



##Extras

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







