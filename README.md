# Smart Tourist Recommender System

This project is a **Tourism Experience Analytics** platform designed to provide **personalized recommendations**, **predict user ratings**, and **classify user visit modes** using machine learning. It integrates data preprocessing, model building, and a **Streamlit application** for interactive exploration.

---

## Project Overview

Tourism businesses and platforms aim to enhance user experience through data-driven decision-making. This project leverages **classification**, **regression**, and **recommendation models** to:

- Predict the **mode of visit** (e.g., Family, Business, Friends)
- Estimate **user ratings** for tourist attractions
- Offer **attraction recommendations** tailored to user profiles

---

## Project Objectives

### Classification: Predict Visit Mode

- Classify the user's visit as `Business`, `Family`, `Friends`, etc.
- Uses demographics and travel attributes

### Regression: Predict Attraction Rating

- Predict user's expected rating (1–5 scale)
- Inputs: user features, location, attraction metadata

### Recommendations: Suggest Attractions

- Collaborative Filtering: Based on similar users
- Content-Based Filtering: Based on previously liked attractions

---

## Data Sources (`/dataset`)

| File                                                         | Description                                          |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| `User.xlsx`                                                  | User demographics (continent, region, country, city) |
| `Transaction.xlsx`                                           | Ratings, visit dates, visit mode                     |
| `Item.xlsx`                                                  | Attraction metadata                                  |
| `Updated_Item.xlsx`                                          | Updated version of attractions                       |
| `City.xlsx`, `Country.xlsx`, `Region.xlsx`, `Continent.xlsx` | Location metadata                                    |
| `Mode.xlsx`                                                  | Visit mode mapping                                   |
| `Type.xlsx`                                                  | Attraction type info                                 |

All files are in **Excel** format.

---

## Machine Learning Models (`/models`)

- **Classification Models** (Visit Mode)

  - `visitmode_classifier_randomforest.pkl`
  - `visitmode_classifier_xgboost.pkl`
  - `visitmode_classifier_lightgbm.pkl`

- **Regression Model**

  - `rating_regressor.pkl`

- **Recommendation Systems**

  - `collaborative_model.pkl`: Matrix-based collaborative filtering
  - `content_based_model.pkl`: Feature-based recommendation

- **Support Files**

  - Encoders: `*_label_encoders.pkl`
  - Feature templates: `input_template_row_*.pkl`
  - Selected features: `*_features.pkl`

---

## Output Files (`/output`)

| File                           | Description                            |
| ------------------------------ | -------------------------------------- |
| `feature_engineered_data.xlsx` | Final processed dataset for modeling   |
| `master_dataset.xlsx`          | Combined dataset from all sources      |
| `synthetic_transaction.xlsx`   | Simulated transaction data for testing |

---

## Application (`app.py`)

Built with **Streamlit**, the app has 4 interactive modules:

1. **Visit Mode Prediction** (Classification)
2. **Rating Score Prediction** (Regression)
3. **Visual Insights** (EDA Visualizations)
4. **Attraction Recommendations** (Collaborative + Content-Based)

To run the app:

```bash
streamlit run app.py
```

---

## Exploratory Data Analysis

Refer to `EDA_Analysis.ipynb` for:

- Rating distribution analysis
- Popular attraction types
- Visit mode vs demographics
- Temporal travel patterns

---

## Implementation Steps

It includes:

- Feature engineering
- Label encoding
- Model training & selection
- Performance evaluation (Accuracy, F1, RMSE, etc.)
- Streamlit integration

---

## Use Cases

- Personalized tourist attraction suggestions
- Travel behavior classification
- User satisfaction prediction
- Location-based market analysis

---

## Key Metrics

| Task           | Metrics Used           |
| -------------- | ---------------------- |
| Classification | Accuracy, F1-score     |
| Regression     | RMSE, R², MAE          |
| Recommender    | MAP, Precision@K, RMSE |

---

## Technologies Used

- Python, Pandas, NumPy, Scikit-learn
- XGBoost, LightGBM, RandomForest
- Streamlit (Frontend & Deployment)
- Seaborn & Matplotlib (Visualization)
- Joblib (Model serialization)

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

> *Note: Create `requirements.txt` using*
> `pip freeze > requirements.txt`

### Launch App

```bash
streamlit run app.py
```

---

## References

- [Streamlit Docs](https://docs.streamlit.io/)

---
