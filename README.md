# SmartCare Hospital AI – Appointment No-Show Prediction System

## 1. Project Overview

SmartCare Hospital AI is a machine learning-based project developed to predict whether a patient is likely to miss a scheduled hospital appointment (No-Show).

Missed hospital appointments can lead to wasted medical resources, increased waiting times, and reduced efficiency in healthcare services. This project uses patient and appointment-related data to identify patterns associated with missed appointments and develop a predictive model to support better appointment management.

The system applies machine learning classification algorithms to predict appointment no-show outcomes and includes a web-based prototype for interacting with the trained model.

This project was independently developed as an academic Machine Learning project.

---

## 2. Project Objectives

- Analyze hospital appointment data to identify patterns associated with missed appointments.
- Perform data preprocessing and feature engineering.
- Conduct Exploratory Data Analysis (EDA).
- Develop and compare multiple machine learning classification models.
- Evaluate model performance using appropriate classification metrics.
- Apply Explainable AI techniques to understand model predictions.
- Develop a simple AI-powered prototype for appointment no-show prediction.

---

## 3. Prediction Problem

**Problem Type:** Binary Classification

**Target Variable:** `no_show`

| Value | Meaning |
|---|---|
| 0 | Attended / Other |
| 1 | No-Show |

The model predicts the likelihood of a patient missing a scheduled appointment based on available patient and appointment information.

---

## 4. Dataset

The project uses the dataset:

`smartcare_ai_dataset_1000.csv`

- **Number of records:** 1,000
- **Number of attributes:** 33
- **Target variable:** `no_show`

The dataset contains patient demographics, appointment information, previous appointment history, and other hospital-related attributes.

### Data Preparation

The following data preparation techniques are applied:

- Missing value analysis and handling
- Duplicate record checking
- Outlier identification
- Categorical variable encoding
- Numerical feature scaling
- Feature selection
- Feature engineering

Information that would not be available before an appointment or could cause target leakage is excluded from model training.

---

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and model evaluation |
| SHAP | Explainable AI |
| Jupyter Notebook | Development and experimentation |
| Google Colab | Cloud-based notebook environment |
| Streamlit | Web application prototype |
| Joblib | Model saving and loading |
| Git & GitHub | Version control and project management |

---

## 6. Machine Learning Models

The following machine learning classification algorithms are implemented and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

The models are evaluated to identify the most suitable model for predicting hospital appointment no-shows.

---

## 7. Model Evaluation

### Binary Classification Metrics

The following metrics are used to evaluate model performance:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Additional Evaluation Techniques

- Confusion Matrix
- Classification Report
- ROC Curve
- Model Comparison Table

The best-performing model is selected based on the evaluation results and the objectives of the prediction task.

---

## 8. Explainable AI (XAI)

Explainable AI techniques are used to understand how the trained model makes predictions.

SHAP (SHapley Additive Explanations) is used to:

- Identify the most influential features.
- Analyze feature importance.
- Explain individual predictions.
- Understand factors associated with higher or lower no-show predictions.
- Improve model transparency and interpretability.

---

## 9. AI Prototype

A web-based prototype is developed using Streamlit.

The prototype allows users to:

- Enter patient and appointment information.
- Submit information to the trained machine learning model.
- Receive an appointment no-show prediction.
- View the predicted no-show probability.
- View the prediction risk category.
- Receive suggested follow-up actions.

The prototype is designed as a decision-support tool and does not replace hospital staff decision-making.

---

## 10. Project Structure

```text
smartcare-hospital-ai/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/
│   │   ├── smartcare_ai_dataset_1000.csv
│   │   └── smartcare_ai_dataset_data_dictionary.csv
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 00_full_pipeline.ipynb
│   ├── 01_problem_definition_and_literature.ipynb
│   ├── 02_dataset_understanding.ipynb
│   ├── 03_preprocessing_feature_engineering.ipynb
│   ├── 04_exploratory_data_analysis.ipynb
│   ├── 05_model_development.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_explainable_ai.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── predict.py
│
├── models/
│   └── best_model.joblib
│
├── app/
│   ├── app.py
│   ├── utils.py
│   ├── requirements.txt
│   └── assets/
│
├── reports/
│   ├── figures/
│   │   ├── eda/
│   │   ├── evaluation/
│   │   └── xai/
│   ├── tables/
│   ├── literature/
│   └── technical_report/
│
├── presentation/
├── demo/
├── docs/
└── tests/
```

*Note: The project structure may be updated as development progresses.*

---

## 11. Project Workflow

The project follows an end-to-end machine learning workflow:

1. Problem Definition and Literature Review
2. Dataset Understanding
3. Data Preprocessing and Feature Engineering
4. Exploratory Data Analysis (EDA)
5. Model Development
6. Model Evaluation
7. Explainable AI
8. AI Prototype Development

---

## 12. Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/YasodhiJaneera/smartcare-hospital-ai.git
```

### Navigate to the Project Directory

```bash
cd smartcare-hospital-ai
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 13. Running the Prototype

Install the application dependencies:

```bash
pip install -r app/requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

Once the application starts, open the following URL in your browser if it does not open automatically:

```text
http://localhost:8501
```

---

## 14. Results

Multiple machine learning models are trained and evaluated using the selected classification metrics. Their performance is compared to identify the most suitable model for appointment no-show prediction.

Evaluation metrics, comparison tables, and visualizations are stored in the `reports/` directory.

---

## 15. Ethical Considerations and Limitations

- Predictions are probabilistic and may not always be accurate.
- The dataset may not fully represent real-world hospital populations.
- Patient data privacy and confidentiality must be maintained.
- Predictions should not be used to discriminate against patients.
- SHAP explains model associations but does not establish causation.
- Human oversight is required when using predictions to support hospital operations.
- The prototype is intended for academic and demonstration purposes, not real clinical deployment.

---

## 16. Future Improvements

- Use larger and more diverse real-world hospital datasets.
- Improve model performance through additional feature engineering.
- Explore advanced machine learning and ensemble techniques.
- Integrate real-time appointment data.
- Implement automated SMS or email appointment reminders.
- Develop an interactive hospital management dashboard.
- Conduct fairness and bias analysis.
- Deploy the prototype to a cloud platform.

---

## 17. Developer

**Developed by:** Yasodhi Janeera Senadheera

Software Engineering Undergraduate  
Sri Lanka Technological Campus (SLTC)

This project was independently developed as an academic Machine Learning project.

---

## 18. License

This project was developed for educational and academic purposes.
