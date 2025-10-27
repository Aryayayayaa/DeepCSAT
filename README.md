# DeepCSAT
This project predicts customer CSAT scores (1-5) using structured features (price, tenure, time) and customer remarks (text data). It uses SMOTE to fix class imbalance and finds RF Baseline (F1: 0.5418) is the best model. Key findings: CSAT drops for high-price items &amp; >90-day agents. 🚀

---

## CSAT Score Prediction and Driver Analysis 📊

This project develops a machine learning solution to predict customer $\text{CSAT}$ scores (1-5) and identifies the key operational and product drivers influencing customer satisfaction. It utilizes a blend of structured operational data and unstructured customer remarks (text) for prediction.

---

## Key Features and Methodology

### 1. Data Sources and Preparation
* **Data Integration:** Combines structured data (item price, agent shift, handling time, agent tenure) with unstructured customer remarks.
* **Feature Engineering:** Created time-based features (`Hour`, `DayOfWeek`) and a critical flag for the worst-performing service hour (`is_4am_plunge`) .
* **Text Processing:** Used $\text{Tf-idf}$ Vectorization to convert customer remarks into numerical features.

### 2. Handling Class Imbalance
* The original $\text{CSAT}$ distribution was severely imbalanced (e.g., $\text{CSAT 5}$ was $\approx 70\%$, $\text{CSAT 2}$ was $\approx 1.5\%$) .
* The **SMOTE** technique was applied to the training data to create a perfectly balanced dataset, ensuring all five classes have equal predictive influence.

### 3. Exploratory Data Analysis (EDA) Insights

| Finding | Inference |
| :--- | :--- |
| **Agent Tenure** | $\text{CSAT}$ peaks with agents in the **61-90 day bucket**, and is **significantly lower** for both trainees and the most veteran agents (>90 days) . |
| **Item Price** | **CSAT is inversely related to price**, with satisfaction significantly lower for items priced **over \$10K** compared to low-priced items . |
| **Response Time** | $\text{CSAT}$ is highest when response time is **less than 30 minutes** and steadily declines as time increases . |

### 4. Machine Learning Modeling & Results

Six models were trained using the $2028$ combined structured and text features.

| Model | Weighted F1-Score | Accuracy | Inference |
| :--- | :--- | :--- | :--- |
| **RF Baseline** | **0.5418** | **0.5082** | **Best Performer.** Best overall balance of precision and recall. |
| RF Optimized | 0.5077 | 0.4512 | Optimization led to **overfitting** and lower test performance. |
| LR Baseline | 0.3933 | 0.3186 | Poor performance, struggling to distinguish mid-range $\text{CSAT}$ classes. |

**Critical Weakness:** All models showed **near-zero F1-Scores** ($\approx 0.02 - 0.07$) for the minority $\text{CSAT 2}$ and $\text{CSAT 3}$ classes, indicating poor predictive power for 'Passive' and 'Detractor' customers .

### 5. Deployment

The best-performing model, the **$\text{RF}$ Baseline**, is deployed as a web application using **Streamlit** for live $\text{CSAT}$ prediction.

### 🛠️ Technologies Used
* **Python:** pandas, numpy
* **ML/Statistical:** scikit-learn, joblib, imblearn ($\text{SMOTE}$)
* **Deployment:** Streamlit
* **Visualization:** seaborn, matplotlib

### 🚀 Get Started

1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the Streamlit app: `streamlit run app.py`
