# Spending Predictor

A machine learning project to predict customer spending patterns using sales transaction data.

## 🔗 Database
Kaggle Database by Vineet Bahl: https://www.kaggle.com/datasets/thedevastator/analyzing-customer-spending-habits-to-improve-sa

## 📊 Project Overview

This project analyzes sales transaction data to build predictive models for customer spending behavior. The pipeline includes data cleaning, exploratory data analysis, feature engineering, and machine learning model training.

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Jupyter Lab
- Required Python packages (see Installation)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd spending-predictor
   ```

2. **Install required packages**
   ```bash
   pip3 install pandas matplotlib seaborn scikit-learn joblib --user --break-system-packages
   ```

3. **Start Jupyter Lab**
   ```bash
   jupyter-lab
   ```

## 📁 Project Structure

```
spending-predictor/
├── data/
│   ├── raw/
│   │   └── SalesForCourse_quizz_table.csv
│   └── processed/
│       └── cleaned_data.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocess.ipynb
├── src/
│   ├── preprocess.py
│   └── train.py
├── models/
│   └── spending_predictor.joblib
├── outputs/
└── README.md
```

## 🔧 Usage

### Data Preprocessing

```python
from src.preprocess import clean_data, save_clean_data

# Load raw data
df = pd.read_csv('data/raw/SalesForCourse_quizz_table.csv')

# Clean and preprocess
cleaned_df = clean_data(df)

# Save processed data
save_clean_data(cleaned_df)
```

### Model Training

```python
from src.train import prepare_features, train_model, evaluate_model

# Prepare features
X, y = prepare_features(cleaned_df)

# Train model
model = train_model(X_train, y_train, model_type='linear')

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
```

### Running the Pipeline

```bash
# Run preprocessing
python src/preprocess.py

# Run training
python src/train.py
```

## 🛠️ Dependencies

- **pandas** (2.3.1): Data manipulation and analysis
- **matplotlib** (3.10.3): Data visualization
- **seaborn** (0.13.2): Statistical data visualization
- **scikit-learn** (1.7.1): Machine learning algorithms
- **joblib** (1.5.1): Model persistence and parallel processing
- **numpy** (2.3.2): Numerical computing
- **jupyter-lab** (4.4.5): Interactive development environment

## 📊 Data

The project uses sales transaction data containing:
- Customer demographics (age, gender, location)
- Product information (category, sub-category)
- Transaction details (date, quantity, revenue, profit)
- Geographic information (country, state)

## 🆘 Support

Check me out on irla.dev!

---

**Note**: This project is designed for educational and research purposes. Always validate model predictions in real-world applications. 