import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


def prepare_features(db: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares features and target for machine learning.
    
    Parameters:
        db (pd.DataFrame): Cleaned input data.
        
    Returns:
        tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).
    """
    # Separate features and target
    X = db.drop(columns=['Revenue'])  # Features
    y = db['Revenue']                 # Target
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'linear') -> object:
    """
    Trains a machine learning model.
    
    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_type (str): Type of model ('linear' or 'random_forest').
        
    Returns:
        object: Trained model.
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
    else:
        raise ValueError("model_type must be 'linear' or 'random_forest'")
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates the trained model.
    
    Parameters:
        model (object): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5  # Using power of 0.5 instead of numpy
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2_Score': r2
    }
    
    return metrics


def save_model(model: object, model_path: str = "models/spending_predictor.joblib") -> None:
    """
    Saves the trained model to a file.

    Parameters:
        model (object): Trained model to save.
        model_path (str): Path where to save the model.
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def main():
    """
    Main function to run the training pipeline.
    """
    # Load cleaned data
    df = pd.read_csv('data/processed/cleaned_data.csv')
    
    # Prepare features and target
    X, y = prepare_features(df)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train, model_type='linear')
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print results
    print("Model Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    save_model(model)


if __name__ == "__main__":
    main()
