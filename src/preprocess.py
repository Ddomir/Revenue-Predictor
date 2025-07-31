import pandas as pd

def clean_data(db: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the sales transaction dataset.
    
    Parameters:
        db (pd.DataFrame): Raw input data.
        
    Returns:
        pd.DataFrame: Cleaned and feature-engineered data.
    """
    # Drop irrelevant or unknown columns
    if 'Column1' in db.columns:
        db = db.drop(columns=['Column1'])
    
    # Drop rows with missing values
    db = db.dropna()

    # Convert date to datetime
    db['Date'] = pd.to_datetime(db['Date'])

    # Clean string formatting
    db['Customer Gender'] = db['Customer Gender'].str.lower().str.strip()
    db['Country'] = db['Country'].str.strip()
    db['State'] = db['State'].str.strip()
    db['Product Category'] = db['Product Category'].str.strip()
    db['Sub Category'] = db['Sub Category'].str.strip()

    # Remove unrealistic values
    db = db[(db['Customer Age'] >= 0) & (db['Customer Age'] <= 100)]
    db = db[db['Revenue'] >= 0]

    # Feature engineering
    db['Year'] = db['Date'].dt.year
    db['Month'] = db['Date'].dt.month
    db['DayOfWeek'] = db['Date'].dt.dayofweek
    db['Profit Margin'] = db['Unit Price'] - db['Unit Cost']

    # Drop data columnm - redundant
    db = db.drop(columns=['Date', 'index'])

    return db


def save_clean_data(db: pd.DataFrame, output_path: str = "data/processed/cleaned_data.csv") -> None:
    """
    Saves the cleaned DataFrame to a CSV file.

    Parameters:
        db (pd.DataFrame): Cleaned data
        output_path (str): Output path for CSV file
    """
    db.to_csv(output_path, index=False)
