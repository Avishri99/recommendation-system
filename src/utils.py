import pandas as pd

def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    import joblib
    return joblib.load(filename)

def save_data(df, filename):
    # Save a DataFrame to CSV
    df.to_csv(filename, index=False)
