import pandas as pd
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

def preprocess_data(df):
    print("Preprocessing data...")
    scaler = StandardScaler()
    
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    return df

def split_and_save(df, output_dir):
    print("Splitting data...")
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Data saved to {output_dir}")

def main(input_file, output_dir):
    df = load_data(input_file)
    df_clean = preprocess_data(df)
    split_and_save(df_clean, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate Data Preprocessing")
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--output", type=str, required=True, help="Folder to save processed data")
    
    args = parser.parse_args()
    main(args.input, args.output)