import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()

    # Encode target Grade
    mapping = {'low':0, 'medium':1, 'high':2}
    df['Grade'] = df['Grade'].map(mapping)

    # Scaling numerik
    scale_cols = ['pH','Temprature','Colour']
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df

def run(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(input_path)
    df_prep = preprocess(df)

    X = df_prep.drop(columns=['Grade'])
    y = df_prep['Grade']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv(os.path.join(output_dir, 'MilkQuality_train_preprocessed.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'MilkQuality_test_preprocessed.csv'), index=False)
    df_prep.to_csv(os.path.join(output_dir, 'MilkQuality_full_preprocessed.csv'), index=False)

    print("Preprocessing selesai dan file berhasil disimpan.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run(args.input, args.output)