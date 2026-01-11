"""
Model Training Script
Trains a Random Forest classifier for DNA-based disease prediction
"""

import sys
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import preprocess_dataset


def train_model(data_path, model_output_path, vectorizer_output_path):
    """
    Train a disease prediction model
    
    Args:
        data_path (str): Path to the genomic dataset CSV
        model_output_path (str): Path to save the trained model
        vectorizer_output_path (str): Path to save the vectorizer
    """
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Disease distribution:\n{df['disease'].value_counts()}\n")
    
    # Extract sequences and labels
    sequences = df['sequence'].tolist()
    labels = df['disease'].tolist()
    
    print("Preprocessing sequences and extracting k-mer features...")
    # Use k=3 (trigrams) and extract up to 1000 features
    X, y, vectorizer = preprocess_dataset(sequences, labels, k=3, max_features=1000)
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Split dataset
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples\n")
    
    # Train Random Forest model
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("Model training completed!\n")
    
    # Evaluate model
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    print(f"Saving model to {model_output_path}...")
    joblib.dump(model, model_output_path)
    
    print(f"Saving vectorizer to {vectorizer_output_path}...")
    joblib.dump(vectorizer, vectorizer_output_path)
    
    print("\nModel training and saving completed successfully!")
    
    return model, vectorizer, accuracy


if __name__ == "__main__":
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'genomic_dataset.csv')
    model_path = os.path.join(base_dir, 'model', 'disease_model.pkl')
    vectorizer_path = os.path.join(base_dir, 'model', 'vectorizer.pkl')
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print("Dataset not found. Generating sample dataset...")
        from data.generate_dataset import create_sample_dataset
        df = create_sample_dataset(num_samples_per_class=50)
        df.to_csv(data_path, index=False)
        print(f"Dataset created at {data_path}\n")
    
    # Train model
    train_model(data_path, model_path, vectorizer_path)
