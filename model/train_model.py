# -*- coding: utf-8 -*-
"""
DNA Disease Risk Classifier - Improved Training Script
Uses XGBoost + LightGBM ensemble on the synthetic_dna_dataset.csv
Target: Disease_Risk (High / Medium / Low)
Features: Tabular genomic stats + TF-IDF k-mer features
"""
# Force UTF-8 output so Windows consoles don't crash on special chars
import sys, os
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found - using GradientBoostingClassifier instead")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not found - using RandomForestClassifier instead")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import sequence_to_kmer_string


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def extract_numeric_features(df):
    """
    Build a rich numeric feature matrix from the dataset.
    Adds interaction terms to help the model find non-linear patterns.
    """
    # Encode categorical 'Class_Label'
    le_class = LabelEncoder()
    df = df.copy()
    df["Class_Label_Enc"] = le_class.fit_transform(df["Class_Label"])

    # Derived / interaction features
    eps = 1e-6
    df["AT_GC_Ratio"]       = df["AT_Content"]  / (df["GC_Content"] + eps)
    df["A_T_Ratio"]         = df["Num_A"]        / (df["Num_T"]       + eps)
    df["C_G_Ratio"]         = df["Num_C"]        / (df["Num_G"]       + eps)
    df["Purine_Pyrimidine"] = (df["Num_A"] + df["Num_G"]) / (df["Num_C"] + df["Num_T"] + eps)
    df["GC_x_kmer"]         = df["GC_Content"]  * df["kmer_3_freq"]
    df["mutation_x_gc"]     = df["Mutation_Flag"] * df["GC_Content"]
    df["A_plus_C"]          = df["Num_A"] + df["Num_C"]
    df["T_plus_G"]          = df["Num_T"] + df["Num_G"]
    df["A_minus_T"]         = df["Num_A"] - df["Num_T"]
    df["C_minus_G"]         = df["Num_C"] - df["Num_G"]
    df["kmer_sq"]           = df["kmer_3_freq"] ** 2
    df["GC_sq"]             = (df["GC_Content"] / 100) ** 2
    df["class_x_mut"]       = df["Class_Label_Enc"] * df["Mutation_Flag"]
    df["kmer_x_mut"]        = df["kmer_3_freq"]    * df["Mutation_Flag"]

    feature_cols = [
        "GC_Content", "AT_Content",
        "Num_A", "Num_T", "Num_C", "Num_G",
        "kmer_3_freq", "Mutation_Flag",
        "Class_Label_Enc",
        "AT_GC_Ratio", "A_T_Ratio", "C_G_Ratio", "Purine_Pyrimidine",
        "GC_x_kmer", "mutation_x_gc",
        "A_plus_C", "T_plus_G", "A_minus_T", "C_minus_G",
        "kmer_sq", "GC_sq", "class_x_mut", "kmer_x_mut",
    ]
    return df[feature_cols].values, le_class


def extract_kmer_features(sequences, vectorizer=None, k=3, max_features=1000):
    """
    Convert each DNA sequence to a space-separated k-mer string and
    compute TF-IDF features over 1-, 2-, and 3-gram k-mer combos.
    """
    kmer_strings = [sequence_to_kmer_string(seq, k) for seq in sequences]

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=2,
            sublinear_tf=True,
        )
        X_kmer = vectorizer.fit_transform(kmer_strings)
    else:
        X_kmer = vectorizer.transform(kmer_strings)

    return X_kmer, vectorizer


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(data_path, model_output_path, vectorizer_output_path):
    """Train and persist the Disease Risk ensemble model."""

    print("=" * 60)
    print("DNA Disease Risk Classifier -- Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df)}  |  Columns: {df.shape[1]}")

    # Detect target column (handles different name casings)
    target_col = None
    for candidate in ["Disease_Risk", "disease_risk", "disease"]:
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        raise ValueError("No target column found. Expected 'Disease_Risk'.")

    print(f"  Target: '{target_col}'")
    print(f"  Class distribution:\n{df[target_col].value_counts().to_string()}")

    # Encode labels
    le_label = LabelEncoder()
    y = le_label.fit_transform(df[target_col])
    print(f"  Encoded classes: {list(le_label.classes_)}")

    # Detect sequence column
    seq_col = next((c for c in ["Sequence", "sequence"] if c in df.columns), None)

    # ------------------------------------------------------------------
    # 2. Feature extraction
    # ------------------------------------------------------------------
    print("\n[2/6] Extracting features...")

    num_features, le_class = extract_numeric_features(df)
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_features)
    print(f"  Engineered numeric features : {num_scaled.shape[1]}")

    kmer_vectorizer = None
    if seq_col is not None:
        kmer_features, kmer_vectorizer = extract_kmer_features(
            df[seq_col].tolist(), max_features=1000
        )
        X = hstack([csr_matrix(num_scaled), kmer_features])
        print(f"  TF-IDF k-mer features       : {kmer_features.shape[1]}")
    else:
        X = csr_matrix(num_scaled)

    print(f"  Total feature matrix shape  : {X.shape}")

    # ------------------------------------------------------------------
    # 3. Train / test split
    # ------------------------------------------------------------------
    print("\n[3/6] Splitting dataset (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 4. Build ensemble
    # ------------------------------------------------------------------
    print("\n[4/6] Building ensemble classifier...")
    estimators = []

    if HAS_XGB:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        estimators.append(("xgb", xgb_clf))
        print("  + XGBoost (400 trees, depth=7)")
    else:
        estimators.append(("gb", GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )))
        print("  + GradientBoosting (XGBoost unavailable)")

    if HAS_LGB:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.04,
            num_leaves=127,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        estimators.append(("lgbm", lgb_clf))
        print("  + LightGBM (400 trees, num_leaves=127)")
    else:
        estimators.append(("rf", RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
        )))
        print("  + RandomForest (LightGBM unavailable)")

    # Always add a RandomForest as a third voter for stability
    estimators.append(("rf_extra", RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        random_state=99, n_jobs=-1,
    )))
    print("  + RandomForest (extra stability voter)")

    model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)

    # ------------------------------------------------------------------
    # 5. Fit
    # ------------------------------------------------------------------
    print("\n[5/6] Training ensemble (this may take a minute)...")
    model.fit(X_train, y_train)
    print("  Done.")

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluating on hold-out test set...")
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)

    print(f"\n  Test Accuracy : {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_label.classes_))

    print("5-fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    print(f"  CV Accuracy : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

    # ------------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------------
    artifacts = {
        "model":               model,
        "label_encoder":       le_label,
        "class_label_encoder": le_class,
        "scaler":              scaler,
        "kmer_vectorizer":     kmer_vectorizer,
        "accuracy":            float(accuracy),
        "cv_mean":             float(cv_scores.mean()),
        "cv_std":              float(cv_scores.std()),
        "classes":             list(le_label.classes_),
        "has_sequence":        seq_col is not None,
        "seq_col":             seq_col,
    }

    print(f"\nSaving model  -> {model_output_path}")
    joblib.dump(artifacts, model_output_path)
    print(f"Saving vectorizer -> {vectorizer_output_path}")
    joblib.dump(kmer_vectorizer, vectorizer_output_path)

    print("\n[DONE] Model saved successfully!")
    print(f"  Test Accuracy : {accuracy*100:.2f}%")
    print(f"  CV Accuracy   : {cv_scores.mean()*100:.2f}%")
    print("=" * 60)

    return model, kmer_vectorizer, accuracy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_candidates = [
        os.path.join(base_dir, "synthetic_dna_dataset.csv"),
        os.path.join(base_dir, "data", "synthetic_dna_dataset.csv"),
        os.path.join(base_dir, "data", "genomic_dataset.csv"),
    ]

    data_path = next((p for p in data_candidates if os.path.exists(p)), None)
    if data_path is None:
        tried = "\n".join(f"  - {p}" for p in data_candidates)
        raise FileNotFoundError(f"Dataset not found. Tried:\n{tried}")

    print(f"Using dataset: {data_path}")

    model_path      = os.path.join(base_dir, "model", "disease_model.pkl")
    vectorizer_path = os.path.join(base_dir, "model", "vectorizer.pkl")

    train_model(data_path, model_path, vectorizer_path)
