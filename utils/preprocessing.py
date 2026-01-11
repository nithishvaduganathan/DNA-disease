"""
DNA Sequence Preprocessing
K-mer feature extraction and sequence vectorization
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def generate_kmers(sequence, k=3):
    """
    Generate k-mers from DNA sequence
    
    Args:
        sequence (str): DNA sequence
        k (int): Length of k-mer (default: 3 for trigrams)
        
    Returns:
        list: List of k-mers
    """
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        kmers.append(kmer)
    return kmers


def sequence_to_kmer_string(sequence, k=3):
    """
    Convert DNA sequence to k-mer string representation
    
    Args:
        sequence (str): DNA sequence
        k (int): Length of k-mer
        
    Returns:
        str: Space-separated k-mers
    """
    kmers = generate_kmers(sequence, k)
    return ' '.join(kmers)


def create_kmer_vectorizer(k=3, max_features=1000):
    """
    Create a CountVectorizer for k-mer features
    
    Args:
        k (int): Length of k-mer
        max_features (int): Maximum number of features to extract
        
    Returns:
        CountVectorizer: Configured vectorizer
    """
    vectorizer = CountVectorizer(
        analyzer='word',
        token_pattern=r'\b\w+\b',
        max_features=max_features,
        lowercase=False
    )
    return vectorizer


def extract_features(sequence, vectorizer, k=3):
    """
    Extract features from DNA sequence using k-mer vectorization
    
    Args:
        sequence (str): DNA sequence
        vectorizer: Fitted CountVectorizer
        k (int): Length of k-mer
        
    Returns:
        numpy.ndarray: Feature vector
    """
    kmer_string = sequence_to_kmer_string(sequence, k)
    features = vectorizer.transform([kmer_string])
    return features


def preprocess_dataset(sequences, labels, k=3, max_features=1000):
    """
    Preprocess a dataset of DNA sequences
    
    Args:
        sequences (list): List of DNA sequences
        labels (list): List of corresponding disease labels
        k (int): Length of k-mer
        max_features (int): Maximum number of features
        
    Returns:
        tuple: (X, y, vectorizer) - features, labels, fitted vectorizer
    """
    # Convert sequences to k-mer strings
    kmer_strings = [sequence_to_kmer_string(seq, k) for seq in sequences]
    
    # Create and fit vectorizer
    vectorizer = create_kmer_vectorizer(k, max_features)
    X = vectorizer.fit_transform(kmer_strings)
    
    return X, np.array(labels), vectorizer
