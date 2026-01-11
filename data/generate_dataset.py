"""
Sample Genomic Dataset Generator
Creates synthetic DNA sequences with disease labels for training
"""

import random
import pandas as pd


def generate_random_dna(length, gc_content=0.5):
    """Generate a random DNA sequence with specified GC content"""
    nucleotides = []
    for _ in range(length):
        if random.random() < gc_content:
            nucleotides.append(random.choice(['G', 'C']))
        else:
            nucleotides.append(random.choice(['A', 'T']))
    return ''.join(nucleotides)


def generate_disease_specific_sequence(disease, length=200):
    """
    Generate DNA sequences with disease-specific patterns
    This is a simplified simulation for demonstration purposes
    """
    base_seq = generate_random_dna(length, gc_content=0.5)
    
    # Add disease-specific motifs (simplified patterns)
    if disease == "Diabetes":
        # Higher GC content in certain regions
        motif = "GCGCGCGC" * 3
        insertion_point = length // 4
        base_seq = base_seq[:insertion_point] + motif + base_seq[insertion_point + len(motif):]
    
    elif disease == "Heart Disease":
        # Specific repeated patterns
        motif = "ATATATAT" * 3
        insertion_point = length // 3
        base_seq = base_seq[:insertion_point] + motif + base_seq[insertion_point + len(motif):]
    
    elif disease == "Cancer":
        # AT-rich regions
        motif = "AAATTTAAA" * 3
        insertion_point = length // 2
        base_seq = base_seq[:insertion_point] + motif + base_seq[insertion_point + len(motif):]
    
    elif disease == "Alzheimer":
        # GC-rich regions with specific pattern
        motif = "CCCGGGCCC" * 3
        insertion_point = length // 5
        base_seq = base_seq[:insertion_point] + motif + base_seq[insertion_point + len(motif):]
    
    elif disease == "Asthma":
        # Mixed pattern
        motif = "ACGTACGT" * 3
        insertion_point = length // 6
        base_seq = base_seq[:insertion_point] + motif + base_seq[insertion_point + len(motif):]
    
    else:  # Healthy
        # Random sequence with balanced GC content
        base_seq = generate_random_dna(length, gc_content=0.5)
    
    return base_seq


def create_sample_dataset(num_samples_per_class=50):
    """
    Create a sample dataset with DNA sequences and disease labels
    """
    diseases = ["Healthy", "Diabetes", "Heart Disease", "Cancer", "Alzheimer", "Asthma"]
    
    sequences = []
    labels = []
    
    for disease in diseases:
        for _ in range(num_samples_per_class):
            # Vary sequence length
            seq_length = random.randint(150, 300)
            sequence = generate_disease_specific_sequence(disease, seq_length)
            sequences.append(sequence)
            labels.append(disease)
    
    # Create DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'disease': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    # Generate and save dataset
    df = create_sample_dataset(num_samples_per_class=50)
    df.to_csv('/home/runner/work/DNA-disease/DNA-disease/data/genomic_dataset.csv', index=False)
    print(f"Dataset created with {len(df)} samples")
    print(f"Disease distribution:\n{df['disease'].value_counts()}")
