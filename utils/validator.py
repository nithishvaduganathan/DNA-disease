"""
DNA Sequence Validator
Validates DNA sequences to ensure only valid nucleotides (A, T, C, G) are accepted
"""

import re


def validate_dna_sequence(sequence):
    """
    Validate if the sequence contains only valid DNA nucleotides (A, T, C, G)
    
    Args:
        sequence (str): DNA sequence to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not sequence or len(sequence.strip()) == 0:
        return False, "Sequence cannot be empty"
    
    # Remove whitespace and convert to uppercase
    sequence = sequence.strip().upper().replace(" ", "").replace("\n", "").replace("\r", "")
    
    # Check minimum length
    if len(sequence) < 50:
        return False, f"Sequence too short ({len(sequence)} nucleotides). Minimum 50 nucleotides required for accurate prediction."
    
    # Check for valid nucleotides only (A, T, C, G)
    if not re.match("^[ATCG]+$", sequence):
        invalid_chars = set(re.findall(r"[^ATCG]", sequence))
        return False, f"Invalid characters found: {', '.join(invalid_chars)}. Only A, T, C, G are allowed."
    
    return True, "Valid DNA sequence"


def clean_dna_sequence(sequence):
    """
    Clean and normalize DNA sequence
    
    Args:
        sequence (str): Raw DNA sequence
        
    Returns:
        str: Cleaned DNA sequence
    """
    # Remove whitespace, newlines, and convert to uppercase
    sequence = sequence.strip().upper()
    sequence = sequence.replace(" ", "").replace("\n", "").replace("\r", "")
    sequence = sequence.replace("\t", "")
    
    return sequence


def parse_fasta(fasta_content):
    """
    Parse FASTA format and extract DNA sequence
    
    Args:
        fasta_content (str): FASTA file content
        
    Returns:
        str: Extracted DNA sequence
    """
    lines = fasta_content.strip().split('\n')
    sequence = ""
    
    for line in lines:
        line = line.strip()
        # Skip header lines (starting with >)
        if not line.startswith('>') and line:
            sequence += line.upper()
    
    return sequence


def calculate_gc_content(sequence):
    """
    Calculate GC content percentage of DNA sequence
    
    Args:
        sequence (str): DNA sequence
        
    Returns:
        float: GC content percentage
    """
    sequence = sequence.upper()
    g_count = sequence.count('G')
    c_count = sequence.count('C')
    total = len(sequence)
    
    if total == 0:
        return 0.0
    
    return ((g_count + c_count) / total) * 100


def get_sequence_stats(sequence):
    """
    Get statistics about the DNA sequence
    
    Args:
        sequence (str): DNA sequence
        
    Returns:
        dict: Dictionary containing sequence statistics
    """
    sequence = sequence.upper()
    
    stats = {
        'length': len(sequence),
        'a_count': sequence.count('A'),
        't_count': sequence.count('T'),
        'c_count': sequence.count('C'),
        'g_count': sequence.count('G'),
        'gc_content': calculate_gc_content(sequence),
        'at_content': ((sequence.count('A') + sequence.count('T')) / len(sequence)) * 100 if len(sequence) > 0 else 0
    }
    
    return stats
