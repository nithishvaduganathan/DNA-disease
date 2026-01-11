# Testing Guide

This document describes how to test the DNA Disease Prediction System.

## Unit Testing

### Test DNA Sequence Validation

```python
from utils.validator import validate_dna_sequence, calculate_gc_content

# Test valid sequence
valid_seq = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
is_valid, message = validate_dna_sequence(valid_seq)
assert is_valid == True

# Test invalid characters
invalid_seq = "ATCG123XYZ"
is_valid, message = validate_dna_sequence(invalid_seq)
assert is_valid == False

# Test GC content calculation
seq = "ATGC"
gc = calculate_gc_content(seq)
assert gc == 50.0
```

### Test K-mer Generation

```python
from utils.preprocessing import generate_kmers, sequence_to_kmer_string

# Test k-mer generation
seq = "ATCG"
kmers = generate_kmers(seq, k=3)
assert kmers == ["ATC", "TCG"]

# Test k-mer string conversion
kmer_str = sequence_to_kmer_string(seq, k=3)
assert kmer_str == "ATC TCG"
```

## Integration Testing

### Test Flask Application

```bash
# Start the application
python app.py

# In another terminal, test the endpoints
curl -X POST http://localhost:5000/predict \
  -F "sequence=ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

# Test history endpoint
curl http://localhost:5000/history
```

### Test Model Prediction

```python
import joblib
from utils.preprocessing import sequence_to_kmer_string

# Load model and vectorizer
model = joblib.load('model/disease_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Test prediction
test_seq = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
kmer_str = sequence_to_kmer_string(test_seq, k=3)
features = vectorizer.transform([kmer_str])
prediction = model.predict(features)[0]
confidence = max(model.predict_proba(features)[0])

print(f"Predicted: {prediction} with confidence {confidence:.2%}")
```

## Manual Testing Checklist

### Input Validation
- [ ] Paste a valid DNA sequence with only A, T, C, G
- [ ] Paste a sequence with invalid characters (should show error)
- [ ] Paste a sequence shorter than 50 nucleotides (should show error)
- [ ] Upload a valid FASTA file
- [ ] Upload an invalid file format
- [ ] Test with empty input (should show error)

### Predictions
- [ ] Load and test Diabetes sample
- [ ] Load and test Cancer sample
- [ ] Load and test Healthy sample
- [ ] Verify confidence scores are displayed
- [ ] Verify risk levels are shown correctly
- [ ] Verify top 3 disease probabilities appear
- [ ] Verify sequence statistics are accurate

### Features
- [ ] Download PDF report (check it opens correctly)
- [ ] Click "New Analysis" button (form should reset)
- [ ] View prediction history (should show past predictions)
- [ ] Check medical disclaimer is visible on results page

### UI/UX
- [ ] Test on desktop browser
- [ ] Test on mobile browser (responsive design)
- [ ] Test navigation between Home and History
- [ ] Verify all buttons are clickable
- [ ] Check for any console errors in browser

## Performance Testing

### Load Testing
```bash
# Install Apache Bench
apt-get install apache2-utils

# Test with 100 requests, 10 concurrent
ab -n 100 -c 10 -p sequence.txt -T "application/x-www-form-urlencoded" \
   http://localhost:5000/predict
```

### Sequence Testing
- Test with sequences of varying lengths (50-500 nucleotides)
- Test with sequences having different GC content
- Test with edge cases (all A's, all T's, etc.)

## Continuous Testing

### Pre-commit Checks
```bash
# Syntax check all Python files
python3 -m py_compile app.py utils/*.py model/*.py data/*.py

# Run the application and verify it starts
timeout 5 python3 app.py || true
```

### Regression Testing
After any code changes:
1. Retrain the model: `python model/train_model.py`
2. Test all sample sequences
3. Verify accuracy hasn't degraded
4. Test all API endpoints
5. Check PDF generation still works

## Bug Reporting

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Error messages from console/logs

## Test Data

Sample sequences for testing are available in:
- `data/sample_sequences.fasta`

Create your own test sequences:
```python
from data.generate_dataset import generate_disease_specific_sequence

# Generate test sequences
diabetes_seq = generate_disease_specific_sequence("Diabetes", 200)
cancer_seq = generate_disease_specific_sequence("Cancer", 200)
healthy_seq = generate_disease_specific_sequence("Healthy", 200)
```
