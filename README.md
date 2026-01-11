# DNA Disease Prediction System

A comprehensive end-to-end DNA-based disease prediction system using machine learning and Flask. This application allows users to input DNA sequences and receive detailed disease predictions with confidence scores and risk assessments.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

### Core Functionality
- ✅ **DNA Sequence Input**: Paste directly or upload FASTA/plain text files
- ✅ **Sequence Validation**: Ensures only valid nucleotides (A, T, C, G) are accepted
- ✅ **Machine Learning Prediction**: Random Forest classifier trained on genomic data
- ✅ **Disease Classification**: Predicts from 6 disease categories
  - Healthy
  - Diabetes
  - Heart Disease
  - Cancer
  - Alzheimer's Disease
  - Asthma
- ✅ **Confidence Scoring**: Shows prediction confidence with probability distribution
- ✅ **Risk Level Assessment**: Classifies as Low, Medium, or High risk
- ✅ **DNA Sequence Statistics**: Length, GC content, AT content, nucleotide counts

### Additional Features
- 📊 **Multi-Disease Probability Output**: Shows top 3 disease predictions
- 📄 **PDF Report Generation**: Download detailed prediction reports
- 📝 **Prediction History**: SQLite-based storage of past predictions
- 🎨 **Responsive UI**: Bootstrap-based clean, modern interface
- ⚠️ **Medical Disclaimer**: Clear educational purpose statement
- 🧪 **Sample Sequences**: Quick test with pre-loaded examples

## Project Structure

```
DNA-disease/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
├── model/
│   ├── __init__.py
│   ├── train_model.py         # Model training script
│   ├── disease_model.pkl      # Trained Random Forest model
│   └── vectorizer.pkl         # K-mer vectorizer
├── utils/
│   ├── __init__.py
│   ├── validator.py           # DNA sequence validation
│   └── preprocessing.py       # K-mer feature extraction
├── data/
│   ├── generate_dataset.py    # Dataset generation script
│   └── genomic_dataset.csv    # Training dataset
├── templates/
│   └── index.html             # Main web interface
└── static/
    ├── css/
    │   └── style.css          # Custom styles
    └── js/
        └── main.js            # Frontend JavaScript
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/nithishvaduganathan/DNA-disease.git
cd DNA-disease
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate Dataset (Optional)
If you want to regenerate the training dataset:
```bash
python data/generate_dataset.py
```

### Step 4: Train the Model (Optional)
If you want to retrain the model:
```bash
python model/train_model.py
```

### Step 5: Run the Application
```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## Usage

### Input Methods

#### Method 1: Paste DNA Sequence
1. Navigate to the application homepage
2. Select the "Paste Sequence" tab
3. Enter your DNA sequence (minimum 50 nucleotides)
4. Click "Analyze DNA Sequence"

#### Method 2: Upload File
1. Select the "Upload File" tab
2. Choose a FASTA or text file containing the DNA sequence
3. Click "Analyze DNA Sequence"

#### Method 3: Use Sample Sequences
Click one of the quick test buttons to load a sample sequence:
- Load Diabetes Sample
- Load Cancer Sample
- Load Healthy Sample

### Understanding Results

The prediction results include:
- **Predicted Disease**: The most likely disease associated with the sequence
- **Confidence Score**: Probability of the prediction (0-100%)
- **Risk Level**: Classification based on confidence (High/Medium/Low)
- **Top Disease Probabilities**: Distribution across all disease categories
- **Sequence Statistics**: Comprehensive DNA sequence analysis

### Downloading Reports

Click the "Download PDF Report" button to generate a detailed PDF containing:
- Prediction results
- Confidence scores
- Sequence statistics
- Medical disclaimer

### Viewing History

Click "History" in the navigation to view past predictions with:
- Timestamps
- Sequence information
- Predicted diseases
- Confidence levels

## Technical Details

### Machine Learning Pipeline

1. **K-mer Feature Extraction**: DNA sequences are converted to 3-mers (trigrams)
2. **Vectorization**: CountVectorizer creates feature vectors (max 1000 features)
3. **Classification**: Random Forest classifier with 100 estimators
4. **Evaluation**: Model achieves ~98% accuracy on test set

### DNA Sequence Validation

- Accepts only A, T, C, G nucleotides
- Minimum length: 50 nucleotides
- Removes whitespace and handles FASTA format
- Provides detailed error messages for invalid input

### Risk Level Classification

- **High Risk**: Confidence ≥ 70%
- **Medium Risk**: Confidence 40-69%
- **Low Risk**: Confidence < 40%

## API Endpoints

### POST /predict
Analyze a DNA sequence and return prediction results.

**Request:**
- Form data with `sequence` field (text), OR
- Form data with `file` field (FASTA/text file)

**Response:**
```json
{
  "disease": "Diabetes",
  "confidence": 0.85,
  "risk_level": "High",
  "probabilities": {
    "Diabetes": 0.85,
    "Healthy": 0.05,
    ...
  },
  "top_predictions": [
    ["Diabetes", 0.85],
    ["Healthy", 0.05],
    ...
  ],
  "stats": {
    "length": 150,
    "gc_content": 45.5,
    "at_content": 54.5,
    ...
  }
}
```

### GET /history
Retrieve prediction history from database.

**Response:**
```json
[
  {
    "timestamp": "2024-01-01T12:00:00",
    "sequence_length": 150,
    "gc_content": 45.5,
    "predicted_disease": "Diabetes",
    "confidence": 0.85,
    "risk_level": "High"
  },
  ...
]
```

### POST /download-report
Generate and download a PDF report.

**Request:**
```json
{
  "disease": "Diabetes",
  "confidence": 0.85,
  "risk_level": "High",
  "stats": { ... }
}
```

**Response:** PDF file download

## Model Performance

The Random Forest classifier achieves excellent performance:

```
Accuracy: 98.33%

Classification Report:
               precision    recall  f1-score   support
    Alzheimer       1.00      0.90      0.95        10
       Asthma       0.91      1.00      0.95        10
       Cancer       1.00      1.00      1.00        10
     Diabetes       1.00      1.00      1.00        10
      Healthy       1.00      1.00      1.00        10
Heart Disease       1.00      1.00      1.00        10
```

## Future Enhancements

- 🧬 Deep learning models (CNN, LSTM) for sequence analysis
- ☁️ Cloud deployment (AWS, Azure, GCP)
- 📈 Real genomic dataset integration
- 🔍 Multi-gene analysis support
- 📊 Advanced visualization (sequence alignment, mutation analysis)
- 🔐 User authentication and personalized history
- 🌐 REST API with authentication
- 📱 Mobile application

## Dependencies

- **Flask 3.0.0**: Web framework
- **NumPy 1.26.2**: Numerical computing
- **Pandas 2.1.4**: Data manipulation
- **scikit-learn 1.3.2**: Machine learning
- **BioPython 1.83**: Bioinformatics tools
- **joblib 1.3.2**: Model serialization
- **ReportLab 4.0.7**: PDF generation
- **Werkzeug 3.0.1**: WSGI utilities

## Medical Disclaimer

⚠️ **IMPORTANT**: This application is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Nithish Vaduganathan

## Acknowledgments

- Inspired by genomic research and bioinformatics applications
- Built with Flask and scikit-learn
- Bootstrap for responsive UI design

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an academic project demonstrating the application of machine learning to genomic data analysis. The predictions are based on synthetic data patterns and should not be used for actual medical diagnosis.