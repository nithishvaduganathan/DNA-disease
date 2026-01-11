"""
DNA Disease Prediction Flask Application
Main application file with routes and prediction logic
"""

from flask import Flask, render_template, request, jsonify, send_file
import joblib
import os
from datetime import datetime
import sqlite3
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch

from utils.validator import (
    validate_dna_sequence, 
    clean_dna_sequence, 
    parse_fasta,
    get_sequence_stats
)
from utils.preprocessing import sequence_to_kmer_string

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and vectorizer
MODEL_PATH = 'model/disease_model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'
DB_PATH = 'predictions.db'

model = None
vectorizer = None


def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    global model, vectorizer
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model and vectorizer loaded successfully!")
        return True
    else:
        print("Model or vectorizer not found. Please train the model first.")
        return False


def init_database():
    """Initialize SQLite database for prediction history"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sequence_length INTEGER,
            gc_content REAL,
            predicted_disease TEXT,
            confidence REAL,
            risk_level TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_prediction_to_db(sequence_length, gc_content, disease, confidence, risk_level):
    """Save prediction to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (timestamp, sequence_length, gc_content, predicted_disease, confidence, risk_level)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), sequence_length, gc_content, disease, confidence, risk_level))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving to database: {e}")


def classify_risk_level(confidence):
    """Classify risk level based on confidence score"""
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.4:
        return "Medium"
    else:
        return "Low"


def predict_disease(sequence):
    """
    Predict disease from DNA sequence
    
    Returns:
        dict: Prediction results including disease, confidence, probabilities
    """
    # Clean sequence
    sequence = clean_dna_sequence(sequence)
    
    # Validate sequence
    is_valid, message = validate_dna_sequence(sequence)
    if not is_valid:
        return {'error': message}
    
    # Get sequence statistics
    stats = get_sequence_stats(sequence)
    
    # Convert to k-mer representation
    kmer_string = sequence_to_kmer_string(sequence, k=3)
    
    # Transform using vectorizer
    features = vectorizer.transform([kmer_string])
    
    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    # Get all class probabilities
    classes = model.classes_
    prob_dict = {disease: float(prob) for disease, prob in zip(classes, probabilities)}
    
    # Sort probabilities
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Classify risk level
    risk_level = classify_risk_level(confidence)
    
    # Save to database
    save_prediction_to_db(
        stats['length'],
        stats['gc_content'],
        prediction,
        confidence,
        risk_level
    )
    
    return {
        'disease': prediction,
        'confidence': float(confidence),
        'risk_level': risk_level,
        'probabilities': prob_dict,
        'top_predictions': sorted_probs[:3],
        'stats': stats
    }


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Check if model is loaded
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded. Please contact administrator.'}), 500
        
        sequence = None
        
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                content = file.read().decode('utf-8')
                # Check if FASTA format
                if content.startswith('>'):
                    sequence = parse_fasta(content)
                else:
                    sequence = content
        
        # Check if sequence was pasted
        if not sequence and 'sequence' in request.form:
            sequence = request.form['sequence']
        
        if not sequence:
            return jsonify({'error': 'No DNA sequence provided'}), 400
        
        # Make prediction
        result = predict_disease(sequence)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/history')
def history():
    """Get prediction history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, sequence_length, gc_content, predicted_disease, confidence, risk_level
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        history_data = []
        for row in rows:
            history_data.append({
                'timestamp': row[0],
                'sequence_length': row[1],
                'gc_content': row[2],
                'predicted_disease': row[3],
                'confidence': row[4],
                'risk_level': row[5]
            })
        
        return jsonify(history_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download-report', methods=['POST'])
def download_report():
    """Generate and download PDF report"""
    try:
        data = request.json
        
        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=1  # Center
        )
        elements.append(Paragraph("DNA Disease Prediction Report", title_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Date
        date_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        elements.append(Paragraph(date_text, styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Prediction Results
        elements.append(Paragraph("Prediction Results", styles['Heading2']))
        elements.append(Spacer(1, 0.2*inch))
        
        results_data = [
            ['Predicted Disease:', data.get('disease', 'N/A')],
            ['Confidence:', f"{data.get('confidence', 0) * 100:.2f}%"],
            ['Risk Level:', data.get('risk_level', 'N/A')],
        ]
        
        results_table = Table(results_data, colWidths=[2*inch, 4*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(results_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Sequence Statistics
        elements.append(Paragraph("Sequence Statistics", styles['Heading2']))
        elements.append(Spacer(1, 0.2*inch))
        
        stats = data.get('stats', {})
        stats_data = [
            ['Length:', f"{stats.get('length', 0)} nucleotides"],
            ['GC Content:', f"{stats.get('gc_content', 0):.2f}%"],
            ['AT Content:', f"{stats.get('at_content', 0):.2f}%"],
            ['A Count:', str(stats.get('a_count', 0))],
            ['T Count:', str(stats.get('t_count', 0))],
            ['C Count:', str(stats.get('c_count', 0))],
            ['G Count:', str(stats.get('g_count', 0))],
        ]
        
        stats_table = Table(stats_data, colWidths=[2*inch, 4*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 0.5*inch))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#7f8c8d'),
            leading=12
        )
        disclaimer_text = """
        <b>Medical Disclaimer:</b> This prediction is for educational and research purposes only. 
        It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions 
        you may have regarding a medical condition.
        """
        elements.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'dna_prediction_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Load model
    if not load_model_and_vectorizer():
        print("\n" + "="*70)
        print("WARNING: Model not found!")
        print("Please run the following command to train the model:")
        print("  python model/train_model.py")
        print("="*70 + "\n")
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)
