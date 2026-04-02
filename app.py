"""
DNA Disease Risk Prediction – Flask Application
Uses ensemble (XGBoost + LightGBM) model trained on synthetic_dna_dataset.csv
Target: Disease_Risk (High / Medium / Low)
"""

from flask import Flask, render_template, request, jsonify, send_file
import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime
import sqlite3
from io import BytesIO

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from utils.validator import (
    validate_dna_sequence,
    clean_dna_sequence,
    parse_fasta,
    get_sequence_stats
)
from utils.preprocessing import sequence_to_kmer_string

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH      = 'model/disease_model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'
DB_PATH         = 'predictions.db'

# Global artifacts (loaded at startup)
artifacts = None   # full dict saved by train_model.py


# ── Load model ─────────────────────────────────────────────────────────────────

def load_model():
    """Load the trained model artifacts."""
    global artifacts
    if os.path.exists(MODEL_PATH):
        try:
            loaded = joblib.load(MODEL_PATH)
            if isinstance(loaded, dict) and 'model' in loaded:
                artifacts = loaded
                print("✓ New-format model loaded successfully!")
                print(f"  Classes: {artifacts.get('classes')}")
                print(f"  Test accuracy: {artifacts.get('accuracy', 0)*100:.1f}%")
                return True
            else:
                # Backward-compatible old model (plain sklearn estimator)
                artifacts = {'model': loaded, 'classes': None,
                             'label_encoder': None, 'scaler': None,
                             'kmer_vectorizer': None, 'has_sequence': True}
                print("✓ Legacy model loaded (k-mer only mode)")
                return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
    else:
        print("✗ Model not found — please run: python model/train_model.py")
    return False


# ── Database ───────────────────────────────────────────────────────────────────

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sequence_length INTEGER,
            gc_content REAL,
            predicted_risk TEXT,
            confidence REAL,
            class_label TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_prediction(seq_len, gc, risk, confidence, class_label='N/A'):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions
                (timestamp, sequence_length, gc_content, predicted_risk, confidence, class_label)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), seq_len, gc, risk, confidence, class_label))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB error: {e}")


# ── Feature extraction ────────────────────────────────────────────────────────

def build_feature_row(stats, sequence, class_label='Human'):
    """
    Build a numeric feature row that EXACTLY matches the training schema in
    train_model.py::extract_numeric_features() — 23 hand-crafted features.
    """
    from scipy.sparse import hstack, csr_matrix

    a = int(stats.get('a_count', 0))
    t = int(stats.get('t_count', 0))
    c = int(stats.get('c_count', 0))
    g = int(stats.get('g_count', 0))
    gc_content = float(stats.get('gc_content', (g + c) / max(a+t+c+g,1) * 100))
    at_content = 100.0 - gc_content
    kmer_3_freq   = 0.5      # sensible default for unknown sequences
    mutation_flag = 0        # unknown → assume no mutation

    # Encode class label with the same LabelEncoder used at training
    le_class = artifacts.get('class_label_encoder')
    known_classes = list(le_class.classes_) if le_class is not None else \
                    ['Bacteria', 'Human', 'Plant', 'Virus']
    if class_label not in known_classes:
        class_label = 'Human'
    class_enc = int(le_class.transform([class_label])[0]) if le_class is not None \
                else known_classes.index(class_label)

    # ── Engineered features (must match train_model.py order) ──────────────
    eps = 1e-6
    at_gc_ratio      = at_content    / (gc_content   + eps)
    a_t_ratio        = a             / (t             + eps)
    c_g_ratio        = c             / (g             + eps)
    pur_pyr          = (a + g)       / (c + t         + eps)
    gc_x_kmer        = gc_content    * kmer_3_freq
    mutation_x_gc    = mutation_flag * gc_content
    a_plus_c         = a + c
    t_plus_g         = t + g
    a_minus_t        = a - t
    c_minus_g        = c - g
    kmer_sq          = kmer_3_freq ** 2
    gc_sq            = (gc_content / 100) ** 2
    class_x_mut      = class_enc * mutation_flag
    kmer_x_mut       = kmer_3_freq * mutation_flag

    # Feature vector (23 values — must match training column order exactly)
    num_row = np.array([[
        gc_content, at_content,
        a, t, c, g,
        kmer_3_freq, mutation_flag,
        class_enc,
        at_gc_ratio, a_t_ratio, c_g_ratio, pur_pyr,
        gc_x_kmer, mutation_x_gc,
        a_plus_c, t_plus_g, a_minus_t, c_minus_g,
        kmer_sq, gc_sq, class_x_mut, kmer_x_mut,
    ]], dtype=float)

    # Scale with the same StandardScaler fitted during training
    scaler = artifacts.get('scaler')
    num_scaled = scaler.transform(num_row) if scaler is not None else num_row

    # Combine with TF-IDF k-mer features
    kmer_vectorizer = artifacts.get('kmer_vectorizer')
    if kmer_vectorizer is not None and sequence:
        kmer_str  = sequence_to_kmer_string(sequence.upper(), k=3)
        kmer_feat = kmer_vectorizer.transform([kmer_str])
        X = hstack([csr_matrix(num_scaled), kmer_feat])
    else:
        X = csr_matrix(num_scaled)

    return X


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_risk(sequence, class_label='Human'):
    """Predict Disease Risk from a DNA sequence string."""
    # Clean & validate
    sequence = clean_dna_sequence(sequence)
    is_valid, message = validate_dna_sequence(sequence)
    if not is_valid:
        return {'error': message}

    stats = get_sequence_stats(sequence)
    model = artifacts['model']
    le_label = artifacts.get('label_encoder')

    # If new-format model
    if le_label is not None:
        X = build_feature_row(stats, sequence, class_label)
        raw_pred = model.predict(X)[0]
        raw_proba = model.predict_proba(X)[0]
        classes = list(le_label.classes_)
        predicted_class = classes[raw_pred]
        prob_dict = {c: float(p) for c, p in zip(classes, raw_proba)}
    else:
        # Legacy k-mer–only model
        vectorizer = artifacts.get('kmer_vectorizer') or joblib.load(VECTORIZER_PATH)
        kmer_str = sequence_to_kmer_string(sequence, k=3)
        X = vectorizer.transform([kmer_str])
        raw_pred = model.predict(X)[0]
        raw_proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        predicted_class = str(raw_pred)
        prob_dict = {c: float(p) for c, p in zip(classes, raw_proba)}

    confidence = max(prob_dict.values())
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    save_prediction(
        stats['length'], stats['gc_content'],
        predicted_class, confidence, class_label
    )

    return {
        'disease': predicted_class,               # "disease" key kept for compat
        'risk_level': predicted_class,            # redundant alias
        'confidence': float(confidence),
        'probabilities': prob_dict,
        'top_predictions': sorted_probs[:3],
        'stats': stats,
        'accuracy': artifacts.get('accuracy', 0),
        'cv_mean': artifacts.get('cv_mean', 0),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    model_info = {}
    if artifacts:
        model_info = {
            'accuracy': round(artifacts.get('accuracy', 0) * 100, 2),
            'cv_mean': round(artifacts.get('cv_mean', 0) * 100, 2),
            'classes': artifacts.get('classes', []),
        }
    return render_template('index.html', model_info=model_info)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if artifacts is None or artifacts.get('model') is None:
            return jsonify({'error': 'Model not loaded. Run: python model/train_model.py'}), 500

        sequence = None
        class_label = request.form.get('class_label', 'Human')

        # File upload
        if 'file' in request.files:
            f = request.files['file']
            if f.filename:
                content = f.read().decode('utf-8')
                sequence = parse_fasta(content) if content.startswith('>') else content

        # Pasted sequence
        if not sequence and 'sequence' in request.form:
            sequence = request.form['sequence']

        if not sequence:
            return jsonify({'error': 'No DNA sequence provided'}), 400

        result = predict_risk(sequence, class_label)
        if 'error' in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/history')
def history():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, sequence_length, gc_content,
                   predicted_risk, confidence, class_label
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        rows = cursor.fetchall()
        conn.close()
        return jsonify([{
            'timestamp': r[0], 'sequence_length': r[1], 'gc_content': r[2],
            'predicted_disease': r[3], 'confidence': r[4], 'risk_level': r[3],
            'class_label': r[5]
        } for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model-info')
def model_info():
    if artifacts is None:
        return jsonify({'status': 'not loaded'})
    return jsonify({
        'accuracy': artifacts.get('accuracy', 0),
        'cv_mean': artifacts.get('cv_mean', 0),
        'cv_std': artifacts.get('cv_std', 0),
        'classes': artifacts.get('classes', []),
    })


@app.route('/download-report', methods=['POST'])
def download_report():
    if not HAS_REPORTLAB:
        return jsonify({'error': 'ReportLab not installed'}), 500
    try:
        data = request.json
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'Title', parent=styles['Heading1'], fontSize=22,
            textColor=rl_colors.HexColor('#0f172a'), spaceAfter=20, alignment=1
        )
        elements.append(Paragraph("DNA Disease Risk Prediction Report", title_style))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("Prediction Results", styles['Heading2']))
        elements.append(Spacer(1, 0.1 * inch))

        risk = data.get('disease', data.get('risk_level', 'N/A'))
        results_data = [
            ['Predicted Risk Level:', risk],
            ['Confidence:', f"{data.get('confidence', 0) * 100:.2f}%"],
            ['Model Accuracy:', f"{data.get('accuracy', 0) * 100:.1f}%"],
        ]
        t = Table(results_data, colWidths=[2 * inch, 4 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), rl_colors.HexColor('#f1f5f9')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph("Sequence Statistics", styles['Heading2']))
        elements.append(Spacer(1, 0.1 * inch))
        stats = data.get('stats', {})
        stats_data = [
            ['Length:',      f"{stats.get('length', 0)} nt"],
            ['GC Content:',  f"{stats.get('gc_content', 0):.2f}%"],
            ['AT Content:',  f"{stats.get('at_content', 0):.2f}%"],
            ['Adenine (A):', str(stats.get('a_count', 0))],
            ['Thymine (T):', str(stats.get('t_count', 0))],
            ['Cytosine (C):', str(stats.get('c_count', 0))],
            ['Guanine (G):', str(stats.get('g_count', 0))],
        ]
        t2 = Table(stats_data, colWidths=[2 * inch, 4 * inch])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), rl_colors.HexColor('#f1f5f9')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ]))
        elements.append(t2)
        elements.append(Spacer(1, 0.4 * inch))

        disc_style = ParagraphStyle(
            'Disc', parent=styles['Normal'], fontSize=8,
            textColor=rl_colors.HexColor('#64748b'), leading=12
        )
        elements.append(Paragraph(
            "<b>Medical Disclaimer:</b> This prediction is for educational and research "
            "purposes only. It should not be used as a substitute for professional medical "
            "advice, diagnosis, or treatment.", disc_style
        ))

        doc.build(elements)
        buffer.seek(0)
        return send_file(
            buffer, as_attachment=True,
            download_name=f'dna_risk_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_database()
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
