// Main JavaScript for DNA Disease Prediction System

// Store last prediction result
let lastPredictionResult = null;

// Sample DNA sequences
const sampleSequences = {
    diabetes: 'ATCGATCGATCGATCGGCGCGCGCGCGCGCGCGCGCGCGCGCGATCGATCGATCGATCGATCGATCGCGCGCGCGCGATCGATCGATCGATCGATCGATCGATCGCGCGCGCGCGCGCGCGCATCGATCGATCGATCGATCGATCGCGCGCGCGCGATCGATCGATCG',
    cancer: 'ATCGATCGATCGATCGATCGAAATTTAAAATTTAAAATTTAAAATTTAAAATCGATCGATCGATCGATCGATCGAAATTTAAAATTTAAAATCGATCGATCGATCGATCGAAATTTAAAATTTAAAATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG',
    healthy: 'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG'
};

// Load sample sequence
function loadSample(type) {
    const textarea = document.getElementById('sequenceInput');
    textarea.value = sampleSequences[type];
    
    // Switch to paste tab
    const pasteTab = document.getElementById('paste-tab');
    pasteTab.click();
}

// Handle form submission
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading spinner
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('submitBtn').disabled = true;
    
    try {
        const formData = new FormData();
        
        // Check which input method is active
        const activeTab = document.querySelector('.tab-pane.active');
        
        if (activeTab.id === 'paste') {
            const sequence = document.getElementById('sequenceInput').value;
            if (!sequence.trim()) {
                throw new Error('Please enter a DNA sequence');
            }
            formData.append('sequence', sequence);
        } else {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files.length) {
                throw new Error('Please select a file');
            }
            formData.append('file', fileInput.files[0]);
        }
        
        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Prediction failed');
        }
        
        // Store result for PDF generation
        lastPredictionResult = result;
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        document.getElementById('loadingSpinner').style.display = 'none';
        document.getElementById('submitBtn').disabled = false;
    }
});

// Display prediction results
function displayResults(result) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').classList.add('fade-in-up');
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Display predicted disease
    document.getElementById('predictedDisease').textContent = result.disease;
    
    // Display risk level badge
    const riskBadge = document.getElementById('riskBadge');
    riskBadge.textContent = `${result.risk_level} Risk`;
    riskBadge.className = 'badge fs-5';
    
    if (result.risk_level === 'High') {
        riskBadge.classList.add('bg-danger');
    } else if (result.risk_level === 'Medium') {
        riskBadge.classList.add('bg-warning');
    } else {
        riskBadge.classList.add('bg-success');
    }
    
    // Display confidence score
    const confidencePercent = (result.confidence * 100).toFixed(2);
    document.getElementById('confidenceText').textContent = `${confidencePercent}%`;
    
    const confidenceBar = document.getElementById('confidenceBar');
    confidenceBar.style.width = `${confidencePercent}%`;
    
    if (result.confidence >= 0.7) {
        confidenceBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-success';
    } else if (result.confidence >= 0.4) {
        confidenceBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-warning';
    } else {
        confidenceBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-danger';
    }
    
    // Display top predictions
    const topPredictionsDiv = document.getElementById('topPredictions');
    topPredictionsDiv.innerHTML = '';
    
    result.top_predictions.forEach((pred, index) => {
        const [disease, prob] = pred;
        const probPercent = (prob * 100).toFixed(2);
        
        const probHtml = `
            <div class="prob-bar">
                <div class="d-flex justify-content-between mb-1">
                    <strong>${disease}</strong>
                    <span>${probPercent}%</span>
                </div>
                <div class="progress" style="height: 25px;">
                    <div class="progress-bar ${index === 0 ? 'bg-primary' : 'bg-secondary'}" 
                         style="width: ${probPercent}%">
                    </div>
                </div>
            </div>
        `;
        topPredictionsDiv.innerHTML += probHtml;
    });
    
    // Display sequence statistics
    const statsDiv = document.getElementById('sequenceStats');
    statsDiv.innerHTML = '';
    
    const stats = result.stats;
    const statsHtml = `
        <div class="col-md-3">
            <div class="stat-card">
                <h6>Length</h6>
                <div class="stat-value">${stats.length}</div>
                <small class="text-muted">nucleotides</small>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stat-card">
                <h6>GC Content</h6>
                <div class="stat-value">${stats.gc_content.toFixed(2)}%</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stat-card">
                <h6>AT Content</h6>
                <div class="stat-value">${stats.at_content.toFixed(2)}%</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stat-card">
                <h6>Nucleotide Counts</h6>
                <div class="text-muted small">
                    A: ${stats.a_count} | T: ${stats.t_count}<br>
                    C: ${stats.c_count} | G: ${stats.g_count}
                </div>
            </div>
        </div>
    `;
    statsDiv.innerHTML = statsHtml;
}

// Download PDF report
async function downloadReport() {
    if (!lastPredictionResult) {
        alert('No prediction results available');
        return;
    }
    
    try {
        const response = await fetch('/download-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(lastPredictionResult)
        });
        
        if (!response.ok) {
            throw new Error('Failed to generate report');
        }
        
        // Download the PDF
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dna_prediction_report_${new Date().getTime()}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Reset form
function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('inputSection').scrollIntoView({ behavior: 'smooth' });
    lastPredictionResult = null;
}

// Show history
async function showHistory() {
    try {
        const response = await fetch('/history');
        const history = await response.json();
        
        if (!response.ok) {
            throw new Error('Failed to load history');
        }
        
        const tableBody = document.getElementById('historyTableBody');
        tableBody.innerHTML = '';
        
        if (history.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6" class="text-center">No prediction history available</td></tr>';
        } else {
            history.forEach(item => {
                const row = `
                    <tr>
                        <td>${new Date(item.timestamp).toLocaleString()}</td>
                        <td>${item.sequence_length}</td>
                        <td>${item.gc_content.toFixed(2)}%</td>
                        <td>${item.predicted_disease}</td>
                        <td>${(item.confidence * 100).toFixed(2)}%</td>
                        <td>
                            <span class="badge bg-${item.risk_level === 'High' ? 'danger' : item.risk_level === 'Medium' ? 'warning' : 'success'}">
                                ${item.risk_level}
                            </span>
                        </td>
                    </tr>
                `;
                tableBody.innerHTML += row;
            });
        }
        
        // Hide input and results, show history
        document.getElementById('inputSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('historySection').style.display = 'block';
        document.getElementById('historySection').scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Hide history
function hideHistory() {
    document.getElementById('historySection').style.display = 'none';
    document.getElementById('inputSection').style.display = 'block';
    if (lastPredictionResult) {
        document.getElementById('resultsSection').style.display = 'block';
    }
    document.getElementById('inputSection').scrollIntoView({ behavior: 'smooth' });
}
