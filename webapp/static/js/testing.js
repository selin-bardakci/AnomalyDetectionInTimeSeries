// Testing Interface - Run Predictions and Visualize Anomalies

let selectedTestModel = 'cnn';
let uploadedTestFile = null;
let useExistingData = false;
let anomalyChart = null;

// Check for existing resources on load
async function checkExistingResources() {
    try {
        const response = await fetch('/api/check_existing_resources');
        const resources = await response.json();
        
        // Update UI based on available resources
        if (resources.test_data_available) {
            document.getElementById('useExistingData').disabled = false;
            document.getElementById('useExistingData').title = 'Use project test data (available)';
        } else {
            document.getElementById('useExistingData').disabled = true;
            document.getElementById('useExistingData').title = 'Project test data not found';
            document.getElementById('useExistingData').style.opacity = '0.5';
        }
        
        console.log('Available resources:', resources);
    } catch (error) {
        console.error('Error checking resources:', error);
    }
}

// Initialize Anomaly Chart
function initChart() {
    const ctx = document.getElementById('anomalyChart').getContext('2d');
    
    anomalyChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Signal',
                    data: [],
                    borderColor: '#94A3B8',
                    backgroundColor: 'rgba(148, 163, 184, 0.1)',
                    showLine: true,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Anomaly Region',
                    data: [],
                    backgroundColor: 'rgba(239, 68, 68, 0.4)',
                    borderColor: '#EF4444',
                    pointRadius: 5,
                    pointHoverRadius: 7
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Window ${context.parsed.x}: ${context.parsed.y.toFixed(6)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Window Index',
                        color: '#94A3B8'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#94A3B8'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Reconstruction Error (MSE)',
                        color: '#94A3B8'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#94A3B8'
                    }
                }
            }
        }
    });
}

// Model Selection
document.querySelectorAll('[data-test-model]').forEach(card => {
    card.addEventListener('click', function() {
        document.querySelectorAll('[data-test-model]').forEach(c => c.classList.remove('selected'));
        this.classList.add('selected');
        selectedTestModel = this.dataset.testModel;
    });
});

// Learning Rate Slider
const learningRateSlider = document.getElementById('learningRate');
const learningRateValue = document.getElementById('learningRateValue');

learningRateSlider.addEventListener('input', function() {
    learningRateValue.textContent = parseFloat(this.value).toFixed(4);
});

// Use Existing Data Button
document.getElementById('useExistingData').addEventListener('click', function() {
    useExistingData = true;
    uploadedTestFile = null;
    
    this.style.background = 'rgba(45, 212, 191, 0.2)';
    this.style.borderColor = '#2DD4BF';
    
    alert('✓ Using existing project test data (custom_window_test.npz)');
});

// File Upload
document.getElementById('testFile').addEventListener('change', async function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Reset existing data button
    useExistingData = false;
    document.getElementById('useExistingData').style.background = '';
    document.getElementById('useExistingData').style.borderColor = '';
    
    try {
        const response = await fetch('/api/upload_test_data', {
            method: 'POST',
            body: formData
        });
       Check if data is available
    if (!useExistingData && !uploadedTestFile) {
        alert('⚠️ Please upload test data or use existing project data');
        return;
    }
    
    const threshold = parseInt(document.getElementById('thresholdPercentile').value);
    
    this.disabled = true;
    this.innerHTML = '<span>⏳</span> Running...';
    
    try {
        const response = await fetch('/api/run_prediction', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_type: selectedTestModel,
                test_filename: uploadedTestFile,
                threshold_percentile: threshold,
                use_existing_data: useExistingDatad
    const testFile = uploadedTestFile || 'test_custom_window_test.npz';
    const threshold = parseInt(document.getElementById('thresholdPercentile').value);
    
    this.disabled = true;
    this.innerHTML = '<span>⏳</span> Running...';
    
    try {
        const response = await fetch('/api/run_prediction', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_type: selectedTestModel,
                test_filename: testFile,
                threshold_percentile: threshold
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update chart
            const signalData = data.errors.map((err, idx) => ({x: idx, y: err}));
            const anomalyData = data.anomaly_indices
                .filter(idx => idx < data.errors.length)
                .map(idx => ({x: idx, y: data.errors[idx]}));
            
            anomalyChart.data.datasets[0].data = signalData;
            anomalyChart.data.datasets[1].data = anomalyData;
            anomalyChart.update();
            
            // Show metrics if available
            if (data.metrics && Object.keys(data.metrics).length > 0) {
                document.getElementById('metricsGrid').style.display = 'grid';
                
                document.getElementById('tpValue').textContent = data.metrics.tp || '--';
                document.getElementById('fpValue').textContent = data.metrics.fp || '--';
                document.getElementById('precisionValue').textContent = 
                    data.metrics.precision ? data.metrics.precision.toFixed(3) : '--';
                document.getElementById('recallValue').textContent = 
                    data.metrics.recall ? data.metrics.recall.toFixed(3) : '--';
                document.getElementById('f1TestValue').textContent = 
                    data.metrics.f1_score ? data.metrics.f1_score.toFixed(4) : '--';
                document.getElementById('aucTestValue').textContent = 
                    data.metrics.auc ? data.metrics.auc.toFixed(4) : '--';
            }
            
            this.innerHTML = '<span>✓</span> Complete';
            this.style.background = '#10B981';
            
            setTimeout(() => {
                this.innerHTML = '<span class="search-icon">🔍</span> Run Prediction';
                this.style.background = '#10B981';
                this.disabled = false;
            }, 2000);
            
        } else {
            alert('Prediction failed: ' + data.error);
            this.innerHTML = '<span class="search-icon">🔍</span> Run Prediction';
            this.disabled = false;
        }
    } catch (error) {
        alert('Prediction error: ' + error.message);
   Initialize on page load
window.addEventListener('DOMContentLoaded', function() {
    initChart();
    checkExistingResources();   });
        
        // Silently handle - user can upload their own data
    } catch (error) {
        console.log('Default data not available, user can upload their own');
    }
});
