// Testing Interface - Financial & Earthquake Anomaly Detection

let selectedTestModel = 'cnn';
let selectedDataset = 'financial';
let selectedFinancialDataset = 'tsla';
let anomalyChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initChart();
    setupEventListeners();
    checkExistingResources();
});

// Setup all event listeners
function setupEventListeners() {
    // Dataset selection
    document.querySelectorAll('[data-test-dataset]').forEach(card => {
        card.addEventListener('click', function() {
            document.querySelectorAll('[data-test-dataset]').forEach(c => c.classList.remove('selected'));
            this.classList.add('selected');
            selectedDataset = this.dataset.testDataset;
            
            // Show/hide financial dataset section and threshold params
            const financialSection = document.getElementById('financialDatasetSection');
            const thresholdSection = document.getElementById('thresholdSection');
            const financialTestControls = document.getElementById('financialTestControls');
            
            if (selectedDataset === 'financial') {
                if (financialSection) financialSection.style.display = 'block';
                if (thresholdSection) thresholdSection.style.display = 'none';
                if (financialTestControls) financialTestControls.style.display = 'block';
            } else {
                if (financialSection) financialSection.style.display = 'none';
                if (thresholdSection) thresholdSection.style.display = 'block';
                if (financialTestControls) financialTestControls.style.display = 'none';
            }
            
            console.log('Dataset selected:', selectedDataset);
        });
    });
    
    // Financial dataset selection
    document.querySelectorAll('[data-financial-dataset]').forEach(card => {
        card.addEventListener('click', function() {
            document.querySelectorAll('[data-financial-dataset]').forEach(c => c.classList.remove('selected'));
            this.classList.add('selected');
            selectedFinancialDataset = this.dataset.financialDataset;
            console.log('Financial dataset selected:', selectedFinancialDataset);
        });
    });
    
    // Model selection
    document.querySelectorAll('[data-test-model]').forEach(card => {
        card.addEventListener('click', function() {
            document.querySelectorAll('[data-test-model]').forEach(c => c.classList.remove('selected'));
            this.classList.add('selected');
            selectedTestModel = this.dataset.testModel;
            console.log('Model selected:', selectedTestModel);
        });
    });
    
    // Threshold method change (for earthquake only)
    const thresholdMethod = document.getElementById('thresholdMethod');
    if (thresholdMethod) {
        thresholdMethod.addEventListener('change', function() {
            const method = this.value;
            
            // Show/hide relevant parameters for earthquake
            const percentileParams = document.getElementById('percentileParams');
            const customParams = document.getElementById('customParams');
            
            if (percentileParams) {
                percentileParams.style.display = method === 'percentile' ? 'block' : 'none';
            }
            if (customParams) {
                customParams.style.display = method === 'custom' ? 'block' : 'none';
            }
        });
    }
    
    // Slider updates
    const madKSlider = document.getElementById('madK');
    if (madKSlider) {
        madKSlider.addEventListener('input', function() {
            document.getElementById('madKValue').textContent = this.value;
        });
    }
    
    const stressSlider = document.getElementById('stressPercentile');
    if (stressSlider) {
        stressSlider.addEventListener('input', function() {
            document.getElementById('stressPercentileValue').textContent = this.value;
        });
    }
    
    // Run prediction button
    const runBtn = document.getElementById('runPrediction');
    if (runBtn) {
        runBtn.addEventListener('click', runPrediction);
    }
}

// Check for existing resources on load
async function checkExistingResources() {
    try {
        const response = await fetch('/api/check_existing_resources');
        const resources = await response.json();
        
        console.log('Available resources:', resources);
    } catch (error) {
        console.error('Error checking resources:', error);
    }
}

// Initialize Anomaly Chart
function initChart() {
    const ctx = document.getElementById('anomalyChart').getContext('2d');
    
    anomalyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Price / Signal',
                    data: [],
                    borderColor: '#4A90E2',
                    backgroundColor: 'rgba(74, 144, 226, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Anomaly',
                    data: [],
                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                    borderColor: '#DC2626',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    pointStyle: 'circle',
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#94A3B8',
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.dataset.label === 'Anomaly') {
                                return `Anomaly: ${context.parsed.y.toFixed(2)}`;
                            }
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date',
                        color: '#94A3B8'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#94A3B8',
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Value',
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

// Run prediction
async function runPrediction() {
    try {
        console.log('Starting prediction...');
        
        // Get parameters
        const thresholdMethod = document.getElementById('thresholdMethod').value;
        const madK = parseFloat(document.getElementById('madK').value);
        const stressPercentile = parseInt(document.getElementById('stressPercentile').value);
        const percentile = parseInt(document.getElementById('thresholdPercentile').value);
        const customValue = parseFloat(document.getElementById('customThreshold').value);
        
        // Prepare request
        const requestData = {
            model_type: selectedTestModel,
            dataset_type: selectedDataset,
            threshold_method: thresholdMethod,
            mad_k: madK,
            stress_percentile: stressPercentile,
            percentile: percentile,
            custom_value: customValue
        };
        
        if (selectedDataset === 'financial') {
            const stockTicker = document.getElementById('testStockTicker')?.value || 'TSLA';
            const startDate = document.getElementById('testStartDate')?.value || '2019-01-01';
            const endDate = document.getElementById('testEndDate')?.value || '2021-12-31';
            
            requestData.stock_ticker = stockTicker;
            requestData.start_date = startDate;
            requestData.end_date = endDate;
        }
        
        console.log('Request data:', requestData);
        
        // Show loading
        const runBtn = document.getElementById('runPrediction');
        runBtn.disabled = true;
        runBtn.innerHTML = '<span>⏳ Running...</span>';
        
        // Make API call
        const response = await fetch('/api/run_inference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log('Inference successful:', result);
            
            // Display results based on dataset type
            if (result.dataset_type === 'financial') {
                displayFinancialResults(result);
            } else {
                displayEarthquakeResults(result);
            }
            
            // Success feedback
            runBtn.innerHTML = '<span>✓</span> Complete';
            runBtn.style.background = '#10B981';
            setTimeout(() => {
                runBtn.innerHTML = '<span class="search-icon">🔍</span> Run Prediction';
                runBtn.style.background = '';
                runBtn.disabled = false;
            }, 2000);
        } else {
            alert('Error: ' + result.error);
            console.error('Inference error:', result);
            runBtn.innerHTML = '<span class="search-icon">🔍</span> Run Prediction';
            runBtn.disabled = false;
        }
        
    } catch (error) {
        console.error('Error running prediction:', error);
        alert('Failed to run prediction: ' + error.message);
        const runBtn = document.getElementById('runPrediction');
        runBtn.innerHTML = '<span class="search-icon">🔍</span> Run Prediction';
        runBtn.disabled = false;
    }
}

// Display financial results
function displayFinancialResults(result) {
    console.log('Displaying financial results:', result);
    
    // Show metrics grid with financial metrics
    const metricsGrid = document.getElementById('metricsGrid');
    metricsGrid.style.display = 'grid';
    
    // Hide earthquake metrics, show financial metrics
    document.querySelectorAll('.earthquake-metrics').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.financial-metrics').forEach(el => el.style.display = 'block');
    
    // Update financial metrics
    document.getElementById('anomalyCount').textContent = result.metrics.anomaly_count;
    document.getElementById('anomalyRate').textContent = (result.metrics.anomaly_rate * 100).toFixed(2) + '%';
    document.getElementById('maeValue').textContent = result.metrics.mae.toFixed(6);
    document.getElementById('thresholdValue').textContent = result.metrics.threshold.toExponential(4);
    document.getElementById('anomalies2021').textContent = result.metrics.anomalies_2021 + ' / ' + result.metrics.samples_2021;
    document.getElementById('stressThreshold').textContent = result.metrics.stress_threshold.toFixed(6);
    
    // Always show recommended threshold for financial data
    const recommendedThreshold = document.getElementById('recommendedThreshold');
    const recommendedValue = document.getElementById('recommendedValue');
    if (recommendedThreshold && recommendedValue) {
        recommendedThreshold.style.display = 'block';
        recommendedValue.textContent = 
            `Threshold (k=${result.metrics.threshold_method_params?.k || 2.5}): ${result.metrics.threshold.toExponential(4)}`;
    }
    
    // Update chart with price data
    if (result.chart_data) {
        updateFinancialChart(result.chart_data);
    }
}

// Update chart with financial data
function updateFinancialChart(chartData) {
    console.log('Updating chart with financial data:', chartData);
    
    // Prepare data
    const dates = chartData.dates;
    const prices = chartData.prices;
    const anomalies = chartData.anomalies;
    
    // Create anomaly points
    const anomalyPoints = [];
    for (let i = 0; i < dates.length; i++) {
        if (anomalies[i] === 1) {
            anomalyPoints.push({
                x: dates[i],
                y: prices[i]
            });
        }
    }
    
    // Update chart
    anomalyChart.data.labels = dates;
    anomalyChart.data.datasets[0].data = prices;
    anomalyChart.data.datasets[1].data = anomalyPoints;
    
    // Update y-axis label
    anomalyChart.options.scales.y.title.text = 'Price ($)';
    
    anomalyChart.update();
}

// Display earthquake results (existing implementation)
function displayEarthquakeResults(result) {
    console.log('Displaying earthquake results:', result);
    
    // Show metrics grid with earthquake metrics
    const metricsGrid = document.getElementById('metricsGrid');
    metricsGrid.style.display = 'grid';
    
    // Hide financial metrics, show earthquake metrics
    document.querySelectorAll('.financial-metrics').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.earthquake-metrics').forEach(el => el.style.display = 'block');
    
    // Update earthquake metrics
    document.getElementById('tpValue').textContent = result.metrics.tp;
    document.getElementById('fpValue').textContent = result.metrics.fp;
    document.getElementById('precisionValue').textContent = result.metrics.precision.toFixed(3);
    document.getElementById('recallValue').textContent = result.metrics.recall.toFixed(3);
    document.getElementById('f1TestValue').textContent = result.metrics.f1_score.toFixed(3);
    document.getElementById('aucTestValue').textContent = result.metrics.auc.toFixed(3);
}
