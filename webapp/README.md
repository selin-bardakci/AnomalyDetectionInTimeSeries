# CHRONOS.AI - Advanced Anomaly Detection Platform

A professional web application for training and testing time series anomaly detection models, designed for seismic and financial data analysis.

![CHRONOS.AI](https://img.shields.io/badge/CHRONOS.AI-Advanced%20Anomaly%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Flask](https://img.shields.io/badge/Flask-3.0-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)

## Features

### 🏠 **Homepage**
- Modern dark-themed UI with animated time series visualization
- Professional branding and navigation
- Hero section with call-to-action buttons

### 🔬 **Training Laboratory**
- **Dataset Selection**: Choose between Earthquake Data (seismic magnitudes) or Financial Market (S&P 500 volatility)
- **Model Architecture**: Select from three models:
  - **iTransformer** (SOTA) - State-of-the-art transformer architecture
  - **1D-CNN** - Convolutional neural network for time series
  - **LSTM** - Long short-term memory network
- **Data Upload**: Support for `.csv`, `.pkl`, and `.npz` formats
- **Live Training**: Real-time training & validation loss visualization
- **Metrics Dashboard**: AUC, F1-Score, and inference latency tracking

### 🧪 **Test Inference & Results**
- **Model Selection**: Test with any trained model
- **Hyperparameters**: 
  - Adjustable learning rate slider
  - Configurable look-back window (64, 100, 128, 256 steps)
  - Threshold percentile selection (95th, 97th, 99th)
- **Visualization**: Interactive anomaly detection chart showing:
  - Time series signal
  - Detected anomaly regions (highlighted in red)
- **Metrics**: Precision, Recall, F1-Score, ROC AUC, TP/FP counts

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Navigate to the webapp directory:**
   ```bash
   cd webapp
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Copy your trained models to the models directory:**
   ```bash
   cp ../1dcnn/cnn_ae_best.keras models/
   cp ../lstm/lstm_ae_best.keras models/
   cp ../iTransformer/itransformer_ae_best.keras models/
   ```

5. **Copy test data to uploads directory:**
   ```bash
   cp ../data/custom_window_test.npz uploads/test_custom_window_test.npz
   ```

## Running the Application

### Start the Flask server:
```bash
python app.py
```

### Access the application:
Open your browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

### Training a Model

1. Navigate to the **Training Laboratory** page
2. Select your dataset type (Earthquake or Financial)
3. Choose a model architecture (iTransformer, 1D-CNN, or LSTM)
4. Upload your training data (.pkl, .csv, or .npz)
5. Click **"Start Training"**
6. Monitor the training progress with live loss curves
7. View final metrics: AUC, F1-Score, and latency

### Testing a Model

1. Navigate to the **Test Inference** page
2. Select the trained model you want to test
3. Adjust hyperparameters:
   - Learning rate (0.0001 - 0.01)
   - Look-back window (64, 100, 128, 256 steps)
   - Threshold percentile (95%, 97%, 99%)
4. Upload your test data (.csv or .npz) or use the default dataset
5. Click **"Run Prediction"**
6. View the anomaly detection visualization
7. Analyze metrics: TP, FP, Precision, Recall, F1-Score, ROC AUC

## File Structure

```
webapp/
├── app.py                          # Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── static/
│   ├── css/
│   │   └── style.css              # Main stylesheet
│   └── js/
│       ├── home.js                # Homepage animations
│       ├── training.js            # Training page logic
│       └── testing.js             # Testing page logic
├── templates/
│   ├── index.html                 # Homepage
│   ├── training.html              # Training Laboratory
│   └── testing.html               # Test Inference
├── uploads/                       # User uploaded files
└── models/                        # Trained models
    ├── cnn_ae_best.keras
    ├── lstm_ae_best.keras
    └── itransformer_ae_best.keras
```

## API Endpoints

### Training
- `POST /api/upload_training_data` - Upload training dataset
- `POST /api/train_model` - Start model training
- `GET /api/training_status` - Get current training status

### Testing
- `POST /api/upload_test_data` - Upload test dataset
- `POST /api/run_prediction` - Run anomaly detection

## Data Format

### Training Data (.pkl)
```python
DataFrame with columns:
- Z_channel: array of seismic/financial time series
```

### Test Data (.npz)
```python
{
    'X': ndarray of shape (N, window_size, 1),
    'y': ndarray of shape (N,) - optional labels (0=normal, 1=anomaly)
}
```

### CSV Format
```csv
timestamp,value1,value2,value3,...
2024-01-01,123.45,234.56,345.67,...
...
```

## Model Architectures

### 1D-CNN Autoencoder
- 3 encoder layers with MaxPooling
- 3 decoder layers with UpSampling
- Best for: Pattern recognition in time series

### LSTM Autoencoder
- Bidirectional LSTM encoder
- RepeatVector + LSTM decoder
- Best for: Sequential dependencies

### iTransformer
- Attention-based architecture
- Multi-head self-attention
- Best for: Long-range dependencies

## Configuration

Edit `app.py` to customize:
- `MAX_CONTENT_LENGTH`: Maximum upload size (default: 500MB)
- `LOOKBACK`: Default window size (default: 128)
- `STRIDE`: Window stride (default: 64)
- `EPOCHS`: Training epochs (default: 50)
- `BATCH_SIZE`: Training batch size (default: 256)

## Troubleshooting

### Port already in use
```bash
# Find and kill the process
lsof -ti:5000 | xargs kill -9

# Or use a different port
python app.py --port 5001
```

### Module not found errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Model not found
Ensure trained models are in the `models/` directory or update paths in `app.py`.

## Contributing

This is a graduation project for anomaly detection in time series data. Feel free to extend with:
- Additional model architectures
- More dataset types
- Advanced hyperparameter tuning
- Real-time monitoring features

## License

MIT License - See LICENSE file for details

## Author

Selin Bardakçı - Graduation Project 2026

---

**CHRONOS.AI** - Detect rare events before they become critical 🔍
