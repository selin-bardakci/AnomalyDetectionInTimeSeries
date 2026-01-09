#!/usr/bin/env python3
"""
CHRONOS.AI - Advanced Anomaly Detection Platform
Flask web application for training and testing time series anomaly detection models
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import threading
import time
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, LSTM, GRU, RepeatVector, TimeDistributed, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import yfinance as yf
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max for large datasets
app.config['SECRET_KEY'] = 'chronos-ai-anomaly-detection'

ALLOWED_EXTENSIONS = {'csv', 'pkl', 'npz'}

# Paths to existing models and data (relative to webapp directory)
EXISTING_MODELS = {
    'cnn': '../1dcnn/cnn_ae_best.keras',
    'lstm': '../lstm/lstm_ae_best.keras',
    'itransformer': '../iTransformer/itransformer_ae_best.keras'
}

EXISTING_TEST_DATA = '../data/custom_window_test.npz'

# Training data sources (hardcoded - no upload needed)
TRAINING_DATA_SOURCES = {
    'earthquake': 'trainData/z_channel_train.pkl',
    'financial': 'trainData/financial_train.pkl'  # Add financial data here if available
}

# Global storage for training state
training_state = {
    'is_training': False,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'history': {'loss': [], 'val_loss': []},
    'model_type': None,
    'dataset_type': None,
    'status_message': 'Ready'
}

# Custom callback for real-time updates
class RealtimeCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global training_state
        logs = logs or {}
        
        # Use explicit global reference and make a copy to ensure thread safety
        import copy
        
        training_state['current_epoch'] = epoch + 1
        training_state['progress'] = int((epoch + 1) / training_state['total_epochs'] * 100)
        
        # Append to history
        loss_val = float(logs.get('loss', 0))
        val_loss_val = float(logs.get('val_loss', 0))
        
        # Directly append to the global training_state
        training_state['history']['loss'].append(loss_val)
        training_state['history']['val_loss'].append(val_loss_val)
        
        training_state['status_message'] = f"Epoch {epoch + 1}/{training_state['total_epochs']} - loss: {loss_val:.6f} - val_loss: {val_loss_val:.6f}"
        
        print(f"✓ Epoch {epoch + 1} complete: loss={loss_val:.6f}, val_loss={val_loss_val:.6f}")
        print(f"  📊 History updated - Length: {len(training_state['history']['loss'])} items")
        print(f"  📊 Current history: loss={training_state['history']['loss']}, val_loss={training_state['history']['val_loss']}")
        print(f"  🔍 Training state ID: {id(training_state)}, History ID: {id(training_state['history'])}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@app.route('/training')
def training():
    """Training Laboratory page"""
    return render_template('training.html')


@app.route('/testing')
def testing():
    """Test Inference page"""
    return render_template('testing.html')


@app.route('/api/check_existing_resources')
def check_existing_resources():
    """Check for existing models and test data"""
    resources = {
        'models': {},
        'test_data_available': False
    }
    
    # Check for existing models
    for model_type, path in EXISTING_MODELS.items():
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            resources['models'][model_type] = True
        else:
            resources['models'][model_type] = False
    
    # Check for existing test data
    test_data_path = os.path.join(os.path.dirname(__file__), EXISTING_TEST_DATA)
    resources['test_data_available'] = os.path.exists(test_data_path)
    
    return jsonify(resources)


# ============================================================
# API ENDPOINTS - TRAINING
# ============================================================

@app.route('/api/upload_training_data', methods=['POST'])
def upload_training_data():
    """Upload training dataset"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    dataset_type = request.form.get('dataset_type', 'earthquake')
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and validate data
        try:
            if filename.endswith('.pkl'):
                df = pd.read_pickle(filepath)
                num_traces = len(df)
            elif filename.endswith('.csv'):
                df = pd.read_csv(filepath)
                num_traces = len(df)
            elif filename.endswith('.npz'):
                data = np.load(filepath)
                num_traces = len(data['X']) if 'X' in data else 0
            
            return jsonify({
                'success': True,
                'filename': filename,
                'num_traces': num_traces,
                'dataset_type': dataset_type
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'Invalid file type'}), 400


@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Start model training in background thread"""
    global training_state
    
    if training_state['is_training']:
        return jsonify({'success': False, 'error': 'Training already in progress'}), 400
    
    data = request.json
    model_type = data.get('model_type', 'cnn')
    dataset_type = data.get('dataset_type', 'earthquake')
    financial_dataset = data.get('financial_dataset', 'tsla')  # 'tsla' or 'yfinance'
    
    # For financial data, we don't use the hardcoded path - we load CSV directly
    if dataset_type == 'financial':
        training_data_path = None  # Will be handled in prepare_financial_data_from_csv
    else:
        # Get training data path from hardcoded sources for earthquake data
        if dataset_type not in TRAINING_DATA_SOURCES:
            return jsonify({'success': False, 'error': f'Dataset type not found: {dataset_type}'}), 400
        
        training_data_path = TRAINING_DATA_SOURCES[dataset_type]
        
        # Check if training data exists
        if not os.path.exists(training_data_path):
            return jsonify({
                'success': False, 
                'error': f'Training data not found: {training_data_path}. Please add the file to trainData/ folder.'
            }), 404
    
    # Hyperparameters
    lookback = int(data.get('lookback', 128))
    learning_rate = float(data.get('learning_rate', 0.001))
    epochs = int(data.get('epochs', 50))
    batch_size = int(data.get('batch_size', 256))
    
    # Initialize training state
    training_state['is_training'] = True
    training_state['progress'] = 0
    training_state['current_epoch'] = 0
    training_state['total_epochs'] = epochs
    training_state['model_type'] = model_type
    training_state['dataset_type'] = dataset_type
    training_state['history'] = {'loss': [], 'val_loss': []}
    training_state['status_message'] = 'Starting training...'
    
    # Start training in background thread
    thread = threading.Thread(
        target=train_model_background,
        args=(training_data_path, model_type, dataset_type, lookback, learning_rate, epochs, batch_size, financial_dataset)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Training started',
        'model_type': model_type,
        'dataset_type': dataset_type,
        'training_data': training_data_path
    })


def train_model_background(training_data_path, model_type, dataset_type, lookback, learning_rate, epochs, batch_size, financial_dataset='tsla'):
    """Background training function"""
    global training_state
    
    try:
        print(f"\n{'='*60}")
        print(f"🚀 TRAINING STARTED")
        print(f"{'='*60}")
        print(f"Model: {model_type}")
        print(f"Dataset: {dataset_type}")
        print(f"Data path: {training_data_path}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"{'='*60}\n")
        
        training_state['status_message'] = 'Loading data...'
        print("📂 Loading training data...")
        
        # Handle financial vs earthquake data differently
        if dataset_type == 'financial':
            # Financial data: load from selected CSV file
            train_scaled, test_scaled, window = prepare_financial_data_from_csv(financial_dataset, lookback=32)
            
            # Create windows for forecasting
            X_train, y_train = make_windows_multifeat(train_scaled, window, target_col=0)
            X_test, y_test = make_windows_multifeat(test_scaled, window, target_col=0)
            
            # Split training data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            print(f"   ✓ Train: {X_train.shape}, Val: {X_val.shape}")
            print(f"   ✓ Features per window: {X_train.shape[-1]}")
            
            n_features = X_train.shape[-1]
            task_type = 'forecasting'
            
        else:
            # Earthquake data: load from file
            if training_data_path.endswith('.pkl'):
                df = pd.read_pickle(training_data_path)
                print(f"   ✓ Loaded {len(df)} traces from pickle file")
                X = prepare_windows_from_df(df, lookback)
                print(f"   ✓ Created {X.shape[0]} windows from traces")
            elif training_data_path.endswith('.npz'):
                data = np.load(training_data_path)
                X = data['X']
                print(f"   ✓ Loaded {X.shape[0]} windows from npz file")
            else:
                training_state['is_training'] = False
                training_state['status_message'] = 'Error: Unsupported file format'
                print("   ✗ Error: Unsupported file format")
                return
            
            training_state['status_message'] = 'Splitting data...'
            print("\n📊 Splitting data...")
            
            # Split data
            X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
            y_train, y_val = X_train, X_val  # Autoencoder
            print(f"   ✓ Train: {X_train.shape}, Val: {X_val.shape}")
            
            n_features = 1
            task_type = 'autoencoder'
        
        training_state['status_message'] = 'Building model...'
        print(f"\n🏗️  Building {model_type.upper()} model...")
        
        # Build model based on dataset type
        model_lookback = 32 if dataset_type == 'financial' else lookback
        
        if model_type == 'cnn':
            model = build_cnn_model(model_lookback, learning_rate, n_features, task_type)
        elif model_type == 'lstm':
            model = build_lstm_model(model_lookback, learning_rate, n_features, task_type)
        elif model_type == 'gru':
            model = build_gru_model(model_lookback, learning_rate, n_features, task_type)
        elif model_type == 'itransformer':
            if dataset_type == 'financial':
                training_state['is_training'] = False
                training_state['status_message'] = 'Error: iTransformer not available for financial data'
                print(f"   ✗ Error: iTransformer not supported for financial dataset")
                return
            model = build_itransformer_model(lookback, learning_rate)
        else:
            training_state['is_training'] = False
            training_state['status_message'] = 'Error: Invalid model type'
            print(f"   ✗ Error: Invalid model type '{model_type}'")
            return
        
        print(f"   ✓ Model built successfully")
        
        training_state['status_message'] = 'Training model...'
        print(f"\n🎯 Starting training for {epochs} epochs...")
        print(f"   (Check browser for real-time loss chart updates)\n")
        
        # Callbacks
        realtime_callback = RealtimeCallback()
        
        # Different filename for financial models
        model_suffix = 'financial' if dataset_type == 'financial' else 'ae'
        model_filename = f'models/{model_type}_{model_suffix}_best.keras'
        
        checkpoint = ModelCheckpoint(
            filepath=model_filename,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train (different target for forecasting vs autoencoder)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[realtime_callback, checkpoint, early_stop],
            verbose=1
        )
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
        print(f"Model saved to: {model_filename}")
        print(f"{'='*60}\n")
        
        training_state['is_training'] = False
        training_state['progress'] = 100
        training_state['status_message'] = 'Training complete!'
        
    except Exception as e:
        training_state['is_training'] = False
        training_state['status_message'] = f'Error: {str(e)}'
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()


@app.route('/api/training_status')
def training_status():
    """Get current training status"""
    global training_state
    # Log the status for debugging
    if training_state['is_training'] and training_state['history']['loss']:
        print(f"📊 Status update - Epoch: {training_state['current_epoch']}/{training_state['total_epochs']}, Loss history length: {len(training_state['history']['loss'])}")
    
    print(f"  🔍 API reading training state ID: {id(training_state)}, History ID: {id(training_state['history'])}")
    print(f"  🔍 Current epoch: {training_state['current_epoch']}, History: {training_state['history']}")
    
    return jsonify(training_state)


# ============================================================
# API ENDPOINTS - TESTING
# ============================================================

@app.route('/api/upload_test_data', methods=['POST'])
def upload_test_data():
    """Upload test dataset"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'test_' + filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'test_filepath': 'test_' + filename
        })
    
    return jsonify({'success': False, 'error': 'Invalid file type'}), 400


@app.route('/api/run_prediction', methods=['POST'])
def run_prediction():
    """Run prediction on test data"""
    data = request.json
    model_type = data.get('model_type', 'cnn')
    test_filename = data.get('test_filename')
    threshold_percentile = int(data.get('threshold_percentile', 97))
    use_existing_data = data.get('use_existing_data', False)
    
    try:
        # Load model - check trained models first, then existing project models
        model_path = f'models/{model_type}_ae_best.keras'
        if not os.path.exists(model_path):
            # Use existing models from project
            if model_type in EXISTING_MODELS:
                model_path = os.path.join(os.path.dirname(__file__), EXISTING_MODELS[model_type])
        
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': f'Model not found: {model_type}'}), 404
        
        model = load_model(model_path)
        
        # Load test data
        if use_existing_data:
            # Use existing test data from project
            test_filepath = os.path.join(os.path.dirname(__file__), EXISTING_TEST_DATA)
            if not os.path.exists(test_filepath):
                return jsonify({'success': False, 'error': 'Existing test data not found'}), 404
        else:
            # Use uploaded test data
            if not test_filename:
                return jsonify({'success': False, 'error': 'No test file specified'}), 400
            test_filepath = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
        
        if test_filepath.endswith('.npz'):
            data = np.load(test_filepath)
            X_test = data['X']
            y_test = data.get('y', None)
        elif test_filepath.endswith('.csv'):
            df = pd.read_csv(test_filepath)
            X_test = prepare_windows_from_csv(df)
            y_test = None
        else:
            return jsonify({'success': False, 'error': 'Unsupported file format'}), 400
        
        # Run prediction
        X_hat = model.predict(X_test, batch_size=256, verbose=0)
        
        # Calculate reconstruction errors
        errors = np.mean((X_test - X_hat) ** 2, axis=(1, 2))
        
        # Determine threshold
        threshold = np.percentile(errors, threshold_percentile)
        
        # Predictions
        y_pred = (errors > threshold).astype(int)
        
        # Calculate metrics if ground truth available
        metrics = {}
        if y_test is not None:
            fpr, tpr, _ = roc_curve(y_test, errors)
            roc_auc = auc(fpr, tpr)
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            
            metrics = {
                'auc': float(roc_auc),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
        
        # Prepare visualization data
        anomaly_indices = np.where(y_pred == 1)[0].tolist()
        
        return jsonify({
            'success': True,
            'errors': errors[:1000].tolist(),  # First 1000 for visualization
            'threshold': float(threshold),
            'anomaly_indices': anomaly_indices[:1000],
            'total_anomalies': len(anomaly_indices),
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def prepare_financial_data(lookback=32):
    """Prepare financial data with feature engineering (ETH-USD) - Legacy function"""
    print("📊 Downloading ETH-USD data from Yahoo Finance...")
    df = yf.download(
        "ETH-USD",
        start="2017-01-01",
        end="2021-12-31",
        interval="1d",
        auto_adjust=False,
        progress=False
    ).reset_index()
    
    df = df[["Date", "Close"]].dropna().reset_index(drop=True)
    print(f"   ✓ Downloaded {len(df)} days of data")
    
    # Feature engineering
    print("🔧 Engineering features...")
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["abs_return"] = df["log_return"].abs()
    df["rolling_std_7"] = df["log_return"].rolling(7).std()
    df["rolling_std_30"] = df["log_return"].rolling(30).std()
    
    eps = 1e-8
    df["norm_return"] = df["log_return"] / (df["rolling_std_30"] + eps)
    df = df.dropna().reset_index(drop=True)
    print(f"   ✓ Features created: norm_return, abs_return, rolling_std_7, rolling_std_30")
    
    # Split data
    split_ratio = 0.7
    split_idx = int(len(df) * split_ratio)
    
    FEATURES = ["norm_return", "abs_return", "rolling_std_7", "rolling_std_30"]
    train_feat = df[FEATURES].values[:split_idx]
    test_feat = df[FEATURES].values[split_idx:]
    
    # Scale features
    print("📏 Scaling features...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat)
    test_scaled = scaler.transform(test_feat)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/financial_scaler.joblib')
    print("   ✓ Scaler saved to models/financial_scaler.joblib")
    
    return train_scaled, test_scaled, lookback


def prepare_financial_data_from_csv(dataset_name, lookback=32):
    """Prepare financial data from CSV files"""
    # Map dataset names to file paths
    csv_files = {
        'tsla': 'trainData/TSLA_2019-2021_pandemic.csv',
        'yfinance': 'trainData/yfinance_clean.csv'
    }
    
    csv_path = csv_files.get(dataset_name, csv_files['tsla'])
    print(f"📊 Loading financial data from {csv_path}...")
    
    # Handle different CSV formats
    if dataset_name == 'tsla':
        # TSLA format: Read with header=0, skip ticker row
        df = pd.read_csv(csv_path, header=0)
        print(f"   ✓ Loaded {len(df)} records")
        print(f"   📋 Columns: {df.columns.tolist()}")
        
        # Skip the ticker row (first row after header)
        df = df.iloc[1:].reset_index(drop=True)
        
        # The first column is labeled as 'Price' but contains dates
        # Rename it to 'Date' and use the 'Close' column
        df = df.rename(columns={'Price': 'Date'})
        
        # Convert to proper types
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df[['Date', 'Close']].dropna().reset_index(drop=True)
    else:
        # yfinance format: already has Datetime and Close
        df = pd.read_csv(csv_path)
        print(f"   ✓ Loaded {len(df)} records")
        
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.rename(columns={'Datetime': 'Date'})
        df = df[['Date', 'Close']].dropna().reset_index(drop=True)
    
    print(f"   ✓ Cleaned data: {len(df)} records")
    
    # Feature engineering
    print("🔧 Engineering features...")
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["abs_return"] = df["log_return"].abs()
    df["rolling_std_7"] = df["log_return"].rolling(7).std()
    df["rolling_std_30"] = df["log_return"].rolling(30).std()
    
    eps = 1e-8
    df["norm_return"] = df["log_return"] / (df["rolling_std_30"] + eps)
    df = df.dropna().reset_index(drop=True)
    print(f"   ✓ Features created: {len(df)} records after feature engineering")
    
    # Split data
    split_ratio = 0.7
    split_idx = int(len(df) * split_ratio)
    
    FEATURES = ["norm_return", "abs_return", "rolling_std_7", "rolling_std_30"]
    train_feat = df[FEATURES].values[:split_idx]
    test_feat = df[FEATURES].values[split_idx:]
    
    # Scale features
    print("📏 Scaling features...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat)
    test_scaled = scaler.transform(test_feat)
    
    # Save scaler with dataset name
    os.makedirs('models', exist_ok=True)
    scaler_path = f'models/financial_{dataset_name}_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"   ✓ Scaler saved to {scaler_path}")
    
    return train_scaled, test_scaled, lookback


def make_windows_multifeat(X, window, target_col=0):
    """Create windowed data for financial forecasting"""
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window, :])  # (window, features)
        ys.append(X[i+window, target_col])  # next norm_return
    return np.array(Xs), np.array(ys)


def prepare_windows_from_df(df, lookback=128, stride=64, max_windows=100):
    """Prepare sliding windows from DataFrame"""
    X_all = []
    
    for z in df["Z_channel"][:4000]:  # Limit to 4000 traces
        z = np.asarray(z, dtype=np.float32)
        z = (z - z.mean()) / (z.std() + 1e-6)
        
        windows = []
        for i in range(0, len(z) - lookback, stride):
            windows.append(z[i:i + lookback])
        
        windows = np.array(windows, dtype=np.float32)
        
        if len(windows) > max_windows:
            idx = np.random.choice(len(windows), max_windows, replace=False)
            windows = windows[idx]
        
        X_all.append(windows)
    
    X = np.vstack(X_all)[..., np.newaxis]
    return X


def prepare_windows_from_csv(df, lookback=128, stride=64):
    """Prepare sliding windows from CSV"""
    # Assuming CSV has time series in columns
    X_all = []
    
    for col in df.columns:
        if col == 'time' or col == 'timestamp':
            continue
        
        signal = df[col].values.astype(np.float32)
        signal = (signal - signal.mean()) / (signal.std() + 1e-6)
        
        windows = []
        for i in range(0, len(signal) - lookback, stride):
            windows.append(signal[i:i + lookback])
        
        if len(windows) > 0:
            X_all.append(np.array(windows))
    
    if len(X_all) > 0:
        X = np.vstack(X_all)[..., np.newaxis]
        return X
    
    return np.array([])


def build_cnn_model(lookback=128, learning_rate=0.001, n_features=1, task='autoencoder'):
    """Build 1D-CNN Model (Autoencoder or Forecasting)"""
    if task == 'forecasting':
        # Financial forecasting model
        inp = Input(shape=(lookback, n_features))
        x = Conv1D(64, 3, activation="relu", padding="same")(inp)
        x = Conv1D(64, 3, activation="relu", padding="same")(x)
        x = GlobalAveragePooling1D()(x)
        out = Dense(1)(x)
        
        model = Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
        return model
    
    # Earthquake autoencoder model
    inp = Input(shape=(lookback, 1))
    
    # Encoder
    x = Conv1D(32, 7, activation="relu", padding="same")(inp)
    x = MaxPooling1D(2, padding="same")(x)
    x = Conv1D(64, 5, activation="relu", padding="same")(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = Conv1D(128, 3, activation="relu", padding="same")(x)
    
    # Decoder
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 5, activation="relu", padding="same")(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 7, activation="relu", padding="same")(x)
    out = Conv1D(1, 7, padding="same")(x)
    
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    
    return model


def build_lstm_model(lookback=128, learning_rate=0.001, n_features=1, task='autoencoder'):
    """Build LSTM Model (Autoencoder or Forecasting)"""
    if task == 'forecasting':
        # Financial forecasting model
        inp = Input(shape=(lookback, n_features))
        x = LSTM(64)(inp)
        out = Dense(1)(x)
        
        model = Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
        return model
    
    # Earthquake autoencoder model
    inp = Input(shape=(lookback, 1))
    
    # Encoder
    x = LSTM(64, activation='relu', return_sequences=True)(inp)
    x = LSTM(32, activation='relu', return_sequences=False)(x)
    
    # Decoder
    x = RepeatVector(lookback)(x)
    x = LSTM(32, activation='relu', return_sequences=True)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    out = TimeDistributed(Dense(1))(x)
    
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    
    return model


def build_gru_model(lookback=128, learning_rate=0.001, n_features=1, task='autoencoder'):
    """Build GRU Model (Autoencoder or Forecasting)"""
    if task == 'forecasting':
        # Financial forecasting model
        inp = Input(shape=(lookback, n_features))
        x = GRU(64)(inp)
        out = Dense(1)(x)
        
        model = Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
        return model
    
    # Earthquake autoencoder model
    inp = Input(shape=(lookback, 1))
    
    # Encoder
    x = GRU(64, activation='relu', return_sequences=True)(inp)
    x = GRU(32, activation='relu', return_sequences=False)(x)
    
    # Decoder
    x = RepeatVector(lookback)(x)
    x = GRU(32, activation='relu', return_sequences=True)(x)
    x = GRU(64, activation='relu', return_sequences=True)(x)
    out = TimeDistributed(Dense(1))(x)
    
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    
    return model


def build_itransformer_model(lookback=128, learning_rate=0.001):
    """Build iTransformer Autoencoder"""
    d_model = 64
    n_heads = 4
    n_layers = 2
    
    def transformer_block(x):
        attn = MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=0.1
        )(x, x)
        x = LayerNormalization()(x + attn)
        
        ff = Dense(d_model * 4, activation="relu")(x)
        ff = Dense(d_model)(ff)
        x = LayerNormalization()(x + ff)
        
        return x
    
    inp = Input(shape=(lookback, 1))
    
    # Inverted embedding (time as tokens)
    x = Dense(d_model)(inp)
    
    # Encoder
    for _ in range(n_layers):
        x = transformer_block(x)
    
    # Bottleneck
    latent = Dense(d_model, activation="relu")(x)
    
    # Decoder
    x = latent
    for _ in range(n_layers):
        x = transformer_block(x)
    
    out = Dense(1)(x)
    
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    
    return model


# ============================================================
# TEST INFERENCE ENDPOINTS
# ============================================================

@app.route('/api/check_models', methods=['GET'])
def check_models():
    """Check which models are available (user-trained and pretrained)"""
    try:
        model_type = request.args.get('model_type', 'cnn')
        
        # Check user-trained model
        user_model_path = f'models/{model_type}_ae_best.keras'
        user_trained_exists = os.path.exists(user_model_path)
        
        # Check pretrained model
        pretrained_path = EXISTING_MODELS.get(model_type, '')
        pretrained_exists = os.path.exists(pretrained_path) if pretrained_path else False
        
        return jsonify({
            'success': True,
            'user_trained': {
                'exists': user_trained_exists,
                'path': user_model_path if user_trained_exists else None
            },
            'pretrained': {
                'exists': pretrained_exists,
                'path': pretrained_path if pretrained_exists else None
            },
            'test_data_exists': os.path.exists('testData/custom_window_test.npz')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/run_inference', methods=['POST'])
def run_inference():
    """Run inference on test data and return metrics with advanced threshold options"""
    try:
        data = request.json
        model_type = data.get('model_type', 'cnn')
        model_source = data.get('model_source', 'user')  # 'user' or 'pretrained'
        threshold_method = data.get('threshold_method', 'percentile')  # 'percentile', 'youden', 'f1', 'mcc', 'custom'
        percentile = data.get('percentile', 95)  # For percentile method
        custom_value = data.get('custom_value', 0.0001)  # For custom method
        
        print(f"\n{'='*60}")
        print(f"🧪 STARTING TEST INFERENCE")
        print(f"{'='*60}")
        print(f"Model type: {model_type}")
        print(f"Model source: {model_source}")
        print(f"Threshold method: {threshold_method}")
        if threshold_method == 'percentile':
            print(f"Percentile: p{percentile}")
        elif threshold_method == 'custom':
            print(f"Custom threshold: {custom_value}")
        
        # Determine model path
        if model_source == 'user':
            model_path = f'models/{model_type}_ae_best.keras'
        else:
            model_path = EXISTING_MODELS.get(model_type, '')
        
        if not os.path.exists(model_path):
            return jsonify({
                'success': False,
                'error': f'Model not found at {model_path}'
            }), 404
        
        # Load model
        print(f"📂 Loading model from: {model_path}")
        model = load_model(model_path)
        print("✅ Model loaded")
        
        # Load test data
        test_data_path = 'testData/custom_window_test.npz'
        if not os.path.exists(test_data_path):
            return jsonify({
                'success': False,
                'error': f'Test data not found at {test_data_path}'
            }), 404
        
        print(f"📂 Loading test data from: {test_data_path}")
        test_data = np.load(test_data_path)
        X = test_data['X']  # (N, 128, 1)
        y = test_data['y']  # (N,) - 0=noise, 1=earthquake
        print(f"✅ Test data loaded: X={X.shape}, y={y.shape}")
        print(f"   Noise windows: {(y==0).sum()}, Earthquake windows: {(y==1).sum()}")
        
        # Run predictions
        print("🔮 Running predictions...")
        X_hat = model.predict(X, batch_size=256, verbose=0)
        
        # Calculate reconstruction errors
        errors = np.mean((X - X_hat) ** 2, axis=(1, 2))
        print(f"✅ Predictions complete")
        print(f"   Error stats: mean={errors.mean():.6f}, median={np.median(errors):.6f}, std={errors.std():.6f}")
        
        # Calculate ROC curve and AUC
        from sklearn.metrics import matthews_corrcoef
        fpr, tpr, thresholds = roc_curve(y, errors)
        roc_auc = auc(fpr, tpr)
        print(f"📊 ROC AUC: {roc_auc:.4f}")
        
        # Calculate optimal thresholds
        noise_errors = errors[y == 0]
        
        # Youden's J
        j_scores = tpr - fpr
        optimal_idx_youden = np.argmax(j_scores)
        youden_threshold = float(thresholds[optimal_idx_youden])
        
        # F1-Score maximization
        precisions = tpr / (tpr + fpr + 1e-10)
        f1_scores = 2 * (precisions * tpr) / (precisions + tpr + 1e-10)
        optimal_idx_f1 = np.argmax(f1_scores)
        f1_threshold = float(thresholds[optimal_idx_f1])
        
        # MCC maximization
        best_mcc = -1
        mcc_threshold = float(thresholds[0])
        for th in thresholds[::max(1, len(thresholds)//100)]:  # Sample for performance
            y_pred_temp = (errors > th).astype(int)
            mcc = matthews_corrcoef(y, y_pred_temp)
            if mcc > best_mcc:
                best_mcc = mcc
                mcc_threshold = float(th)
        
        # Select threshold based on method
        if threshold_method == 'youden':
            final_threshold = youden_threshold
            threshold_name = "Youden's J (ROC-Optimal)"
        elif threshold_method == 'f1':
            final_threshold = f1_threshold
            threshold_name = "F1-Score Maximization"
        elif threshold_method == 'mcc':
            final_threshold = mcc_threshold
            threshold_name = "MCC Maximization"
        elif threshold_method == 'custom':
            final_threshold = float(custom_value)
            threshold_name = "Custom Threshold"
        else:  # percentile (default)
            final_threshold = float(np.percentile(noise_errors, int(percentile)))
            threshold_name = f"Percentile p{int(percentile)}"
        
        print(f"📊 Selected threshold method: {threshold_name}")
        print(f"📊 Threshold value: {final_threshold:.8e}")
        
        # Apply threshold
        y_pred = (errors > final_threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp + 1e-9)
        
        print(f"📊 Metrics with {threshold_name}:")
        print(f"   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"   Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}")
        
        # Generate ROC curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_type.upper()} AE (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve — Earthquake vs Noise Detection', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        roc_plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Generate confusion matrix heatmap
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix ({threshold_name})', fontsize=14, fontweight='bold')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Noise', 'Earthquake'])
        plt.yticks(tick_marks, ['Noise', 'Earthquake'])
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=16, fontweight='bold')
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        cm_plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        print(f"✅ Inference complete!")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'metrics': {
                'auc': float(roc_auc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'specificity': float(specificity),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'threshold': float(final_threshold),
                'threshold_method': threshold_method,
                'threshold_name': threshold_name,
                'available_thresholds': {
                    'youden': float(youden_threshold),
                    'f1': float(f1_threshold),
                    'mcc': float(mcc_threshold),
                    'p90': float(np.percentile(noise_errors, 90)),
                    'p95': float(np.percentile(noise_errors, 95)),
                    'p97': float(np.percentile(noise_errors, 97)),
                    'p99': float(np.percentile(noise_errors, 99))
                }
            },
            'plots': {
                'roc_curve': f'data:image/png;base64,{roc_plot_base64}',
                'confusion_matrix': f'data:image/png;base64,{cm_plot_base64}'
            },
            'stats': {
                'total_samples': int(len(y)),
                'noise_samples': int((y == 0).sum()),
                'earthquake_samples': int((y == 1).sum()),
                'error_mean': float(errors.mean()),
                'error_median': float(np.median(errors)),
                'error_std': float(errors.std())
            }
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error during inference: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': error_trace
        }), 500


# ============================================================
# RUN APP
# ============================================================

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('testData', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5004)
