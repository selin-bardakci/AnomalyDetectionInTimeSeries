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
    financial_dataset = data.get('financial_dataset', 'tsla')  # For backward compatibility
    stock_ticker = data.get('stock_ticker', 'TSLA')  # Stock symbol
    start_date = data.get('start_date', '2019-01-01')  # Training start date
    end_date = data.get('end_date', '2021-12-31')  # Training end date
    
    # For financial data, we don't use the hardcoded path - we fetch dynamically
    if dataset_type == 'financial':
        training_data_path = None  # Will be fetched via yfinance
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
    # Default epochs: CNN=80, others=50 (matching pipeline)
    default_epochs = 80 if model_type == 'cnn' else 50
    epochs = int(data.get('epochs', default_epochs))
    batch_size = int(data.get('batch_size', 32))  # Pipeline default: 32
    
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
        args=(training_data_path, model_type, dataset_type, lookback, learning_rate, epochs, batch_size, financial_dataset, stock_ticker, start_date, end_date)
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


def train_model_background(training_data_path, model_type, dataset_type, lookback, learning_rate, epochs, batch_size, financial_dataset='tsla', stock_ticker='TSLA', start_date='2019-01-01', end_date='2021-12-31'):
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
            # Financial data: fetch dynamically via yfinance
            train_scaled, test_scaled, window = prepare_financial_data_dynamic(stock_ticker, start_date, end_date, lookback=32)
            
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
            patience=5,  # Pipeline default: 5
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
        
        # For financial data, save anomaly detection artifacts
        if dataset_type == 'financial':
            print(f"\n📊 Computing anomaly detection artifacts...")
            
            # 1) Compute residuals
            yhat_tr = model.predict(X_train, verbose=0).ravel()
            yhat_te = model.predict(X_test, verbose=0).ravel()
            res_tr = np.abs(y_train - yhat_tr)
            res_te = np.abs(y_test - yhat_te)
            
            print(f"   ✓ Residuals computed - Train: {res_tr.mean():.6f}, Test: {res_te.mean():.6f}")
            
            # 2) Save test data metadata for inference
            test_metadata = {
                'window': 32,
                'n_test_samples': len(y_test),
                'split_ratio': 0.7,
                'features': FEATURES,
                'model_type': model_type,
                'dataset': financial_dataset
            }
            
            metadata_path = f'models/{model_type}_financial_{financial_dataset}_meta.json'
            with open(metadata_path, 'w') as f:
                json.dump(test_metadata, f, indent=2)
            print(f"   ✓ Metadata saved to {metadata_path}")
            
            # 3) Compute MAD threshold from training residuals
            # Need to normalize residuals by volatility first
            # Load original data to get rolling_std_30
            csv_files = {
                'tsla': 'trainData/TSLA_2019-2021_pandemic.csv',
                'yfinance': 'trainData/yfinance_clean.csv'
            }
            csv_path = csv_files.get(financial_dataset, csv_files['tsla'])
            
            if financial_dataset == 'tsla':
                df_orig = pd.read_csv(csv_path, header=0)
                df_orig = df_orig.iloc[1:].reset_index(drop=True)
                df_orig = df_orig.rename(columns={'Price': 'Date'})
                df_orig['Date'] = pd.to_datetime(df_orig['Date'], errors='coerce')
                df_orig['Close'] = pd.to_numeric(df_orig['Close'], errors='coerce')
                df_orig = df_orig[['Date', 'Close']].dropna().reset_index(drop=True)
            else:
                df_orig = pd.read_csv(csv_path)
                df_orig['Datetime'] = pd.to_datetime(df_orig['Datetime'])
                df_orig = df_orig.rename(columns={'Datetime': 'Date'})
                df_orig = df_orig[['Date', 'Close']].dropna().reset_index(drop=True)
            
            # Recompute features
            df_orig["log_return"] = np.log(df_orig["Close"] / df_orig["Close"].shift(1))
            df_orig["rolling_std_30"] = df_orig["log_return"].rolling(30).std()
            df_orig = df_orig.dropna().reset_index(drop=True)
            
            split_idx_orig = int(len(df_orig) * 0.7)
            std30_train = df_orig["rolling_std_30"].values[:split_idx_orig]
            std30_train = std30_train[32:]  # Skip window
            
            # Normalize residuals
            norm_res_tr = res_tr / (std30_train + 1e-8)
            
            # Compute MAD threshold (k=2.5)
            def mad_threshold(residuals, k=2.5):
                med = np.median(residuals)
                mad = np.median(np.abs(residuals - med)) + 1e-12
                return med + k * 1.4826 * mad
            
            thr = mad_threshold(norm_res_tr, k=2.5)
            
            # Save threshold
            threshold_path = f'models/{model_type}_financial_{financial_dataset}_threshold.txt'
            with open(threshold_path, 'w') as f:
                f.write(str(thr))
            print(f"   ✓ MAD Threshold (k=2.5): {thr:.8f}")
            print(f"   ✓ Threshold saved to {threshold_path}")
            
            # Save recommended parameters
            anomaly_params = {
                'mad_k': 2.5,
                'stress_percentile': 90,
                'threshold_value': float(thr),
                'anomaly_method': 'residual + stress + MAD'
            }
            
            params_path = f'models/{model_type}_financial_{financial_dataset}_params.json'
            with open(params_path, 'w') as f:
                json.dump(anomaly_params, f, indent=2)
            print(f"   ✓ Anomaly params saved to {params_path}")
            print(f"\n{'='*60}\n")
        
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


def prepare_financial_data_dynamic(stock_ticker, start_date, end_date, lookback=32):
    """Fetch and prepare financial data dynamically via yfinance"""
    import yfinance as yf
    
    print(f"📊 Fetching data for {stock_ticker} from {start_date} to {end_date}...")
    
    # Fetch data from yfinance
    df = yf.download(stock_ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for {stock_ticker} in date range {start_date} to {end_date}")
    
    print(f"   ✓ Downloaded {len(df)} records")
    
    # Reset index to have Date as column
    df = df.reset_index()
    df = df.rename(columns={'Date': 'Date'})
    df = df[['Date', 'Close']].dropna().reset_index(drop=True)
    
    print(f"   ✓ Cleaned data: {len(df)} records")
    
    # Feature engineering (matching pipeline)
    print("🔧 Engineering features...")
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["abs_return"] = df["log_return"].abs()
    df["rolling_std_7"] = df["log_return"].rolling(7).std()
    df["rolling_std_30"] = df["log_return"].rolling(30).std()
    
    eps = 1e-8
    df["norm_return"] = df["log_return"] / (df["rolling_std_30"] + eps)
    df = df.dropna().reset_index(drop=True)
    print(f"   ✓ Features created: {len(df)} records after feature engineering")
    
    # Split data (70/30 train/test)
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
    
    # Save scaler with stock ticker name
    os.makedirs('models', exist_ok=True)
    scaler_path = f'models/financial_{stock_ticker.replace("-", "_").lower()}_scaler.joblib'
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

def run_financial_inference(model_type, stock_ticker, start_date, end_date, threshold_method, 
                            mad_k, stress_percentile, custom_value, percentile):
    """Run financial anomaly detection inference with dynamic date range"""
    try:
        # Load model - check user-trained first, then pretrained
        user_trained_path = f'models/{model_type}_financial_best.keras'
        pretrained_path = f'pretrained/{model_type}_financial_best.keras'
        
        if os.path.exists(user_trained_path):
            model_path = user_trained_path
            print(f"📂 Loading user-trained model from: {model_path}")
        elif os.path.exists(pretrained_path):
            model_path = pretrained_path
            print(f"📂 Loading pretrained model from: {model_path}")
        else:
            return jsonify({
                'success': False,
                'error': f'Model not found. Please train the model first or place pretrained model in pretrained/ folder.'
            }), 404
        
        model = load_model(model_path)
        print("✅ Model loaded")
        
        # Load scaler (try stock-specific first, then generic)
        scaler_path = f'models/financial_{stock_ticker.replace("-", "_").lower()}_scaler.joblib'
        if not os.path.exists(scaler_path):
            # Try generic scaler
            scaler_path = 'models/financial_tsla_scaler.joblib'
        
        if not os.path.exists(scaler_path):
            return jsonify({
                'success': False,
                'error': f'Scaler not found. Please train the model first.'
            }), 404
        
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded from {scaler_path}")
        
        # Fetch data dynamically for test period
        import yfinance as yf
        print(f"📊 Fetching test data for {stock_ticker} from {start_date} to {end_date}...")
        df = yf.download(stock_ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return jsonify({
                'success': False,
                'error': f'No data found for {stock_ticker} in date range'
            }), 404
        
        # Reset index to have Date as column
        df = df.reset_index()
        df = df.rename(columns={'Date': 'Date'})
        df = df[['Date', 'Close']].dropna().reset_index(drop=True)
        
        print(f"📊 Loaded {len(df)} records for {stock_ticker}")
        
        # Feature engineering
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["abs_return"] = df["log_return"].abs()
        df["rolling_std_7"] = df["log_return"].rolling(7).std()
        df["rolling_std_30"] = df["log_return"].rolling(30).std()
        
        eps = 1e-8
        df["norm_return"] = df["log_return"] / (df["rolling_std_30"] + eps)
        df = df.dropna().reset_index(drop=True)
        
        # Use ALL data for testing (no split, since user selected specific date range)
        FEATURES = ["norm_return", "abs_return", "rolling_std_7", "rolling_std_30"]
        test_feat = df[FEATURES].values
        test_dates = df["Date"].values
        test_prices = df["Close"].values
        
        # Scale features
        test_scaled = scaler.transform(test_feat)
        
        # Create windows
        WINDOW = 32
        X_test, y_test = make_windows_multifeat(test_scaled, WINDOW, target_col=0)
        
        print(f"📊 Test data: {X_test.shape}, Targets: {y_test.shape}")
        
        # Run predictions
        print("🔮 Running predictions...")
        yhat_te = model.predict(X_test, verbose=0).ravel()
        res_te = np.abs(y_test - yhat_te)
        
        print(f"✅ Predictions complete - MAE: {res_te.mean():.6f}")
        
        # Get test dates and prices (after windowing)
        dates_test = test_dates[WINDOW:]
        price_test = test_prices[WINDOW:]
        abs_ret_test = df["abs_return"].values[WINDOW:]
        std30_test = df["rolling_std_30"].values[WINDOW:]
        
        # Compute stress mask
        abs_ret_thr = np.percentile(abs_ret_test, stress_percentile)
        stress_mask = (abs_ret_test > abs_ret_thr).astype(int)
        
        print(f"📊 Stress threshold (|return| {stress_percentile}th pct): {abs_ret_thr:.6f}")
        print(f"📊 Stress day rate: {stress_mask.mean():.3f}")
        
        # Normalize residuals by volatility
        norm_res_te = res_te / (std30_test + eps)
        
        # Determine threshold
        if threshold_method == 'mad':
            # Load recommended MAD threshold
            threshold_path = f'models/{model_type}_financial_{financial_dataset}_threshold.txt'
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    recommended_threshold = float(f.read().strip())
                threshold = recommended_threshold
                threshold_name = f"MAD (k={mad_k}, Recommended)"
                print(f"📊 Using recommended MAD threshold: {threshold:.8f}")
            else:
                # Compute MAD from training data
                print("⚠️ Recommended threshold not found, computing from training data...")
                train_feat = df[FEATURES].values[:split_idx]
                train_scaled = scaler.transform(train_feat)
                X_train, y_train = make_windows_multifeat(train_scaled, WINDOW, target_col=0)
                yhat_tr = model.predict(X_train, verbose=0).ravel()
                res_tr = np.abs(y_train - yhat_tr)
                
                std30_train = df["rolling_std_30"].values[:split_idx]
                std30_train = std30_train[WINDOW:]
                norm_res_tr = res_tr / (std30_train + eps)
                
                def mad_threshold(residuals, k=2.5):
                    med = np.median(residuals)
                    mad = np.median(np.abs(residuals - med)) + 1e-12
                    return med + k * 1.4826 * mad
                
                threshold = mad_threshold(norm_res_tr, k=mad_k)
                threshold_name = f"MAD (k={mad_k}, Computed)"
                print(f"📊 Computed MAD threshold: {threshold:.8f}")
        elif threshold_method == 'percentile':
            threshold = float(np.percentile(norm_res_te, percentile))
            threshold_name = f"Percentile p{percentile}"
        elif threshold_method == 'custom':
            threshold = float(custom_value)
            threshold_name = "Custom Threshold"
        else:
            threshold = float(np.percentile(norm_res_te, 95))
            threshold_name = "Percentile p95 (Default)"
        
        print(f"📊 Selected threshold: {threshold_name} = {threshold:.8f}")
        
        # Apply threshold with stress mask
        raw_anomaly = (norm_res_te > threshold).astype(int)
        anomaly = ((norm_res_te > threshold) & (stress_mask == 1)).astype(int)
        
        print(f"📊 Raw anomaly rate: {raw_anomaly.mean():.3f}")
        print(f"📊 Final anomaly rate (with stress): {anomaly.mean():.3f}")
        print(f"📊 Anomaly count: {anomaly.sum()} / {len(anomaly)}")
        
        # Filter for 2021 data
        dates_test_dt = pd.to_datetime(dates_test)
        mask_2021 = (dates_test_dt >= "2021-01-01") & (dates_test_dt < "2022-01-01")
        
        dates_2021 = dates_test_dt[mask_2021]
        price_2021 = price_test[mask_2021]
        anomaly_2021 = anomaly[mask_2021]
        norm_return_2021 = y_test[mask_2021]
        
        print(f"📊 2021 data: {len(dates_2021)} points, {anomaly_2021.sum()} anomalies")
        
        # Generate price vs anomaly plot
        plt.figure(figsize=(16, 5))
        plt.plot(dates_2021, price_2021, label="Price", linewidth=2, color='#4A90E2')
        
        # Mark anomalies
        idx_2021 = np.where(anomaly_2021 == 1)[0]
        if len(idx_2021) > 0:
            plt.scatter(dates_2021[idx_2021], price_2021[idx_2021], 
                       color="red", s=50, label="Anomaly", zorder=5, marker='o', edgecolors='darkred', linewidths=1.5)
        
        plt.title(f"Price vs Anomaly Detection ({model_type.upper()}, 2021)", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price ($)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        price_plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Generate norm return plot
        plt.figure(figsize=(16, 4))
        plt.plot(dates_2021, norm_return_2021, label="Normalized Return", alpha=0.7, linewidth=1.5, color='#64748B')
        if len(idx_2021) > 0:
            plt.scatter(dates_2021[idx_2021], norm_return_2021[idx_2021], 
                       color="red", s=30, label="Anomaly", zorder=5)
        plt.title(f"Normalized Return Anomaly Detection ({model_type.upper()})", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Normalized Return", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        return_plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        print(f"✅ Inference complete!")
        print(f"{'='*60}\n")
        
        # Prepare chart data for frontend
        chart_data = {
            'dates': dates_2021.strftime('%Y-%m-%d').tolist(),
            'prices': price_2021.tolist(),
            'anomalies': anomaly_2021.tolist(),
            'returns': norm_return_2021.tolist()
        }
        
        return jsonify({
            'success': True,
            'dataset_type': 'financial',
            'metrics': {
                'total_samples': int(len(anomaly)),
                'anomaly_count': int(anomaly.sum()),
                'anomaly_rate': float(anomaly.mean()),
                'raw_anomaly_count': int(raw_anomaly.sum()),
                'stress_threshold': float(abs_ret_thr),
                'threshold': float(threshold),
                'threshold_method': threshold_method,
                'threshold_name': threshold_name,
                'threshold_method_params': {
                    'k': float(mad_k),
                    'stress_percentile': int(stress_percentile)
                },
                'mae': float(res_te.mean()),
                'samples_2021': int(len(dates_2021)),
                'anomalies_2021': int(anomaly_2021.sum())
            },
            'plots': {
                'price_anomaly': f'data:image/png;base64,{price_plot_base64}',
                'return_anomaly': f'data:image/png;base64,{return_plot_base64}'
            },
            'chart_data': chart_data,
            'stats': {
                'total_samples': int(len(y_test)),
                'mean_residual': float(res_te.mean()),
                'median_residual': float(np.median(res_te)),
                'std_residual': float(res_te.std())
            }
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error during financial inference: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': error_trace
        }), 500


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
        dataset_type = data.get('dataset_type', 'earthquake')  # 'earthquake' or 'financial'
        
        # Financial-specific parameters
        stock_ticker = data.get('stock_ticker', 'TSLA')
        start_date = data.get('start_date', '2019-01-01')
        end_date = data.get('end_date', '2021-12-31')
        
        threshold_method = data.get('threshold_method', 'percentile')  # 'percentile', 'youden', 'f1', 'mcc', 'custom', 'mad'
        percentile = data.get('percentile', 95)  # For percentile method
        custom_value = data.get('custom_value', 0.0001)  # For custom method
        mad_k = data.get('mad_k', 2.5)  # For MAD method
        stress_percentile = data.get('stress_percentile', 90)  # For stress mask
        
        print(f"\n{'='*60}")
        print(f"🧪 STARTING TEST INFERENCE")
        print(f"{'='*60}")
        print(f"Model type: {model_type}")
        print(f"Model source: {model_source}")
        print(f"Dataset type: {dataset_type}")
        if dataset_type == 'financial':
            print(f"Stock ticker: {stock_ticker}")
            print(f"Date range: {start_date} to {end_date}")
        print(f"Threshold method: {threshold_method}")
        if threshold_method == 'percentile':
            print(f"Percentile: p{percentile}")
        elif threshold_method == 'custom':
            print(f"Custom threshold: {custom_value}")
        elif threshold_method == 'mad':
            print(f"MAD k: {mad_k}, Stress percentile: {stress_percentile}")
        
        # Handle financial vs earthquake inference differently
        if dataset_type == 'financial':
            return run_financial_inference(
                model_type, stock_ticker, start_date, end_date, threshold_method,
                mad_k, stress_percentile, custom_value, percentile
            )
        
        # Continue with earthquake inference...
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
