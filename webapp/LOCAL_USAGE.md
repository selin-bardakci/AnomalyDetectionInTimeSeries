# CHRONOS.AI - Local Usage Guide

## Quick Start (No External Database Needed!)

Everything runs locally on your laptop. No cloud services or external databases required.

### 1. Start the Application

```bash
cd /Users/selinbardakci/BitirmeProjesi/webapp
python app.py
```

Open your browser to: **http://localhost:5000**

---

## Training Models

### Option 1: Upload Your Own Data from Laptop

1. Go to **Training Laboratory** page
2. Select dataset type: **Earthquake Data** or **Financial Market**
3. Choose model: **iTransformer**, **1D-CNN**, or **LSTM**
4. Click **"📁 Choose File"** and select a file from your laptop:
   - `.pkl` files (pandas DataFrame with Z_channel column)
   - `.npz` files (numpy arrays with X data)
   - `.csv` files (time series data)
5. Click **"Start Training"**
6. Watch real-time training progress with live loss charts!

### Supported Training Data Formats:

**Pickle (.pkl)**
```python
# DataFrame with structure:
df['Z_channel'] = [array of time series data]
```

**NumPy (.npz)**
```python
# Arrays:
X: (N, window_size, 1)  # Pre-windowed data
```

**CSV (.csv)**
```csv
timestamp,sensor1,sensor2,sensor3,...
2024-01-01,123.45,234.56,345.67,...
```

---

## Testing Models

### Option 1: Use Existing Project Data (Easiest!)

1. Go to **Test Inference** page
2. Select model: **1D-CNN**, **LSTM**, or **iTransformer**
3. Click **"📂 Use Project Data"** button
   - This uses: `/Users/selinbardakci/BitirmeProjesi/data/custom_window_test.npz`
4. Adjust hyperparameters if needed
5. Click **"🔍 Run Prediction"**

### Option 2: Upload Test Data from Laptop

1. Go to **Test Inference** page
2. Click **"📁 Upload .csv/.npz"**
3. Select test file from your laptop
4. Click **"🔍 Run Prediction"**

### Supported Test Data Formats:

**NumPy (.npz)** - Best for earthquake/seismic data
```python
{
    'X': array of shape (N, 128, 1),  # Test windows
    'y': array of shape (N,)           # Labels (0=normal, 1=anomaly) - optional
}
```

**CSV (.csv)** - Good for financial/general time series
```csv
timestamp,value1,value2,value3
2024-01-01,100.5,200.3,150.2
2024-01-02,105.2,198.7,152.1
```

---

## Where Files Are Stored

### Your Laptop Structure:
```
BitirmeProjesi/
├── webapp/
│   ├── app.py              ← Flask app (run this!)
│   ├── uploads/            ← Your uploaded files go here
│   ├── models/             ← Newly trained models saved here
│   ├── templates/          ← HTML pages
│   └── static/             ← CSS, JavaScript
├── 1dcnn/
│   └── cnn_ae_best.keras   ← Pre-trained CNN model
├── lstm/
│   └── lstm_ae_best.keras  ← Pre-trained LSTM model
├── iTransformer/
│   └── itransformer_ae_best.keras  ← Pre-trained iTransformer
└── data/
    └── custom_window_test.npz  ← Default test data
```

### What Happens When You Use the App:

**Training:**
- Upload file → Saved to `webapp/uploads/`
- Train model → Saved to `webapp/models/{model_type}_ae_best.keras`
- All processing happens on your laptop (CPU/GPU)

**Testing:**
- Upload file → Saved to `webapp/uploads/`
- Use existing → Loads from `../data/custom_window_test.npz`
- Models loaded from either `webapp/models/` or parent directories
- Results displayed in browser

---

## Features

✅ **Fully Local** - No internet or external services needed
✅ **Real-time Training** - Watch loss curves update live
✅ **Interactive Charts** - Beautiful visualizations
✅ **File Upload** - Drag and drop from your laptop
✅ **Use Existing Data** - Quick testing with project data
✅ **Multiple Models** - CNN, LSTM, iTransformer
✅ **Anomaly Detection** - Visualize detected anomalies

---

## Example Workflow

### 1. Test Existing Models First
```
1. Start app: python app.py
2. Open: http://localhost:5000
3. Go to: Test Inference
4. Click: "Use Project Data"
5. Click: "Run Prediction"
6. See results instantly!
```

### 2. Train Your Own Model
```
1. Go to: Training Laboratory
2. Choose: iTransformer model
3. Upload: your_data.pkl from laptop
4. Click: "Start Training"
5. Watch: real-time progress
6. Result: new model saved to models/
```

### 3. Test Your New Model
```
1. Go to: Test Inference
2. Model automatically available
3. Upload or use existing test data
4. Run prediction
5. Analyze results
```

---

## Tips

💡 **Faster Training**: Reduce epochs to 20-30 for quick tests
💡 **GPU Acceleration**: TensorFlow will auto-use GPU if available
💡 **Large Files**: Increase `MAX_CONTENT_LENGTH` in app.py if needed
💡 **Multiple Tests**: Upload different test files to compare results
💡 **Save Results**: Take screenshots of charts for reports

---

## Troubleshooting

**Port in use:**
```bash
# Kill existing process
lsof -ti:5000 | xargs kill -9
```

**File too large:**
```python
# In app.py, increase:
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1GB
```

**Model not found:**
- Check models exist in parent directories
- Or train a new model first

---

## No Database, No Cloud, No Problem! 🚀

Everything stays on your laptop:
- ✅ Data uploaded → Local `uploads/` folder
- ✅ Models trained → Local `models/` folder  
- ✅ Processing → Your CPU/GPU
- ✅ Results → Shown in browser
- ✅ Privacy → Complete data privacy

Perfect for graduation projects, research, and local development!
