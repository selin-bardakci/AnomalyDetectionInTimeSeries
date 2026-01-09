# Training Data Upload Guide

## ✅ Ready to Upload Training Data!

Your webapp is fully configured to accept training data uploads from your laptop.

---

## Upload Limits

**Maximum File Size:** 2GB per upload

This is large enough for:
- ✅ Thousands of seismic traces
- ✅ Years of financial data
- ✅ Large time series datasets
- ✅ Multiple sensor recordings

---

## Supported Training Data Formats

### 1. **Pickle Files (.pkl)** - BEST for STEAD-style data

**Format Requirements:**
```python
import pandas as pd

# Your DataFrame should have this structure:
df = pd.DataFrame({
    'Z_channel': [
        np.array([...]),  # Trace 1: 1D time series
        np.array([...]),  # Trace 2: 1D time series
        np.array([...]),  # Trace 3: 1D time series
        # ... more traces
    ]
})

# Save it
df.to_pickle('my_training_data.pkl')
```

**What the app does:**
- Takes each trace in `Z_channel`
- Normalizes it (z-score)
- Creates sliding windows (128 samples, 64 stride)
- Limits to 100 windows per trace
- Uses first 4000 traces
- Splits 80/20 train/validation

**Example sizes:**
- 1000 traces × 6000 samples = ~46 MB
- 4000 traces × 6000 samples = ~183 MB
- 10000 traces × 6000 samples = ~458 MB

---

### 2. **NumPy Files (.npz)** - Pre-windowed data

**Format Requirements:**
```python
import numpy as np

# Pre-window your data
X = np.random.randn(10000, 128, 1)  # (N_windows, lookback, channels)

# Save it
np.savez('my_training_windows.npz', X=X)
```

**What the app does:**
- Directly uses your pre-windowed data
- Splits 80/20 train/validation
- No additional windowing needed

**Example sizes:**
- 10,000 windows × 128 × 1 = ~10 MB
- 100,000 windows × 128 × 1 = ~100 MB
- 1,000,000 windows × 128 × 1 = ~1 GB

---

### 3. **CSV Files (.csv)** - Time series columns

**Format Requirements:**
```csv
timestamp,sensor1,sensor2,sensor3
2024-01-01 00:00:00,123.45,234.56,345.67
2024-01-01 00:00:01,123.50,234.60,345.70
2024-01-01 00:00:02,123.55,234.65,345.75
...
```

**What the app does:**
- Reads each sensor column (except timestamp)
- Normalizes each series
- Creates sliding windows
- Combines all sensor data

**Example sizes:**
- 10,000 rows × 5 columns = ~1 MB
- 100,000 rows × 10 columns = ~20 MB
- 1,000,000 rows × 20 columns = ~400 MB

---

## How to Upload Training Data

### Step 1: Prepare Your Data

**For Earthquake/Seismic Data (use .pkl):**
```python
import pandas as pd
import numpy as np

# Load your STEAD or custom seismic data
traces = []
for i in range(1000):
    trace = load_seismic_trace(i)  # Your loading function
    traces.append(trace)

df = pd.DataFrame({'Z_channel': traces})
df.to_pickle('earthquake_training.pkl')
```

**For Financial Data (use .csv):**
```python
import pandas as pd

# Your financial time series
df = pd.DataFrame({
    'timestamp': pd.date_range('2020-01-01', periods=10000, freq='1H'),
    'price': np.random.randn(10000).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 10000),
    'volatility': np.random.randn(10000)
})

df.to_csv('financial_training.csv', index=False)
```

---

### Step 2: Upload via Web Interface

1. **Start the app:**
   ```bash
   cd /Users/selinbardakci/BitirmeProjesi/webapp
   python app.py
   ```

2. **Open browser:** http://localhost:5000/training

3. **Select dataset type:**
   - Earthquake Data (for seismic)
   - Financial Market (for financial)

4. **Choose model architecture:**
   - iTransformer (best for long sequences)
   - 1D-CNN (fast, good for patterns)
   - LSTM (good for sequential data)

5. **Upload file:**
   - Click "📁 Choose File"
   - Select your .pkl, .npz, or .csv file
   - Wait for upload confirmation

6. **Start training:**
   - Click "Start Training"
   - Watch real-time progress!

---

## Training Configuration

**Current settings (can adjust in app.py):**

```python
LOOKBACK = 128          # Window size
STRIDE = 64             # Window overlap
BATCH_SIZE = 256        # Training batch size
EPOCHS = 30             # Training epochs (default in UI)
MAX_TRACES = 4000       # Max traces from pkl
MAX_WINDOWS_PER_TRACE = 100  # Max windows per trace
```

**Training time estimates (on typical laptop):**
- 10,000 windows, 30 epochs, CNN: ~5 minutes
- 10,000 windows, 30 epochs, LSTM: ~8 minutes
- 10,000 windows, 30 epochs, iTransformer: ~10 minutes

With GPU: 3-5x faster!

---

## What Happens During Upload & Training

### Upload Process:
1. File uploaded to `webapp/uploads/`
2. File validated (format, size)
3. Basic info extracted (traces/windows count)
4. Ready indicator shown

### Training Process:
1. **Load Data** - Read uploaded file
2. **Windowing** - Create sliding windows (if needed)
3. **Normalization** - Z-score normalize each window
4. **Split** - 80% train, 20% validation
5. **Build Model** - Create CNN/LSTM/iTransformer
6. **Train** - Run epochs with early stopping
7. **Save** - Best model saved to `webapp/models/`

### Real-time Updates:
- Current epoch (e.g., "Epoch 5/30")
- Training loss curve
- Validation loss curve
- Progress percentage
- Status messages

---

## File Size Examples

### Small Dataset (Quick Testing)
- **Size:** 50-100 MB
- **Content:** 1,000-2,000 traces
- **Training time:** 3-5 minutes
- **Good for:** Testing, prototyping

### Medium Dataset (Standard)
- **Size:** 100-500 MB
- **Content:** 2,000-5,000 traces
- **Training time:** 5-15 minutes
- **Good for:** Regular training, good models

### Large Dataset (Best Results)
- **Size:** 500 MB - 2 GB
- **Content:** 5,000-20,000 traces
- **Training time:** 15-60 minutes
- **Good for:** Production models, research

---

## Memory Requirements

**Your laptop should have:**

| Dataset Size | RAM Needed | Recommended |
|-------------|-----------|-------------|
| < 100 MB | 4 GB | 8 GB |
| 100-500 MB | 8 GB | 16 GB |
| 500 MB - 1 GB | 16 GB | 32 GB |
| 1-2 GB | 32 GB | 64 GB |

**GPU (optional but faster):**
- Any NVIDIA GPU with 4GB+ VRAM will work
- Apple Silicon (M1/M2/M3) works great too

---

## Troubleshooting

### "File too large" error
```python
# In app.py, increase:
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
```

### "Out of memory" during training
```python
# In app.py, reduce batch size:
batch_size = 128  # Instead of 256

# Or limit data:
MAX_TRACES = 2000  # Instead of 4000
```

### Slow upload
- Use .npz instead of .csv (more compressed)
- Use .pkl with numpy arrays (efficient)
- Check your disk space

### Training takes too long
- Reduce epochs to 10-20 for testing
- Use smaller dataset first
- Try 1D-CNN (fastest model)

---

## Tips for Best Results

✅ **Data Quality:**
- Clean data (no NaNs, Infs)
- Consistent sampling rate
- Representative samples

✅ **Data Quantity:**
- Minimum: 500 traces or 5,000 windows
- Recommended: 2,000+ traces or 20,000+ windows
- More data = better anomaly detection

✅ **Training Strategy:**
1. Start with small dataset (quick test)
2. Use 1D-CNN first (fastest)
3. If results good, try full dataset
4. Try iTransformer for best performance

✅ **Save Your Models:**
- Models automatically saved to `webapp/models/`
- Backup important models elsewhere
- Can load them later for testing

---

## Example: Complete Training Workflow

```bash
# 1. Prepare data (in Python/Jupyter)
import pandas as pd
import numpy as np

traces = load_my_seismic_data()  # Your data source
df = pd.DataFrame({'Z_channel': traces})
df.to_pickle('my_earthquake_data.pkl')
# File size: ~200 MB

# 2. Start webapp
cd /Users/selinbardakci/BitirmeProjesi/webapp
python app.py

# 3. In browser:
# - Go to http://localhost:5000/training
# - Select "Earthquake Data"
# - Choose "iTransformer"
# - Upload "my_earthquake_data.pkl"
# - Click "Start Training"
# - Watch progress (takes ~10 minutes)

# 4. Model saved to:
# webapp/models/itransformer_ae_best.keras

# 5. Test it:
# - Go to http://localhost:5000/testing
# - Select "iTransformer"
# - Upload test data or use existing
# - Click "Run Prediction"
# - See anomaly detection results!
```

---

## Ready to Train! 🚀

Your webapp can now handle:
- ✅ Files up to 2GB
- ✅ Three formats: .pkl, .npz, .csv
- ✅ Three models: CNN, LSTM, iTransformer
- ✅ Real-time training visualization
- ✅ Automatic model saving

Just upload and train! No configuration needed.
