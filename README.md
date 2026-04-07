# Spam and Malware Detection System

A complete full-stack machine learning platform combining **training/modeling pipelines** with a **production-ready web application**.

---

## 🎯 Project Overview

**Part 3** integrates:
- **ML training, data processing, model evaluation, and reporting**
- **FastAPI backend with spam prediction API and React frontend UI**

This unified structure contains everything needed to:
1. Train models on spam and malware datasets
2. Deploy models via REST API
3. Interact with predictions through a modern web UI
4. Manage history and feedback at the application level

---

## ✅ Pre-Installation Checklist

Before starting, ensure you have:

- **Python 3.10+** installed (check: `python --version`)
- **pip** package manager (check: `pip --version`)
- **Node.js 14+** and **npm** (check: `node --version` and `npm --version`)
- **Virtual Environment** support
- **Git** (optional, for version control)
- **Internet access** (needed for first-time NLTK data download)

---

## 🚀 Quick Start

### Fastest Setup (5 minutes)

```bash
# 1. Navigate to part3 directory
cd part3

# 2. Create and activate Python virtual environment
python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Start the FastAPI server (Terminal 1)
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 5. Start the React frontend (Terminal 2)
cd frontend
npm install
npm start
```

**Then visit:** http://localhost:3000

---

## 📁 Directory Structure

```
part3/
├── backend/                    # FastAPI application
│   ├── main.py                # API entry point
│   ├── routers/               # Endpoint handlers
│   ├── utils/                 # Services, logging, middleware
│   ├── schemas/               # Request/response validation
│   ├── models/                # Trained ML model artifacts
│   └── requirements.txt        # Backend dependencies
│
├── frontend/                   # React application
│   ├── src/                   # React components, pages, services
│   ├── public/                # Static assets
│   └── package.json           # Frontend dependencies
│
├── src/                       # Reusable ML utilities
│   ├── common/                # Shared tools (paths, evaluation, text processing)
│   ├── spam_detection/        # Spam-specific modules
│   └── malware_detection/     # Malware-specific modules
│
├── models/                    # Trained model artifacts
│   ├── spam/                  # Spam detection models
│   └── malware/               # Malware detection models
│
├── data/                      # Datasets
│   ├── raw/                   # Original datasets
│   └── processed/             # Cleaned/processed datasets
│
├── scripts/                   # Training & processing scripts
│   ├── train_spam_*.py        # Model training scripts
│   ├── process_spam.py        # Data preprocessing
│   └── ... (other scripts)
│
├── notebooks/                 # Jupyter notebooks
├── requirements.txt           # Combined dependencies
├── .env.example               # Environment configuration template
├── TODO.md                    # Development tasks
└── __init__.py                # Makes root a package
```

---

## 🔧 Detailed Setup Instructions

### Step 1: Backend Setup (FastAPI)

#### 1.1 Create and Activate Virtual Environment

```bash
# Navigate to part3 directory
cd part3

# Create virtual environment
python -m venv .venv

# Activate it (choose based on your OS):

# ✅ Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# ✅ Windows (Command Prompt)
.\.venv\Scripts\activate.bat

# ✅ macOS/Linux
source .venv/bin/activate
```

**Expected output:** You should see `(.venv)` prefix in your terminal.

#### 1.2 Install Python Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

**Expected packages:**
- fastapi, uvicorn
- scikit-learn, pandas, numpy
- matplotlib, seaborn, nltk
- And more...

#### 1.3 Configure Backend Environment (Optional)

Create `.env` file in `part3/backend/`:

```bash
# Copy the example file
cp .env.example backend/.env

# Edit backend/.env if needed (most defaults work fine)
```

**Default .env works because:**
- Frontend runs on `http://localhost:3000`
- Models auto-load from `models/spam/multinomial_nb.joblib`

#### 1.4 Test Backend Startup

```bash
# From part3 root with .venv activated
cd backend

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
INFO:     Loaded pipeline fallback from C:\...\models\spam\multinomial_nb.joblib
```

**Quick API Tests:**
- Health check: Visit http://localhost:8000/health
- API docs: Visit http://localhost:8000/docs
- Swagger UI: http://localhost:8000/redoc

**To stop the server:** Press `Ctrl+C`

### Step 2: Frontend Setup (React)

#### 2.1 Install Node Dependencies

**Open a NEW terminal** (keep backend running in first terminal)

```bash
# Navigate to frontend directory
cd part3/frontend

# Install Node packages
npm install
```

**Expected output:**
```
up to date, audited XX packages in Xs
```

#### 2.2 Start React Development Server

With **backend still running** in the first terminal:

```bash
# From part3/frontend directory
npm start
```

**Expected output:**
```
Compiled successfully!
You can now view the app in the browser.
  Local:            http://localhost:3000
```

Browser should automatically open. If not, visit: **http://localhost:3000**

---

## 🎯 Verify Full Integration

### Check All Systems Running:

| Component | Expected Status | URL |
|-----------|-----------------|-----|
| Backend API | ✅ Running | http://localhost:8000 |
| API Docs | ✅ Swagger UI | http://localhost:8000/docs |
| Frontend | ✅ Running | http://localhost:3000 |
| Model | ✅ Loaded | Check backend console logs |

### Test Spam Prediction:

1. **Via Frontend UI:**
   - Navigate to http://localhost:3000
   - Type a test message: "Free money! Click here now!!!"
   - Click "Submit" or "Predict"
   - Should show: **Spam** probability

2. **Via API (manual):**
   ```bash
   # Open another terminal (keep backend running)
   curl -X POST "http://localhost:8000/predict_spam" \
     -H "Content-Type: application/json" \
     -d '{"message":"Free money! Click here now!!!"}'
   ```

   **Expected response:**
   ```json
   {
     "prediction": "spam",
     "spam_probability": 0.95,
     "source": "pipeline"
   }
   ```

3. **Via Swagger UI:**
   - Go to http://localhost:8000/docs
   - Find "POST /predict_spam"
   - Click "Try it out"
   - Enter test message
   - Click "Execute"

---

## 📊 Key Features

### Backend (FastAPI)
- ✅ High-performance REST API
- ✅ Input validation with Pydantic schemas
- ✅ Request logging and error handling
- ✅ SQLite-based history persistence
- ✅ CORS enabled for frontend integration
- ✅ Graceful model loading with fallbacks

### Frontend (React)
- ✅ Real-time spam detection UI
- ✅ Prediction history tracking
- ✅ Dark mode support
- ✅ User feedback submission
- ✅ Interactive charts and metrics
- ✅ Responsive design

### ML Pipeline
- ✅ Automated data cleaning (spam + malware)
- ✅ Multiple model types (Classification, Clustering, Regression)
- ✅ Comprehensive evaluation metrics
- ✅ Publication-quality visualizations
- ✅ Reusable utilities for NLP preprocessing

---

## 🔧 Model Configuration

The backend automatically loads spam detection models in this order:

1. **Standalone model+vectorizer** at `part3/backend/models/spam_model.pkl` + `spam_vectorizer.pkl`
2. **Packaged pipeline** at `part3/backend/models/*.joblib`
3. **Environment variable** `SPAM_PIPELINE_PATH` (if set)
4. **Fallback** to `part3/models/spam/multinomial_nb.joblib` (recommended)

Current setup uses the **Multinomial Naive Bayes** model from Part 1: `models/spam/multinomial_nb.joblib`

---

## 🔬 Using ML Training Scripts (Optional)

If you want to retrain models or process data:

```bash
# Ensure you're in the part3 environment
# From part3 root with .venv activated

# Process spam data
python scripts/process_spam.py

# Train a spam model
python scripts/train_spam_nb.py

# Generate evaluation plots
python scripts/generate_evaluation_plots.py
```

All scripts will output to `data/processed/` and `models/spam/` directories.

---

## 🔐 Environment Variables

Create a `.env` file in `part3/backend/` if needed:

```bash
# For custom frontend origin
FRONTEND_ORIGIN=http://localhost:3000

# Optional: specify custom model path
SPAM_PIPELINE_PATH=/path/to/custom/model.joblib

# Optional: email configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
RECIPIENT_EMAIL=recipient@example.com
```

---

## 📚 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict_spam` | Classify a message as spam/ham |
| `GET` | `/history` | Get prediction history |
| `PUT` | `/history/{id}` | Update history entry |
| `DELETE` | `/history/{id}` | Delete history entry |
| `POST` | `/feedback` | Submit user feedback |
| `GET` | `/health` | Health check |

### Example API Usage

```bash
# Test prediction
curl -X POST "http://localhost:8000/predict_spam" \
  -H "Content-Type: application/json" \
  -d '{"message":"Free money! Click here now!!!"}'

# Get history
curl http://localhost:8000/history

# Health check
curl http://localhost:8000/health
```

**Interactive API Docs:** http://localhost:8000/docs

---

## 🛑 Troubleshooting

### Issue: "Module not found" error when starting backend

**Solution:**
```bash
# Make sure you're in part3 root
cd part3

# Check venv is activated (.venv) prefix visible
# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Port 8000 or 3000 already in use

**Solution:**
```bash
# Run on different ports
# Backend on 8001:
uvicorn main:app --port 8001 --reload

# Frontend on 3001:
PORT=3001 npm start
```

Then update backend's CORS to include new frontend origin.

### Issue: Model not loading (see "model_not_loaded" in health check)

**Solution:**
```bash
# Check models exist
ls -la part3/models/spam/

# Verify multinomial_nb.joblib is there
# It should be ~5MB

# Check backend logs for error messages
# Usually shows what tried to read but failed
```

### Issue: CORS errors (5xx errors when frontend calls backend)

**Solution:**
```bash
# Check .env has correct FRONTEND_ORIGIN
# Default: http://localhost:3000

# If using different Frontend port:
# 1. Update part3/backend/.env:
FRONTEND_ORIGIN=http://localhost:3001

# 2. Restart backend
```

### Issue: NLTK data not found (on first run with new environment)

**Solution:**
```bash
# First-time download happens automatically
# Just ensure you have internet access
# Or manually download:
python -c "import nltk; nltk.download('stopwords')"
```

---

## 📊 Model Information

### Active Models in Production

**Spam Detection:**
- **Model:** Multinomial Naive Bayes (`multinomial_nb.joblib`)
- **Format:** scikit-learn pipeline with TF-IDF vectorizer
- **Accuracy:** High baseline for spam classification
- **Features:** Count + TF-IDF vectors from text

**Additional Models:** (via scripts)
- Logistic Regression with L2 regularization
- Elastic Net regressor for confidence scores
- K-Means clustering for document similarity

---

## 🎓 Academic Use

This integrated system is suitable for:
- Machine learning coursework submissions
- Spam/malware detection research
- Full-stack ML application demonstrations
- System design and architecture case studies

All components are production-ready with:
- Proper error handling and logging
- Input validation and security measures
- Clean architecture and separation of concerns
- Comprehensive README and documentation


