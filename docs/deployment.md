# SatQuery AI — Deployment Guide

## 1. Local Deployment (React Frontend + FastAPI Backend)

### Step A: Backend Setup & Launch
```bash
# Clone repository
git clone https://github.com/huldahtejaswinikunti-AI/SatQuery-AI.git
cd SatQuery-AI

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies and package
pip install -r requirements.txt
pip install -e .

# Launch FastAPI Mission Backend
uvicorn app.api_server:app --host 127.0.0.1 --port 8000 --reload
```
*The FastAPI REST API and Swagger interactive documentation will be available at `http://127.0.0.1:8000/docs`.*

### Step B: React Frontend Setup & Launch
```bash
# In a separate terminal window
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```
*Access the Cinematic 3D Mission Workstation in your browser at `http://localhost:5173`.*

---

## 2. Running Automated Tests

```bash
# Run the complete pytest test suite (145 tests)
python -m pytest tests/ -v
```

---

## 3. Production Cloud Deployment

### Frontend (Vercel)
1. Import repository on [Vercel](https://vercel.com).
2. Set **Root Directory** to `frontend`.
3. Set **Framework Preset** to `Vite`.
4. Build Command: `npm run build`, Output Directory: `dist`.
5. Set Environment Variable:
   - `VITE_API_BASE_URL` = `https://your-backend-service.onrender.com`
6. Deploy.

### Backend (Render / Docker / VPS)
1. Deploy using the included `render.yaml` or create a Web Service on [Render](https://render.com).
2. Build Command:
   ```bash
   pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
   ```
3. Start Command:
   ```bash
   uvicorn app.api_server:app --host 0.0.0.0 --port $PORT
   ```
4. Set Environment Variables:
   - `PYTHON_VERSION` = `3.11.9`
   - `SATQUERY_USE_MOCK_FALLBACKS` = `true` (for CPU tiers)
