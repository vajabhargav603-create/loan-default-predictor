# Task 6: ML Project Deployment Guide

This project is fully configured and ready for live deployment across all platforms specified in **Task 6**:
1. **Part 1**: Streamlit Community Cloud (Single Python App)
2. **Part 2**: Render (FastAPI / Flask Backend API) + Vercel (Frontend Web Interface)

---

## 📌 Quick Summary of Platform Configurations

| Platform | Deployment Target | Root File / Entry Point | Configuration File |
| :--- | :--- | :--- | :--- |
| **Streamlit Cloud** | Full-Stack App | `streamlit_app.py` | `requirements.txt` |
| **Render** | Backend API | `week 4/backend/main.py` (FastAPI) or `app.py` (Flask) | `render.yaml` / `Procfile` |
| **Vercel** | Frontend Web UI | `week 4/frontend/index.html` | `vercel.json` |

---

## 🚀 Part 1: Deploying to Streamlit Community Cloud

Deploy the complete interactive Loan Default prediction application directly from GitHub in minutes.

### Step-by-Step Instructions:
1. Push your project repository to GitHub:
   ```bash
   git add .
   git commit -m "Configure Task 6 Streamlit & Web deployment"
   git push origin main
   ```
2. Open **[share.streamlit.io](https://share.streamlit.io)** in your browser and sign in with GitHub.
3. Click **"New app"** from your workspace.
4. Fill in the deployment parameters:
   - **Repository**: Select your GitHub repository.
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Click **Deploy**. Streamlit Cloud will install dependencies from `requirements.txt` and launch the app at `https://<your-app-name>.streamlit.app`.

---

## ⚡ Part 2: Deploying FastAPI Backend (Render) & React/HTML Frontend (Vercel)

### Step A: Deploy Backend API to Render (`render.com`)
1. Sign in to **[render.com](https://render.com)** using your GitHub account.
2. Click **New +** $\rightarrow$ **Web Service**.
3. Select your GitHub repository.
4. Configure the Web Service settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn week 4.backend.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`
5. Click **Create Web Service**. Once deployed, copy your backend live URL (e.g. `https://loanguard-backend.onrender.com`).

---

### Step B: Deploy Frontend UI to Vercel (`vercel.com`)
1. Sign in to **[vercel.com](https://vercel.com)** using your GitHub account.
2. Click **Add New...** $\rightarrow$ **Project**.
3. Import your GitHub repository.
4. In project settings:
   - Set **Root Directory** to `week 4/frontend` (or leave default to use root `vercel.json`).
5. Click **Deploy**. Vercel will instantly serve your static frontend with global CDN acceleration.

---

## 🛠️ Local Execution Commands

### Run Streamlit App Locally:
```powershell
streamlit run streamlit_app.py
```

### Run Flask Backend Server Locally:
```powershell
python "week 4/backend/app.py"
```

### Run FastAPI Server Locally:
```powershell
uvicorn "week 4.backend.main:app" --reload --port 8000
```
