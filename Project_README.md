# Fakejob-prediction — Quick README

Purpose

- A small Flask web app that classifies job postings as legitimate or fraudulent using a trained LSTM text model.

High-level flow

1. User submits a job description through the web UI.
2. The server loads the saved tokenizer and model, preprocesses the text (tokenize → pad), and runs inference.
3. The numeric model output is converted to a label (e.g., "Legitimate" / "Fraudulent") and returned to the client.

Key files

- app.py — Flask app and prediction endpoint(s).
- tokenizer.pkl — saved text tokenizer (used for preprocessing).
- fake_job_lstm_model.h5 or fake_job_lstm_model.tflite — trained model used for inference.
- requirements.txt — pinned Python packages used by the project.

Technologies (what and why)

- Python 3.11 (recommended) — runtime (TensorFlow wheels are compatible with 3.11).
- Flask — web framework to serve the UI and endpoints.
- TensorFlow / Keras — model inference (Keras .h5 or TFLite).
- TensorFlow Lite (tflite interpreter) — lightweight inference option (may require select-ops/flex delegate).
- numpy, pandas, scikit-learn — preprocessing and utility functions.
- gunicorn — production WSGI server (Linux deployment).
- pickle (stdlib) — load/save tokenizer.

Run locally (Windows)

1. From repo root:
   py -3.11 -m venv .venv
2. Activate the venv:
   - PowerShell: .venv\Scripts\Activate.ps1
   - CMD: .venv\Scripts\activate.bat
   - Git Bash: source .venv/Scripts/activate
3. Upgrade pip and install deps:
   python -m pip install --upgrade pip
   pip install -r requirements.txt
4. Start the app:
   python app.py
5. Open http://127.0.0.1:5000 in a browser.

Notes / troubleshooting

- TensorFlow wheels are not available for every Python version (e.g., 3.14). Use Python 3.10 or 3.11 to avoid "No matching distribution" errors.
- If the TFLite interpreter raises "Select TensorFlow op(s) ... not supported", either:
  - Install TFLite select-ops delegate (platform-specific) OR
  - Use the full Keras `.h5` model for inference (simpler on desktop).
- To capture exact working dependencies from the current venv:
  pip freeze > requirements.txt

Committing & pushing changes (basic)
git add .
git commit -m "Update requirements and README"
git push origin BRANCH_NAME
(If push fails due to permissions, fork the repo, add your fork as a remote, push there, then open a PR.)

Example test (manual)

- Start server and paste a sample job description into the web form at http://127.0.0.1:5000 to see predictions.

Contact / next steps

- If you want, I can: (a) generate a one-page developer README with troubleshooting steps and sample HTTP requests, or (b) open a PR template and push the changes for you (you must provide push credentials or a fork remote).
