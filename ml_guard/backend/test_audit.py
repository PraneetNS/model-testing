"""End-to-end audit integration test."""
import httpx, asyncio, io, joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate training data
X, y = make_classification(n_samples=300, n_features=5, random_state=42)
feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5']

clf = RandomForestClassifier(n_estimators=10, random_state=42)
df_train_X = pd.DataFrame(X[:200], columns=feature_cols)
clf.fit(df_train_X, y[:200])

model_buf = io.BytesIO()
joblib.dump(clf, model_buf)
model_buf.seek(0)
model_bytes = model_buf.read()

train_df = pd.DataFrame(X[:200], columns=feature_cols)
train_df['target'] = y[:200]
train_csv = train_df.to_csv(index=False).encode()

val_df = pd.DataFrame(X[200:], columns=feature_cols)
val_df['target'] = y[200:]
val_csv = val_df.to_csv(index=False).encode()

API_KEY = 'mlg_PeNfpwQSOtJkWr1Tow62Kr5luLuEugGi'

async def run_audit():
    hdr = {'X-API-Key': API_KEY}
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(
            'http://localhost:8000/api/v1/audit/run',
            files={
                'model_file': ('model.pkl', model_bytes, 'application/octet-stream'),
                'train_file': ('train.csv', train_csv, 'text/csv'),
                'val_file': ('val.csv', val_csv, 'text/csv'),
            },
            data={
                'model_name': 'TestModel-RF',
                'label_col': 'target',
                'selected': ['accuracy', 'f1', 'psi_drift', 'calibration_check'],
            },
            headers=hdr
        )
        print(f'Status: {r.status_code}')
        if r.status_code == 200:
            data = r.json()
            print(f"Audit status: {data.get('status')}")
            print(f"Scan ID: {data.get('scan_id')}")
            gov = data.get('governance', {})
            print(f"Governance score: {gov.get('governance_score')}")
            print(f"Deployment allowed: {gov.get('deployment_allowed')}")
            print(f"Risk level: {data.get('risk_level')}")
            print(f"Metrics: {data.get('metrics')}")
            pol = data.get('policy', {})
            print(f"Policy gate: {pol.get('gate_status')}")
            print(f"Advisories: {len(data.get('advisories', []))}")
            print(f"Drifted features: {data.get('top5_drifted_features', [])}")
            print("SUCCESS - audit pipeline working!")
        else:
            print(f"ERROR {r.status_code}: {r.text[:600]}")

asyncio.run(run_audit())
