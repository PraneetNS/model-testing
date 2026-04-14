"""Test for behavioral test endpoint."""
import httpx, asyncio, io, joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate training data
X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42)
feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5']

clf = RandomForestClassifier(n_estimators=5, random_state=42)
df_X = pd.DataFrame(X, columns=feature_cols)
clf.fit(df_X, y)

model_buf = io.BytesIO()
joblib.dump(clf, model_buf)
model_buf.seek(0)
model_bytes = model_buf.read()

ref_df = pd.DataFrame(X, columns=feature_cols)
ref_df['target'] = y
ref_csv = ref_df.to_csv(index=False).encode()

API_KEY = 'mlg_PeNfpwQSOtJkWr1Tow62Kr5luLuEugGi'

async def test_behavior():
    hdr = {'X-API-Key': API_KEY}
    async with httpx.AsyncClient(timeout=30.0) as c:
        # Use 'ref_file' as expected by our fix
        r = await c.post(
            'http://localhost:8000/api/v1/behavior/test',
            files={
                'model_file': ('model.pkl', model_bytes, 'application/octet-stream'),
                'ref_file': ('ref.csv', ref_csv, 'text/csv'),
            },
            data={
                'scenarios': 'sensitivity_analysis,noise_perturbation',
                'label_col': 'target',
            },
            headers=hdr
        )
        print(f'Status: {r.status_code}')
        if r.status_code == 200:
            data = r.json()
            print(f"Robustness score: {data.get('robustness_score')}")
            print(f"Status: {data.get('status')}")
            print("SUCCESS - behavioral test working!")
        else:
            print(f"ERROR {r.status_code}: {r.text[:600]}")

asyncio.run(test_behavior())
