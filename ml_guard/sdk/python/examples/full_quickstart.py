"""
examples/full_quickstart.py — ML Guard SDK End-to-End Demo

Demonstrates every unique ML Guard feature vs Evidently/WhyLabs:

  1. Client initialization
  2. Prediction logging (fire-and-forget)
  3. Model wrapping (zero-code instrumentation)
  4. Decorator-based monitoring
  5. Privacy-preserving data profiles
  6. Policy test suites (Evidently-style + governance)
  7. Governance-aware CI/CD gate
  8. Compliance certificate generation

Run:
    pip install "mlguard[sklearn]"
    python examples/full_quickstart.py
"""
import sys
import os

# If running from source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sdk/python"))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import ml_guard

# ── 0. Setup ──────────────────────────────────────────────────────────────────

ML_GUARD_HOST = os.getenv("MLGUARD_HOST", "http://localhost:8000")
ML_GUARD_KEY  = os.getenv("MLGUARD_API_KEY", "your-api-key-here")
MODEL_ID      = os.getenv("DEV_MODEL_ID", "f9597635-5c66-4b17-9e4b-38e3fde81a53")

print("=" * 62)
print("  ML Guard SDK -- Full Quickstart Demo")
print("=" * 62)

# ── 1. Client initialization ──────────────────────────────────────────────────

print("\n[1] Initializing ML Guard client...")
client = ml_guard.Client(host=ML_GUARD_HOST, api_key=ML_GUARD_KEY)
print(f"    ✓ Connected: {ML_GUARD_HOST}")

# ── 2. Create sample data and model ───────────────────────────────────────────

print("\n[2] Building sample model + data...")
X, y = make_classification(n_samples=1000, n_features=5,
                           n_informative=3, random_state=42)
feature_names = ["age", "spend", "tenure", "login_count", "support_tickets"]
X_df = pd.DataFrame(X, columns=feature_names)
y_s  = pd.Series(y, name="target")

X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2, random_state=42)
train_df = pd.concat([X_train, y_train], axis=1)
test_df  = pd.concat([X_test,  y_test],  axis=1)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print(f"    ✓ Model trained — accuracy: {model.score(X_test, y_test):.2%}")

# ── 3. Prediction logging (fire-and-forget) ───────────────────────────────────

print("\n[3] Logging predictions (fire-and-forget)...")
for i, (idx, row) in enumerate(X_test.head(10).iterrows()):
    features = row.to_dict()
    pred = int(model.predict([list(features.values())])[0])
    proba = float(model.predict_proba([list(features.values())])[0].max())
    result = client.log(
        model_id=MODEL_ID,
        features=features,
        prediction=pred,
        proba=proba,
        latency_ms=4.2,
        environment="demo",
    )
print(f"    ✓ Logged 10 predictions")

# ── 4. Model wrapping (zero-code instrumentation) ─────────────────────────────

print("\n[4] Wrapping model for zero-code instrumentation...")
monitored = ml_guard.wrap_sklearn(
    model,
    model_id=MODEL_ID,
    client=client,
    feature_names=feature_names,
    profile_every=50,
)
_ = monitored.predict(X_test.values)
_ = monitored.predict_proba(X_test.values)
print(f"    ✓ Model wrapped — {len(X_test)} preds auto-logged")
print(f"    ✓ Wrapper: {monitored!r}")

# ── 5. Decorator monitoring ───────────────────────────────────────────────────

print("\n[5] Decorator-based monitoring...")

@ml_guard.monitor(model_id=MODEL_ID, client=client, environment="demo")
def predict_single(features: dict) -> int:
    return int(model.predict([list(features.values())])[0])

for row in X_test.head(5).to_dict(orient="records"):
    predict_single(row)

print(f"    ✓ @monitor: 5 predictions auto-logged to ML Guard")

# ── 6. Privacy-preserving data profiles ──────────────────────────────────────

print("\n[6] Building privacy-preserving data profile...")
prof = ml_guard.profile.from_dataframe(
    df=test_df,
    model_id=MODEL_ID,
    dataset_name="test_split",
    label_col="target",
    client=client,
)
print(f"    ✓ Profile built: {prof!r}")

quality = prof.quality_report()
print(f"    ✓ Quality score: {quality['quality_score']}/100")
print(f"    ✓ Issues: {len(quality['issues'])}")

# Diff two profiles (privacy-safe — no raw data)
ref_prof = ml_guard.profile.from_dataframe(
    df=train_df, model_id=MODEL_ID, dataset_name="train",
    label_col="target", client=client,
)
diff = prof.diff(ref_prof)
print(f"    ✓ Profile diff: {diff['drifted_columns']} drifted columns "
      f"/ {diff['total_columns']} total")

compact_json = prof.to_json()
print(f"    ✓ Compact JSON size: {len(compact_json)} bytes "
      f"(raw CSV: ~{len(test_df.to_csv())//1024}KB)")

# Upload profile (non-blocking)
try:
    client.upload_profile(prof)
    print(f"    ✓ Profile uploaded to backend")
except Exception as e:
    print(f"    ~ Profile upload: {e} (backend may not be running)")

# ── 7. Policy Test Suite ──────────────────────────────────────────────────────

print("\n[7] Running Policy Test Suite...")

suite = ml_guard.Suite(model_id=MODEL_ID, name="Production Quality Gate")

suite.add(ml_guard.tests.accuracy_above(0.75))
suite.add(ml_guard.tests.drift_psi_below(0.30))
suite.add(ml_guard.tests.null_rate_below(0.10))
suite.add(ml_guard.tests.custom(
    name="feature_count_stable",
    fn=lambda ctx: len(ctx["df_current"].columns) == len(ctx["df_reference"].columns),
    message="Feature count mismatch between reference and current"
))

results = suite.run(
    df_reference=train_df,
    df_current=test_df,
    model=model,
    label_col="target",
)
results.print_summary()

# ── 8. Governance score check ─────────────────────────────────────────────────

print("\n[8] Fetching governance score...")
try:
    score_data = client.get_score(MODEL_ID)
    print(f"    Score:   {score_data['overall_score']}")
    print(f"    Verdict: {score_data['verdict']}")
    for component, score in score_data['component_scores'].items():
        print(f"    - {component:15}: {score:.1f}")
except Exception as e:
    print(f"    ~ Score fetch: {e}")

# ── 9. Compliance certificate ─────────────────────────────────────────────────

print("\n[9] Generating compliance certificate...")
try:
    cert = client.certify(MODEL_ID)
    print(f"    ✓ cert_hash: {cert.get('cert_hash', '')[:32]}...")
    print(f"    ✓ Verdict:   {cert.get('verdict')}")
    print(f"    ✓ Score:     {cert.get('overall_score')}")
    print(f"    ✓ Verify at: http://localhost:3000/verify/{cert.get('cert_hash','')[:16]}...")
except Exception as e:
    print(f"    ~ Certificate: {e}")

# ── 10. @gate decorator demo ──────────────────────────────────────────────────

print("\n[10] Governance gate check (@gate decorator)...")

@ml_guard.gate(model_id=MODEL_ID, min_score=30.0, client=client, fail_open=True)
def deploy_demo():
    print("    ✓ @gate PASSED — deployment authorized!")
    return True

try:
    deploy_demo()
except RuntimeError as e:
    print(f"    ✗ @gate BLOCKED: {e}")

print("\n" + "=" * 62)
print("  [OK] ML Guard SDK Quickstart Complete!")
print("=" * 62)
print()
print("  Next steps:")
print("  1. pip install 'mlguard[sklearn]'")
print("  2. Wrap your real model with ml_guard.wrap_sklearn()")
print("  3. Add @ml_guard.gate() to your deploy functions")
print("  4. View dashboard at http://localhost:3000")
print()
