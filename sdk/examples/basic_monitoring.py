import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from niyantrana.client import NiyantranaClient
from niyantrana.local.drift import detect_drift
from niyantrana.local.fairness import check_fairness

def main():
    print("1. Generating synthetic data and training a model...")
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=42)
    feature_names = [f"feature_{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    
    # Add a sensitive feature for fairness
    np.random.seed(42)
    df["gender"] = np.random.choice(["Male", "Female"], size=len(df))

    X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
        df[feature_names], df["label"], df["gender"], test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("\n2. Computing drift on held-out data...")
    # Simulate drift by adding noise to test data
    X_test_drifted = X_test.copy()
    X_test_drifted["feature_0"] = X_test_drifted["feature_0"] + 2.0

    report = detect_drift(X_train, X_test_drifted, method="psi")
    print(f"Overall Drift Detected: {report.overall_drift_detected}")
    for feature in report.per_feature:
        if feature.drifted:
            print(f" - {feature.feature} drifted (PSI: {feature.statistic:.3f})")

    print("\n3. Running fairness check...")
    # Get predictions for test set
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["label"] = y_test.values
    test_df["gender"] = gender_test.values
    test_df["prediction"] = model.predict(X_test)

    fairness_report = check_fairness(
        df=test_df, 
        label_col="label", 
        prediction_col="prediction", 
        sensitive_features=["gender"]
    )
    
    print(f"Overall Fair: {fairness_report.overall_fair}")
    for f in fairness_report.per_feature:
        print(f" - {f.feature}: DP Diff={f.demographic_parity_diff:.3f}, EO Diff={f.equalized_odds_diff:.3f}, DI Ratio={f.disparate_impact_ratio:.3f}")
        for flag in f.flags:
            print(f"   * Flag: {flag}")

    print("\n4. Logging a prediction (local mode)...")
    client = NiyantranaClient() # No API key = local mode
    
    sample_features = X_test.iloc[0].to_dict()
    pred = int(model.predict([X_test.iloc[0]])[0])
    prob = float(model.predict_proba([X_test.iloc[0]])[0][1])
    
    client.log_prediction(
        model_id="my_sklearn_model",
        features=sample_features,
        prediction=pred,
        probability=prob,
        latency_ms=12.5
    )
    print("Done!")

if __name__ == "__main__":
    main()
