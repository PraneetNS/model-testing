import pandas as pd
from niyantrana.local.fairness import check_fairness

def main():
    print("Evaluating fairness locally on a dataset...")
    
    # Create a synthetic dataset demonstrating bias
    data = {
        "age": [25, 45, 30, 50, 22, 60],
        "gender": ["M", "M", "M", "F", "F", "F"],
        "true_label": [1, 1, 0, 1, 0, 0],
        "prediction": [1, 1, 1, 0, 0, 0] # Model is biased towards 'M'
    }
    df = pd.DataFrame(data)
    
    print("\nDataset:")
    print(df)
    
    report = check_fairness(
        df=df,
        label_col="true_label",
        prediction_col="prediction",
        sensitive_features=["gender"]
    )
    
    print(f"\nOverall Fair: {report.overall_fair}")
    for feature in report.per_feature:
        print(f"\nFeature: {feature.feature}")
        print(f" Demographic Parity Diff: {feature.demographic_parity_diff:.3f}")
        print(f" Equalized Odds Diff: {feature.equalized_odds_diff:.3f}")
        print(f" Disparate Impact Ratio: {feature.disparate_impact_ratio:.3f}")
        if feature.flags:
            print(" Flags:")
            for flag in feature.flags:
                print(f"  - {flag}")

if __name__ == "__main__":
    main()
