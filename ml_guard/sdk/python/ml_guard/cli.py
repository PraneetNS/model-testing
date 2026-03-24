import argparse
import sys
import os
import joblib
import pandas as pd
import json
from .client import Guard

def main():
    parser = argparse.ArgumentParser(description="ML Guard CLI - Enterprise Model Quality Governance")
    subparsers = parser.add_subparsers(dest="command")

    # Evaluate Command
    eval_parser = subparsers.add_parser("evaluate", help="Upload and evaluate a model")
    eval_parser.add_argument("--model", required=True, help="Path to model file (.pkl or .joblib)")
    eval_parser.add_argument("--train", required=True, help="Path to training dataset (.csv)")
    eval_parser.add_argument("--val", required=True, help="Path to validation dataset (.csv)")
    eval_parser.add_argument("--project", default=os.getenv("MLGUARD_PROJECT", "default"), help="Project ID")
    eval_parser.add_argument("--target", default="target", help="Target column name")
    eval_parser.add_argument("--query", help="NLP query for test selection")
    eval_parser.add_argument("--suite", help="Predefined test suite name")
    eval_parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if args.command == "evaluate":
        try:
            # Load artifacts
            model = joblib.load(args.model)
            train_df = pd.read_csv(args.train)
            val_df = pd.read_csv(args.val)

            guard = Guard(project=args.project)
            result = guard.evaluate(
                model=model,
                train_df=train_df,
                val_df=val_df,
                target_column=args.target,
                query=args.query or args.suite
            )

            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("\n" + "="*50)
                print("         ML GUARD EVALUATION REPORT")
                print("="*50)
                print(f"Project:      {args.project}")
                print(f"Risk Level:   {result.get('risk_level', 'UNKNOWN')}")
                print(f"Quality Score: {result.get('score', 0.0)}/100")
                print(f"Status:       {'✅ PASSED' if result.get('deployment_allowed') else '❌ FAILED'}")
                print("="*50)

            if not result.get("deployment_allowed"):
                sys.exit(1)
            sys.exit(0)

        except Exception as e:
            print(f"Error during evaluation: {str(e)}", file=sys.stderr)
            sys.exit(2)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
