from typing import Callable, Any, Dict
import operator

class Constraint:
    def __init__(self, name: str, metric_function: Callable, threshold: float, op: str):
        self.name = name
        self.metric_function = metric_function
        self.threshold = threshold
        self.op_str = op
        self.operator_func = self._parse_op(op)
        
    def _parse_op(self, op: str):
        ops = {
            ">=": operator.ge,
            "<=": operator.le,
            ">": operator.gt,
            "<": operator.lt,
            "==": operator.eq,
        }
        if op not in ops:
            raise ValueError(f"Operator {op} not supported. Must be one of {list(ops.keys())}.")
        return ops[op]

    def evaluate(self, y_true, y_pred, y_prob=None) -> Dict[str, Any]:
        """Runs the metric function and evaluates against threshold."""
        actual_value = self.metric_function(y_true, y_pred, y_prob)
        passed = self.operator_func(actual_value, self.threshold)
        deviation = actual_value - self.threshold
        
        return {
            "name": self.name,
            "passed": bool(passed),
            "actual_value": actual_value,
            "threshold": self.threshold,
            "operator": self.op_str,
            "deviation": deviation,
            "reason": f"Required {self.op_str} {self.threshold}, got {actual_value:.4f}" if not passed else None
        }

class PredictorValidationRule:
    """Allows custom conditional rules: IF condition(row) THEN expectation(pred)."""
    def __init__(self, name: str, condition: Callable[[Dict], bool], expectation: Callable[[Any], bool]):
        self.name = name
        self.condition = condition
        self.expectation = expectation

    def evaluate(self, X_df, y_pred) -> Dict[str, Any]:
        violations = []
        for idx, row in X_df.iterrows():
            if self.condition(row.to_dict()):
                if not self.expectation(y_pred[idx]):
                    violations.append({
                        "index": int(idx),
                        "row_data": row.to_dict(),
                        "prediction": float(y_pred[idx])
                    })
        return {
            "name": self.name,
            "passed": len(violations) == 0,
            "violation_count": len(violations),
            "sample_violations": violations[:10]  # Cap list to prevent massive logs
        }
