import numpy as np
import os
from typing import Callable, List, Dict, Any

def fgsm(model, X: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """
    Fast Gradient Sign Method.
    Note: Real FGSM requires gradient access. 
    If model is sklearn, we use a heuristic perturbation.
    """
    try:
        if hasattr(model, "__module__") and "sklearn" in model.__module__:
            # Heuristic perturbation for black-box sklearn
            noise = epsilon * np.sign(np.random.normal(0, 1, X.shape))
            return X + noise
        
        # Assume torch if not sklearn
        import torch
        X_tensor = torch.tensor(X, requires_grad=True).float()
        # For a real attack, we'd need the target labels. Using dummy.
        outputs = model(X_tensor)
        # Assuming classification for dummy loss
        _, preds = torch.max(outputs, 1)
        loss = torch.nn.functional.cross_entropy(outputs, preds)
        model.zero_grad()
        loss.backward()
        
        perturbed_X = X_tensor + epsilon * X_tensor.grad.data.sign()
        return perturbed_X.detach().numpy()
    except Exception:
        # Fallback to random if torch logic fails
        return X + epsilon * np.random.choice([-1, 1], size=X.shape)

def square_attack(predict_fn: Callable, X: np.ndarray, n_queries: int = 100) -> np.ndarray:
    """
    Black-box square attack.
    Randomly perturbs X to find an adversarial example that changes the prediction.
    """
    best_X = X.copy()
    try:
        initial_pred = predict_fn({"features": X[0].tolist()})
        initial_class = initial_pred.get("output")
        
        for _ in range(n_queries):
            # Apply a random 'square' or block perturbation
            candidate = best_X.copy()
            idx = np.random.randint(0, X.shape[1])
            candidate[0, idx] += np.random.normal(0, 0.1)
            
            cand_pred = predict_fn({"features": candidate[0].tolist()})
            if cand_pred.get("output") != initial_class:
                return candidate
            
    except Exception:
        pass
    return best_X

def prompt_injection_suite(predict_fn: Callable, prompts_path: str = "ml_guard/sandbox/injection_payloads.txt") -> List[Dict[str, Any]]:
    """
    Fires known prompt injection payloads at the model.
    """
    # Fix path relative to repo root if needed
    if not os.path.exists(prompts_path):
        # Try local path
        prompts_path = os.path.join(os.path.dirname(__file__), "injection_payloads.txt")
        
    if not os.path.exists(prompts_path):
        return []

    with open(prompts_path, "r", encoding="utf-8") as f:
        payloads = [line.strip() for line in f.readlines() if line.strip()]

    results = []
    # Baseline check
    try:
        baseline = predict_fn({"prompt": "Hello"})
        baseline_out = baseline.get("output")
    except:
        baseline_out = None

    for payload in payloads[:50]:
        try:
            out_resp = predict_fn({"prompt": payload})
            out = out_resp.get("output")
            
            # Record if output changed significantly or matches success patterns
            results.append({
                "payload": payload,
                "output": out,
                "violated": out != baseline_out if baseline_out else True
            })
        except:
            continue
            
    return results
