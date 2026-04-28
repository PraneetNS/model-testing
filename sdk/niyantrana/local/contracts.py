from typing import Optional
from ..models import ContractResult

class Contract:
    def __init__(self, name: str, confidence_min: Optional[float] = None, latency_max_ms: Optional[float] = None):
        self.name = name
        self.confidence_min = confidence_min
        self.latency_max_ms = latency_max_ms

    def validate(self, prediction: any, probability: float, latency_ms: float) -> ContractResult:
        breaches = []
        
        if self.confidence_min is not None and probability < self.confidence_min:
            breaches.append(f"Confidence {probability} is below minimum {self.confidence_min}")
            
        if self.latency_max_ms is not None and latency_ms > self.latency_max_ms:
            breaches.append(f"Latency {latency_ms}ms exceeds maximum {self.latency_max_ms}ms")
            
        return ContractResult(
            passed=len(breaches) == 0,
            breaches=breaches
        )
