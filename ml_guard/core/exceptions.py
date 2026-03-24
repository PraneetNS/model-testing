class MLGuardException(Exception):
    """Base exception for ML Guard engine."""
    pass

class ModelValidationError(MLGuardException):
    """Raised when model predictions fail or model API is incompatible."""
    pass

class DataMismatchError(MLGuardException):
    """Raised when data dimensions, types, or shapes are incompatible."""
    pass

class SchemaError(MLGuardException):
    """Raised when feature schema differs between train and validation."""
    pass

class MetricComputationError(MLGuardException):
    """Raised when a metric cannot be mathematically computed."""
    pass
