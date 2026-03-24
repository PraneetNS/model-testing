import logging
import json
import time

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(record.created)),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        return json.dumps(log_entry)

def setup_logger(name="ml-guard-sdk", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
