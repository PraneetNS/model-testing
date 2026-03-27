import os
import time
import json
import asyncio
import numpy as np
import websockets
import logging
import asyncio
from typing import Dict, List, Any, Optional
from collections import deque
from ml_guard.core.drift import compute_psi

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [SENTINEL] %(levelname)s: %(message)s')
logger = logging.getLogger("mlguard-sentinel")

class MLGuardSentinel:
    """
    Real-Time Drift Sentinel Sidecar Agent.
    Intercepts inference features and streams PSI deltas to ML Guard.
    """
    def __init__(self):
        self.host = os.getenv("MLGUARD_HOST", "localhost:8000")
        self.model_id = os.getenv("MODEL_ID")
        self.sample_rate = float(os.getenv("SAMPLE_RATE", "0.1"))
        self.window_size = int(os.getenv("PSI_WINDOW_SIZE", "5000"))
        self.compute_every = int(os.getenv("COMPUTE_EVERY", "100"))
        self.baseline_path = os.getenv("BASELINE_PATH")
        
        if not self.model_id:
            raise ValueError("MODEL_ID environment variable is required.")

        # State
        self.feature_window = deque(maxlen=self.window_size)
        self.baseline_data: Optional[Dict[str, np.ndarray]] = None
        self.sample_count = 0
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.active = False

    async def load_baseline(self):
        """
        Load baseline distributions from MinIO or local path.
        Expected format: JSON mapping {feature_name: [values]}
        """
        logger.info(f"Loading baseline from {self.baseline_path}...")
        try:
            # For demonstration, we assume a JSON file. 
            # In production, this would use boto3 to fetch from MinIO.
            if os.path.exists(self.baseline_path):
                with open(self.baseline_path, 'r') as f:
                    data = json.load(f)
                    self.baseline_data = {k: np.array(v) for k, v in data.items()}
                logger.info("Baseline loaded successfully.")
            else:
                logger.warning("Baseline path not found. Initializing with empty baseline.")
                self.baseline_data = {}
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            self.baseline_data = {}

    def capture_inference(self, features: Dict[str, Any]):
        """
        Sync method to be called by the model proxy/wrapper.
        Samples features into the rolling window.
        """
        if np.random.random() > self.sample_rate:
            return

        self.feature_window.append(features)
        self.sample_count += 1
        
        # Every N samples, we trigger a PSI check if running in a background loop
        pass

    async def connect_backend(self):
        """Establish WebSocket connection with backpressure and reconnection logic."""
        uri = f"ws://{self.host}/api/v1/sentinel/stream/{self.model_id}"
        while self.active:
            try:
                async with websockets.connect(uri) as websocket:
                    logger.info(f"Connected to ML Guard at {uri}")
                    self.ws = websocket
                    while self.active:
                        # Computation Loop
                        if self.sample_count >= self.compute_every:
                            await self._compute_and_stream()
                            self.sample_count = 0
                        await asyncio.sleep(1) # Frequency of check
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}. Retrying in 5s...")
                await asyncio.sleep(5)

    async def _compute_and_stream(self):
        """Compute PSI for all features in window and stream to backend."""
        if not self.feature_window or not self.baseline_data:
            return

        try:
            # Convert window to DataFrame-like structure
            current_df = {k: [] for k in self.baseline_data.keys()}
            for f in list(self.feature_window):
                for k in current_df.keys():
                    if k in f:
                        current_df[k].append(f[k])

            # Compute PSI per feature
            psi_results = {}
            for feature, baseline_vals in self.baseline_data.items():
                actual_vals = np.array(current_df[feature])
                if len(actual_vals) > 10: # Minimum samples for stability
                    psi = compute_psi(baseline_vals, actual_vals)
                    psi_results[feature] = round(psi, 4)

            if not psi_results:
                return

            avg_psi = sum(psi_results.values()) / len(psi_results)
            
            payload = {
                "model_id": self.model_id,
                "timestamp": time.time(),
                "avg_psi": avg_psi,
                "feature_psi": psi_results,
                "window_size": len(self.feature_window)
            }

            if self.ws:
                await self.ws.send(json.dumps(payload))
                logger.debug(f"Sent PSI payload: {avg_psi:.4f}")

        except Exception as e:
            logger.error(f"Error in PSI computation/stream: {e}")

    async def start(self):
        self.active = True
        await self.load_baseline()
        await self.connect_backend()

    def stop(self):
        self.active = False
        logger.info("Sentinel stopping...")

# Example usage as a background task
# sentinel = MLGuardSentinel()
# asyncio.create_task(sentinel.start())
