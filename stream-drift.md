# 🌊 ML Guard: Stream Drift Deep Dive

The **Stream Drift** module is the "Live Telemetry" system of the ML Guard platform. It tracks how your model is behaving in real-time as new data flows through it.

---

## 1. Core Concept: What is actually happening?
In production, models often experience **"Silent Failure."** This happens when the data in the real world starts looking different from the data the model was trained on. 

ML Guard's Stream Drift engine does not just look at a single data point; it maintains a **Rolling Window** (typically the last 1000 events). Every time a new prediction arrives, the engine recalculates statistical distances to see if the "distribution" of your model has shifted.

---

## 2. Key Controls & Buttons

### 🆔 Model ID
*   **What is it?**: A unique name for your model's prediction stream (e.g., `fraud-detection-v2`).
*   **Where do I get it?**: 
    *   In a real company, this ID is generated when you register a model in the **Model Audit** tab. 
    *   For testing, you can **invent any name** you want. The system will automatically create a new "In-Memory Window" for any ID you type.
*   **Active Streams**: At the bottom left, you see a list of IDs. These are models currently receiving data. If you click one, it will switch your dashboard to view that model's specific charts.

### 🔌 Connect WebSocket
*   **Action**: Opens a persistent, bi-directional tunnel between your browser and the backend server.
*   **Effect**: This allows the server to "push" updates to your screen the millisecond drift is detected, without you having to refresh the page.

### 📐 Set Baseline
*   **Action**: Captures the *current* state of the window as the reference point.
*   **Effect**: Statistical metrics like **PSI (Population Stability Index)** compare "New Data" vs "Reference Data." If you don't set a baseline, the system has nothing to compare against, and your PSI will stay at `0.0`.
*   **Rule of Thumb**: You should "Set Baseline" when you know the model is performing correctly (e.g., right after deployment).

### 🧪 Send Test Event (UI Button)
*   **Action**: Sends a single, random prediction event to the current Model ID.
*   **Use Case**: Quick verification that the connection is working.

### 📡 Poll HTTP Status
*   **Action**: Manually asks the backend: *"Give me the latest numbers."*
*   **Use Case**: This is a backup for the WebSocket. If the WebSocket disconnects, you can still get the current drift score by clicking this.

---

## 3. The Dashboard Cards (The Metrics)

### 🪟 Window
The total number of predictions currently stored in the in-memory rolling buffer. As it hits its limit (e.g., 1000), the oldest events are dropped to make room for new ones.

### 📈 Rolling PSI (Population Stability Index)
**The most important metric.**
*   **0.0 - 0.1**: No Change (Green). Data looks exactly like the baseline.
*   **0.1 - 0.25**: Warning (Yellow). The distribution is shifting slightly.
*   **> 0.25**: **CRITICAL (Red)**. Massive drift. The model is likely making wrong predictions because the input data has fundamentally changed.

### 📉 Rolling JSD (Jensen-Shannon Divergence)
A secondary mathematical check. While PSI looks at "stability," JSD looks at how much the two distributions "overlap." High JSD means the new data is literally moving to a different region than the training data.

### 🎯 Confidence
The average "Certainty Score" your model is outputting. If your model usually has 90% confidence but suddenly drops to 60%, it means it's encountering things it doesn't recognize.

---

## 4. How the PowerShell Simulation Works

When you ran the script, here is what happened step-by-step:
1.  **Events 1-50**: The script sent predictions between `0.4` and `0.6`. Since this matched your baseline, the PSI stayed low (Green).
2.  **Events 51-100**: The script suddenly started sending predictions between `0.85` and `0.98`. 
3.  **The Engine Reacts**: The backend realized the "Shape" of the data shifted from a bell curve in the middle to a spike on the far right.
4.  **The Spikes**: You saw the lines on the chart jump up vertically. Once the score passed 0.25, the system triggered a **Critical Alert**.

---

## 5. Finding Your ID & Other Samples
*   **Listing IDs**: Use the `POLL HTTP STATUS` or look at the **Active Streams** list to see what IDs are currently "alive."
*   **New Samples**: You can create a second stream by simply changing the ID to `production-v2` and running the script again with a different model ID variable. You can then toggle between them in the UI to see independent drift metrics for each.
