# algae_sensor_cnn

🌱 Algae Health Monitoring using CNNs on Multivariate Sensor Data 📌 Project Overview

This project implements an end-to-end machine learning pipeline to monitor algae system health using multivariate time-series sensor data. The system identifies unhealthy operating conditions early and prioritizes high-risk periods for intervention.

🎯 Business Problem
Industrial algae cultivation systems generate continuous sensor data (oxygen, PAM fluorescence, density). Manual monitoring is inefficient, and failures are often detected too late.

Goal: Automatically detect unhealthy system behavior while minimizing missed failures.

🧠 Solution Approach
Converted raw sensor streams into sliding time windows

Labeled windows using domain-informed voting logic

Trained a 1D CNN to learn temporal health patterns

Evaluated using imbalance-aware, deployment-safe metrics

🔍 Key Features
Sliding-window time-series modeling

Robust preprocessing for noisy sensor data

CNN-based temporal feature learning

Risk-based evaluation (Recall@Top-K)

Overfitting-safe validation strategy

📊 Evaluation Metrics
PR-AUC (Primary): Measures unhealthy detection quality under imbalance

ROC-AUC (Secondary): Overall separability

Recall@Top-K: Operational usefulness when resources are limited

High Recall@Top-K means most unhealthy states are detected by inspecting only a small fraction of time windows.

🛠 Tech Stack

Python

NumPy, Pandas

Scikit-learn

TensorFlow / Keras

CNNs for time-series

🚀 Use Cases

Algae cultivation monitoring

Industrial sensor health detection

Early warning systems

Predictive maintenance pipelines

Challenges faced (and how they were handled)
⚠ Challenge 1: Severe class imbalance

Healthy dominates unhealthy

Accuracy and ROC-AUC misleading

✅ Solution:

PR-AUC

Recall-focused evaluation

Risk-based metrics

⚠ Challenge 2: Sliding-window leakage

Overlapping windows inflate metrics

✅ Solution:

No threshold tuning

Conservative evaluation

Emphasis on ranking quality, not absolute scores

⚠ Challenge 3: Sensor noise & NaNs

Density and PAM signals noisy

Scaling caused NaNs initially

✅ Solution:

RobustScaler

Explicit NaN checks

Window-level aggregation

⚠ Challenge 4: Day-night bias

Oxygen naturally oscillates

Static thresholds fail

✅ Solution:

Global mean comparison

Window-based voting

CNN learns temporal context

✅ Key Takeaway
The model is not optimized for perfect accuracy but for early and reliable detection of unhealthy states under real operational constraints.

