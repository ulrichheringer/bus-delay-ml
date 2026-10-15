# bus-delay-ml

Predicts bus delays from route, time, weather, traffic, and passenger-load features.

## Data

Deterministic synthetic trip data is generated locally in `pipeline.py`. No private data or external download is required.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Evaluation results

With seed 42, the model was evaluated chronologically: 1,920 trips for training and the final 480 trips for testing. The test MAE was 2.3220 minutes, the RMSE was 2.9843 minutes, and R² was 0.6793. Traffic, rain, passenger load, and delays were simulated, not collected from a real transit system.

## Limitations

The generated trips cannot capture the complexity of a real transit network. Real use would require historical arrival data, route and schedule details, and evaluation across future service periods.
