# Celtic AI - Win Prediction Model

ML-powered Celtics game win predictor using logistic regression on four factors data.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000 in your browser.

## Project Structure

```
celtic-ai/
├── app.py              # FastAPI backend (serve this)
├── data_loader.py      # CSV data loading utilities
├── index.html          # Web UI
├── train_models.py     # Train/update prediction models
├── data/               # Game data CSVs (2017-2026)
├── models/             # Trained models and predictions
│   ├── model_home.pkl
│   ├── model_away.pkl
│   ├── feature_cols.pkl
│   └── game_predictions.csv
├── Dockerfile
└── docker-compose.yml
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/predict` | POST | Predict win from game stats |
| `/game-stats` | GET | Model accuracy and recent games |

### `/predict` Request Body

```json
{
  "location": "home" | "away",
  "pace": 98.5,
  "ftr": 0.22,
  "efg_pct": 0.54,
  "tov_pct": 0.13,
  "orb_pct": 0.25
}
```

## Training

To retrain models with updated data:

```bash
python train_models.py
```

## Docker

```bash
docker-compose up -d
```

Access at http://localhost:8000
