"""FastAPI backend for Celtics win prediction."""
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import os
import pandas as pd
import subprocess
from datetime import datetime
import io

app = FastAPI()

# Load models
with open("models/model_home.pkl", "rb") as f:
    model_home = pickle.load(f)

with open("models/model_away.pkl", "rb") as f:
    model_away = pickle.load(f)

with open("models/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Load game predictions
game_predictions = pd.read_csv("models/game_predictions.csv")


class PredictionRequest(BaseModel):
    location: str  # "home" or "away"
    fg_pct: float
    fg3_pct: float
    fta: float
    oreb: float
    dreb: float
    stl: float
    blk: float
    tov: float
    pf: float


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict Celtics win based on game stats."""
    features = [
        request.fg_pct,
        request.fg3_pct,
        request.fta,
        request.oreb,
        request.dreb,
        request.stl,
        request.blk,
        request.tov,
        request.pf
    ]
    
    if request.location.lower() == "home":
        model = model_home
    else:
        model = model_away
    
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    return {
        "location": request.location,
        "win_prediction": bool(prediction),
        "win_probability": float(probability[1]),
        "loss_probability": float(probability[0])
    }


@app.get("/game-stats")
async def game_stats():
    """Get game-by-game prediction statistics."""
    # Calculate overall stats
    home_games = game_predictions[game_predictions["location"] == "home"]
    away_games = game_predictions[game_predictions["location"] == "away"]
    
    overall_accuracy = (game_predictions["correct"].sum() / len(game_predictions) * 100)
    home_accuracy = (home_games["correct"].sum() / len(home_games) * 100) if len(home_games) > 0 else 0
    away_accuracy = (away_games["correct"].sum() / len(away_games) * 100) if len(away_games) > 0 else 0
    
    # Get recent games sorted by date (most recent first) and fill NaN values
    recent_games = game_predictions.copy()
    recent_games["date"] = pd.to_datetime(recent_games["date"])
    recent_games = recent_games.sort_values("date", ascending=False).head(20)
    recent_games = recent_games.fillna(0)  # Replace NaN with 0
    recent_games = recent_games.to_dict("records")
    
    return {
        "overall_accuracy": float(overall_accuracy),
        "home_accuracy": float(home_accuracy),
        "away_accuracy": float(away_accuracy),
        "total_games": len(game_predictions),
        "home_games_count": len(home_games),
        "away_games_count": len(away_games),
        "recent_games": recent_games
    }


@app.post("/upload-games")
async def upload_games(file: UploadFile = File(...)):
    """Upload a CSV file with games to add to training data."""
    try:
        # Create uploaded_games directory if it doesn't exist
        os.makedirs("uploaded_games", exist_ok=True)
        
        # Save uploaded file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uploaded_games/games_{timestamp}.csv"
        
        # Read and validate the CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode()))
        
        # Validate required columns
        required_cols = ["location", "date", "opponent", "fg_pct", "fg3_pct", "fta", 
                        "oreb", "dreb", "stl", "blk", "tov", "pf", "actual_result"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return {
                "success": False,
                "error": f"Missing required columns: {', '.join(missing_cols)}"
            }
        
        # Save the file
        with open(filename, "wb") as f:
            f.write(contents)
        
        # Retrain models with all data
        result = subprocess.run(
            ["/home/tom/Repositories/celtic-ai/.venv/bin/python", "train_models_all.py"],
            cwd="/home/tom/Repositories/celtic-ai",
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Model training failed: {result.stderr}"
            }
        
        # Reload models and predictions
        global model_home, model_away, game_predictions
        with open("models/model_home.pkl", "rb") as f:
            model_home = pickle.load(f)
        with open("models/model_away.pkl", "rb") as f:
            model_away = pickle.load(f)
        game_predictions = pd.read_csv("models/game_predictions.csv")
        
        return {
            "success": True,
            "message": f"Uploaded {len(df)} games and retrained model",
            "games_count": len(df)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
