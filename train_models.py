"""Train and save the Celtics prediction models."""
import kagglehub
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download dataset
path = kagglehub.dataset_download("wyattowalsh/basketball")

game_path = os.path.join(path, "csv", "game.csv")
game_all = pd.read_csv(game_path)

# Filter for BOS games (Since 2000)
game_bos_home = game_all[game_all["team_abbreviation_home"] == "BOS"].copy()
game_bos_away = game_all[game_all["team_abbreviation_away"] == "BOS"].copy()

home_games = game_bos_home[game_bos_home["game_date"] >= "2000-01-01"].copy()
away_games = game_bos_away[game_bos_away["game_date"] >= "2000-01-01"].copy()

home_games = home_games.drop_duplicates()
away_games = away_games.drop_duplicates()

home_games["celtics_win"] = (home_games["wl_home"] == "W")
away_games["celtics_win"] = (away_games["wl_away"] == "W")  # Fixed: was "wl_home" == "L"

y1 = home_games["celtics_win"]
y2 = away_games["celtics_win"]

feature_cols1 = ["fg_pct_home", "fg3_pct_home", "fta_home",
                 "oreb_home", "dreb_home", "stl_home",
                 "blk_home", "tov_home", "pf_home"]

feature_cols2 = ["fg_pct_away", "fg3_pct_away", "fta_away",
                 "oreb_away", "dreb_away", "stl_away",
                 "blk_away", "tov_away", "pf_away"]

subset1 = home_games[feature_cols1]
subset2 = away_games[feature_cols2]

mask1 = subset1.notna().all(axis=1)
mask2 = subset2.notna().all(axis=1)

subset1 = subset1[mask1]
y1 = y1[mask1]

subset2 = subset2[mask2]
y2 = y2[mask2]

print("Home:", subset1.shape, y1.shape)
print("Away:", subset2.shape, y2.shape)

# Train away model
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    subset2, y2,
    test_size=0.2,
    random_state=42,
    stratify=y2
)

model_away = LogisticRegression(max_iter=1000)
model_away.fit(X_train2, y_train2)

y_pred2 = model_away.predict(X_test2)
print("\nAway accuracy:", accuracy_score(y_test2, y_pred2))
print(classification_report(y_test2, y_pred2))

# Train home model
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    subset1, y1,
    test_size=0.2,
    random_state=42,
    stratify=y1
)

model_home = LogisticRegression(max_iter=1000)
model_home.fit(X_train1, y_train1)

y_pred1 = model_home.predict(X_test1)
print("\nHome accuracy:", accuracy_score(y_test1, y_pred1))
print(classification_report(y_test1, y_pred1))

# Save models
os.makedirs("models", exist_ok=True)

with open("models/model_home.pkl", "wb") as f:
    pickle.dump(model_home, f)

with open("models/model_away.pkl", "wb") as f:
    pickle.dump(model_away, f)

# Save feature columns
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump({"home": feature_cols1, "away": feature_cols2}, f)

# Generate game-by-game predictions for display
home_predictions = model_home.predict(X_test1)
home_probs = model_home.predict_proba(X_test1)
away_predictions = model_away.predict(X_test2)
away_probs = model_away.predict_proba(X_test2)

# Get the game dates from original data
home_test_indices = X_test1.index
away_test_indices = X_test2.index

home_dates = home_games.loc[home_test_indices, "game_date"].values
away_dates = away_games.loc[away_test_indices, "game_date"].values
home_opponents = home_games.loc[home_test_indices, "team_abbreviation_away"].values
away_opponents = away_games.loc[away_test_indices, "team_abbreviation_home"].values

# Create DataFrames for game predictions with features
home_games_test = X_test1.copy()
home_games_test["location"] = "home"
home_games_test["date"] = home_dates
home_games_test["opponent"] = home_opponents
home_games_test["prediction"] = home_predictions
home_games_test["actual"] = y_test1.values
home_games_test["win_prob"] = home_probs[:, 1]
home_games_test["correct"] = home_predictions == y_test1.values

away_games_test = X_test2.copy()
away_games_test["location"] = "away"
away_games_test["date"] = away_dates
away_games_test["opponent"] = away_opponents
away_games_test["prediction"] = away_predictions
away_games_test["actual"] = y_test2.values
away_games_test["win_prob"] = away_probs[:, 1]
away_games_test["correct"] = away_predictions == y_test2.values

# Combine and save
all_games = pd.concat([home_games_test, away_games_test], ignore_index=True)
all_games.to_csv("models/game_predictions.csv", index=False)

print("\nModels saved to models/")
print(f"Game predictions saved: {len(all_games)} games")
print(f"Overall accuracy on test set: {(all_games['correct'].sum() / len(all_games) * 100):.2f}%")
