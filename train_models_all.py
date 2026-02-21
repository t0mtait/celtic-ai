"""Train models on all data (original + uploaded games)."""
import kagglehub
import os
import pandas as pd
import numpy as np
import pickle
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download original dataset
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
away_games["celtics_win"] = (away_games["wl_away"] == "W")

feature_cols1 = ["fg_pct_home", "fg3_pct_home", "fta_home",
                 "oreb_home", "dreb_home", "stl_home",
                 "blk_home", "tov_home", "pf_home"]

feature_cols2 = ["fg_pct_away", "fg3_pct_away", "fta_away",
                 "oreb_away", "dreb_away", "stl_away",
                 "blk_away", "tov_away", "pf_away"]

# Create lists to track: features, targets, dates, opponents
home_features_list = [home_games[feature_cols1].reset_index(drop=True)]
home_targets_list = [home_games["celtics_win"].reset_index(drop=True)]
home_dates_list = [home_games["game_date"].reset_index(drop=True)]
home_opponents_list = [home_games["team_abbreviation_away"].reset_index(drop=True)]

away_features_list = [away_games[feature_cols2].reset_index(drop=True)]
away_targets_list = [away_games["celtics_win"].reset_index(drop=True)]
away_dates_list = [away_games["game_date"].reset_index(drop=True)]
away_opponents_list = [away_games["team_abbreviation_home"].reset_index(drop=True)]

# Load uploaded games if they exist
if os.path.exists("uploaded_games"):
    csv_files = glob.glob("uploaded_games/*.csv")
    if csv_files:
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Assume uploaded games have columns: location, date, opponent, fg_pct, fg3_pct, fta, oreb, dreb, stl, blk, tov, pf, actual_result
            if "location" in df.columns:
                home_df = df[df["location"] == "home"].copy()
                away_df = df[df["location"] == "away"].copy()
                
                if len(home_df) > 0:
                    home_df_subset = home_df[["fg_pct", "fg3_pct", "fta", "oreb", "dreb", "stl", "blk", "tov", "pf"]].copy()
                    home_df_subset.columns = feature_cols1
                    home_df_subset = home_df_subset.reset_index(drop=True)
                    
                    # Convert actual_result to boolean (1 or "W" means win)
                    home_wins = (home_df["actual_result"].astype(str).str.upper() == "W").values | (home_df["actual_result"] == 1).values
                    
                    home_features_list.append(home_df_subset)
                    home_targets_list.append(pd.Series(home_wins))
                    home_dates_list.append(pd.Series(home_df["date"].values))
                    home_opponents_list.append(pd.Series(home_df["opponent"].values))
                
                if len(away_df) > 0:
                    away_df_subset = away_df[["fg_pct", "fg3_pct", "fta", "oreb", "dreb", "stl", "blk", "tov", "pf"]].copy()
                    away_df_subset.columns = feature_cols2
                    away_df_subset = away_df_subset.reset_index(drop=True)
                    
                    # Convert actual_result to boolean (1 or "W" means win)
                    away_wins = (away_df["actual_result"].astype(str).str.upper() == "W").values | (away_df["actual_result"] == 1).values
                    
                    away_features_list.append(away_df_subset)
                    away_targets_list.append(pd.Series(away_wins))
                    away_dates_list.append(pd.Series(away_df["date"].values))
                    away_opponents_list.append(pd.Series(away_df["opponent"].values))

# Combine all data
subset1 = pd.concat(home_features_list, ignore_index=True)
y1 = pd.concat(home_targets_list, ignore_index=True)
home_dates = pd.concat(home_dates_list, ignore_index=True)
home_opponents = pd.concat(home_opponents_list, ignore_index=True)

subset2 = pd.concat(away_features_list, ignore_index=True)
y2 = pd.concat(away_targets_list, ignore_index=True)
away_dates = pd.concat(away_dates_list, ignore_index=True)
away_opponents = pd.concat(away_opponents_list, ignore_index=True)

# Filter out rows with missing values
mask1 = subset1.notna().all(axis=1)
mask2 = subset2.notna().all(axis=1)

subset1 = subset1[mask1].reset_index(drop=True)
y1 = y1[mask1].reset_index(drop=True)
home_dates = home_dates[mask1].reset_index(drop=True)
home_opponents = home_opponents[mask1].reset_index(drop=True)

subset2 = subset2[mask2].reset_index(drop=True)
y2 = y2[mask2].reset_index(drop=True)
away_dates = away_dates[mask2].reset_index(drop=True)
away_opponents = away_opponents[mask2].reset_index(drop=True)

print("Home:", subset1.shape, y1.shape)
print("Away:", subset2.shape, y2.shape)

# Train away model
X_train2, X_test2, y_train2, y_test2, indices_away_test = train_test_split(
    subset2, y2, np.arange(len(subset2)),
    test_size=0.2,
    random_state=42,
    stratify=y2
)

away_test_dates = away_dates.iloc[indices_away_test].reset_index(drop=True)
away_test_opponents = away_opponents.iloc[indices_away_test].reset_index(drop=True)

model_away = LogisticRegression(max_iter=1000)
model_away.fit(X_train2, y_train2)

y_pred2 = model_away.predict(X_test2)
print("\nAway accuracy:", accuracy_score(y_test2, y_pred2))
print(classification_report(y_test2, y_pred2))

# Train home model
X_train1, X_test1, y_train1, y_test1, indices_home_test = train_test_split(
    subset1, y1, np.arange(len(subset1)),
    test_size=0.2,
    random_state=42,
    stratify=y1
)

home_test_dates = home_dates.iloc[indices_home_test].reset_index(drop=True)
home_test_opponents = home_opponents.iloc[indices_home_test].reset_index(drop=True)

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

# Create DataFrames for game predictions with features
home_games_test = X_test1.copy()
home_games_test["location"] = "home"
home_games_test["date"] = home_test_dates.values
home_games_test["opponent"] = home_test_opponents.values
home_games_test["prediction"] = home_predictions
home_games_test["actual"] = y_test1.values
home_games_test["win_prob"] = home_probs[:, 1]
home_games_test["correct"] = home_predictions == y_test1.values

away_games_test = X_test2.copy()
away_games_test["location"] = "away"
away_games_test["date"] = away_test_dates.values
away_games_test["opponent"] = away_test_opponents.values
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
