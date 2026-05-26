# src/features.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class EloSystem:
    def __init__(self, default_elo=1500.0, k_factor=32):
        self.elo_dict = {}
        self.default_elo = default_elo
        self.k_factor = k_factor

    def get_elo(self, team):
        if team not in self.elo_dict:
            self.elo_dict[team] = self.default_elo
        return self.elo_dict[team]

    def update_elo(self, home_team, away_team, result):
        home_elo = self.get_elo(home_team)
        away_elo = self.get_elo(away_team)

        # Calculăm probabilitatea așteptată (Expected Outcome)
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home

        # Scorul real (Actual Outcome)
        if result == 'H':
            actual_home, actual_away = 1.0, 0.0
        elif result == 'A':
            actual_home, actual_away = 0.0, 1.0
        else: # Egalitate
            actual_home, actual_away = 0.5, 0.5

        # Actualizăm rating-urile în dicționar
        self.elo_dict[home_team] = home_elo + self.k_factor * (actual_home - expected_home)
        self.elo_dict[away_team] = away_elo + self.k_factor * (actual_away - expected_away)

def build_features(input_path, output_path):
    logging.info("Pornire generare caracteristici (Feature Engineering)...")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 1. Calculare ELO dinamic pas cu pas
    elo = EloSystem()
    home_elos = []
    away_elos = []

    for idx, row in df.iterrows():
        # Salvăm elo-ul CURENT (înainte de meci) pentru modelare
        home_elos.append(elo.get_elo(row['home_team']))
        away_elos.append(elo.get_elo(row['away_team']))
        
        # Actualizăm sistemul cu rezultatul meciului
        elo.update_elo(row['home_team'], row['away_team'], row['result'])

    df['home_elo'] = home_elos
    df['away_elo'] = away_elos
    df['elo_diff'] = df['home_elo'] - df['away_elo']

    # 2. Identificare meciuri pe teren neutru (0 = teren propriu, 1 = neutru)
    df['is_neutral'] = df['neutral'].astype(int)

    # Salvare dataset extins
    df.to_csv(output_path, index=False)
    logging.info(f"Done! Caracteristicile au fost salvate în: {output_path}")
    return df

if __name__ == "__main__":
    build_features(
        "data/processed/cleaned_matches.csv",
        "data/processed/features_matches.csv"
    )