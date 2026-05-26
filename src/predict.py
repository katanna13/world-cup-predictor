# src/predict.py
import pandas as pd
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO)

class MatchPredictor:
    def __init__(self, model_path="models/world_cup_xgb.pkl", encoder_path="models/label_encoder.pkl", features_path="data/processed/features_matches.csv"):
        # Încărcăm artefactele modelului
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        
        # Încărcăm ultimele ELO-uri calculate din setul de date ca să avem o bază de date cu valorile curente
        df = pd.read_csv(features_path)
        
        # Construim un dicționar cu cel mai recent ELO pentru fiecare echipă
        self.elo_database = {}
        for _, row in df.iterrows():
            self.elo_database[row['home_team']] = row['home_elo']
            self.elo_database[row['away_team']] = row['away_elo']
            
        logging.info(f"Predictor inițializat cu {len(self.elo_database)} echipe în baza de date ELO.")

    def get_team_elo(self, team):
        # Dacă echipa nu există în istoric, primește un ELO default de 1500
        return self.elo_database.get(team, 1500.0)

    def predict_match(self, home_team, away_team, is_neutral=1):
        """
        Prezice probabilitățile unui meci. 
        Implicit is_neutral=1 pentru că la Cupa Mondială se joacă pe teren neutru.
        """
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        elo_diff = home_elo - away_elo
        
        # Structura vectorului de caracteristici trebuie să fie identică cu cea de la antrenare
        # features = ['home_elo', 'away_elo', 'elo_diff', 'is_neutral']
        input_data = pd.DataFrame([{
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': elo_diff,
            'is_neutral': int(is_neutral)
        }])
        
        # Obținem probabilitățile de la XGBoost
        probabilities = self.model.predict_proba(input_data)[0]
        
        # Mapăm probabilitățile înapoi la etichetele lor originale ('A', 'D', 'H')
        result_dict = {}
        for idx, clasa in enumerate(self.encoder.classes_):
            label_mapping = {'H': home_team, 'D': 'Draw', 'A': away_team}
            result_dict[label_mapping[clasa]] = round(float(probabilities[idx]), 3)
            
        return result_dict

if __name__ == "__main__":
    # Testăm rapid sistemul de predicție
    predictor = MatchPredictor()
    
    meci_test = predictor.predict_match("France", "Romania")
    print(f"\n🔮 Predicție Franța vs. România (Teren Neutru):")
    print(meci_test)
    
    meci_test_2 = predictor.predict_match("Argentina", "Brazil")
    print(f"\n🔮 Predicție Argentina vs. Brazilia (Teren Neutru):")
    print(meci_test_2)