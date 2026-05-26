# src/train.py
import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib  # Pentru salvarea modelului antrenat

logging.basicConfig(level=logging.INFO)

def train_prediction_model(input_path, model_output_path, encoder_output_path):
    logging.info("Pornire antrenare model XGBoost...")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Nu s-a gasit fisierul de caracteristici: {input_path}")

    # 1. Incarcare date
    df = pd.read_csv(input_path)

    # 2. Definire caracteristici (X) si tinta (y)
    features = ['home_elo', 'away_elo', 'elo_diff', 'is_neutral']
    X = df[features]
    y_raw = df['result']

    # 3. Codificare etichete string ('H', 'D', 'A') in numere (0, 1, 2)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    
    # Afisam clasele mapate pentru verificare
    for index, label in enumerate(encoder.classes_):
        logging.info(f"Clasa codificata {index}: {label}")

    # 4. Impartire in Train si Test split (cronologic sau random - mergem pe random pentru baseline stabil)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Initializare si antrenare XGBoost
    # Folosim objective='multi:softprob' pentru a obtine probabilitati pentru fiecare rezultat (Home, Draw, Away)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='multi:softprob',
        random_state=42
    )
    
    logging.info("Se antreneaza modelul pe setul de date...")
    model.fit(X_train, y_train)

    # 6. Evaluare model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"🔥 Acuratete model pe setul de test: {acc:.4f} ({acc * 100:.2f}%)")
    
    # Afisare raport detaliat de clasificare
    print("\nRaport detaliat de clasificare:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # 7. Salvare model si encoder pe disc (esential pentru src/predict.py si aplicatia Streamlit)
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    joblib.dump(encoder, encoder_output_path)
    
    logging.info(f"✅ Modelul a fost salvat cu succes in: {model_output_path}")
    logging.info(f"✅ LabelEncoder-ul a fost salvat cu succes in: {encoder_output_path}")

if __name__ == "__main__":
    train_prediction_model(
        "data/processed/features_matches.csv",
        "models/world_cup_xgb.pkl",
        "models/label_encoder.pkl"
    )