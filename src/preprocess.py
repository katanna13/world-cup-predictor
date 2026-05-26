import pandas as pd
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def clean_match_history(raw_path, output_path):
    logging.info("Pornire preprocesare date...")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Nu s-a gasit fisierul: {raw_path}")

    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.lower()

    required_cols = ['date', 'home_score', 'away_score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Lipsește coloana: {col}")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    df = df[df['date'].dt.year >= 2010].copy()
    df = df.dropna(subset=['home_score', 'away_score'])

    # Conversie explicită la int pentru scoruri, în caz că Pandas le-a citit ca float din cauza unor NaN-uri anterioare
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    df['result'] = np.where(
        df['home_score'] > df['away_score'], 'H',
        np.where(df['home_score'] < df['away_score'], 'A', 'D')
    )

    # PRO-TIP: Asigură-te că folderul de destinație există înainte de salvare
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    logging.info(f"Done! {len(df)} matches saved.")

    return df


if __name__ == "__main__":
    clean_match_history(
        "data/raw/matches.csv",
        "data/processed/cleaned_matches.csv"
    )