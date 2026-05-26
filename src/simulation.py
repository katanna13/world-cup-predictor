# src/simulation.py
import sys
import os
import numpy as np
import pandas as pd
import logging
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import MatchPredictor

logging.basicConfig(level=logging.INFO)

class WorldCupSimulator:
    def __init__(self):
        self.predictor = MatchPredictor()
        # Definim giganții fotbalului mondial care au pedigree de campioni
        self.elita = ['Argentina', 'France', 'Spain', 'Brazil', 'Germany', 'England', 'Netherlands', 'Portugal', 'Uruguay']
        
    def simulate_match_outcome(self, team_a, team_b, can_draw=True):
        probs = self.predictor.predict_match(team_a, team_b, is_neutral=1)
        
        # --- CALIBRARE FINĂ PENTRU REALISM ---
        # Dacă o echipă din elită joacă cu una din afara elitei, ajustăm probabilitățile nerealiste
        if team_a in self.elita and team_b not in self.elita:
            probs[team_a] += 0.15
            probs[team_b] = max(0.02, probs[team_b] - 0.15)
        elif str(team_b) in self.elita and team_a not in self.elita:
            probs[team_b] += 0.15
            probs[team_a] = max(0.02, probs[team_a] - 0.15)
            
        teams = [team_a, 'Draw', team_b]
        prob_vector = [probs[team_a], probs['Draw'], probs[team_b]]
        prob_vector = np.array(prob_vector) / np.sum(prob_vector)
        
        choice = np.random.choice(teams, p=prob_vector)
        
        if not can_draw and choice == 'Draw':
            elo_a = self.predictor.get_team_elo(team_a)
            elo_b = self.predictor.get_team_elo(team_b)
            
            # Bonus de penalty-uri/experiență pentru echipele mari în faze eliminatorii
            weight_a = elo_a * (1.15 if team_a in self.elita else 1.0)
            weight_b = elo_b * (1.15 if team_b in self.elita else 1.0)
            
            p_advance_a = weight_a / (weight_a + weight_b)
            return np.random.choice([team_a, team_b], p=[p_advance_a, 1 - p_advance_a])
            
        return choice

    def simulate_group(self, teams):
        points = {team: 0 for team in teams}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                res = self.simulate_match_outcome(t1, t2, can_draw=True)
                if res == t1: points[t1] += 3
                elif res == t2: points[t2] += 3
                else: points[t1] += 1; points[t2] += 1
                    
        sorted_teams = sorted(points.items(), key=lambda x: x[1], reverse=True)
        return [sorted_teams[0][0], sorted_teams[1][0]]

    def simulate_tournament(self, groups):
        locuri_1, locuri_2, locuri_3 = [], [], []
        
        for group_name, group_teams in sorted(groups.items()):
            points = {team: 0 for team in group_teams}
            for i in range(len(group_teams)):
                for j in range(i + 1, len(group_teams)):
                    t1, t2 = group_teams[i], group_teams[j]
                    res = self.simulate_match_outcome(t1, t2, can_draw=True)
                    if res == t1: points[t1] += 3
                    elif res == t2: points[t2] += 3
                    else: points[t1] += 1; points[t2] += 1
            
            sorted_group = [item[0] for item in sorted(points.items(), key=lambda x: x[1], reverse=True)]
            locuri_1.append(sorted_group[0])
            locuri_2.append(sorted_group[1])
            locuri_3.append(sorted_group[2])
            
        current_round = locuri_1 + locuri_2 + locuri_3[:8]
        np.random.shuffle(current_round)
        
        while len(current_round) > 1:
            next_round = []
            for i in range(0, len(current_round), 2):
                winner = self.simulate_match_outcome(current_round[i], current_round[i+1], can_draw=False)
                next_round.append(winner)
            current_round = next_round
            
        return current_round[0]

    def run_monte_carlo(self, groups, iterations=1000):
        logging.info(f"Pornire simulare Monte Carlo ({iterations} rulări)...")
        winners = []
        for i in range(iterations):
            winner = self.simulate_tournament(groups)
            winners.append(winner)
            
        counter = Counter(winners)
        results_df = pd.DataFrame(counter.items(), columns=['Team', 'Wins'])
        results_df['Probability (%)'] = (results_df['Wins'] / iterations) * 100
        results_df = results_df.sort_values(by='Probability (%)', ascending=False).reset_index(drop=True)
        return results_df

if __name__ == "__main__":
    path_echipe = "data/raw/teams_2026.csv"
    if os.path.exists(path_echipe):
        teams_df = pd.read_csv(path_echipe, names=['team', 'elo', 'group'])
        world_cup_groups = {}
        for group_name, group_data in teams_df.groupby('group'):
            world_cup_groups[f"Group {group_name}"] = group_data['team'].tolist()
        sim = WorldCupSimulator()
        prospects = sim.run_monte_carlo(world_cup_groups, iterations=1000)
        print(prospects.head(10))