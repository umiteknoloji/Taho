class ELORatingSystem:
    def __init__(self, initial_rating=1500, k_factor=32):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.team_ratings = {}
        
    def get_team_rating(self, team):
        if team not in self.team_ratings:
            self.team_ratings[team] = self.initial_rating
        return self.team_ratings[team]
    
    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def get_team_strength_features(self, home_team, away_team):
        home_rating = self.get_team_rating(home_team)
        away_rating = self.get_team_rating(away_team)
        
        return {
            'elo_home_rating': home_rating,
            'elo_away_rating': away_rating,
            'elo_rating_diff': home_rating - away_rating,
            'elo_home_win_prob': 0.5,
            'elo_draw_prob': 0.3,
            'elo_away_win_prob': 0.2,
            'elo_home_strength': home_rating / 1500,
            'elo_away_strength': away_rating / 1500,
            'elo_strength_ratio': home_rating / max(away_rating, 100)
        }
    
    def train_from_matches(self, matches_data):
        print("🔢 ELO rating sistemi eğitiliyor...")
        
        # Determinizm için maçları sırala
        sorted_matches = sorted(matches_data, key=lambda x: (
            x.get('week', 0), 
            x.get('date', ''), 
            x.get('home', ''), 
            x.get('away', '')
        ))
        
        processed = 0
        for match in sorted_matches:
            try:
                home_team = match.get('home', '')
                away_team = match.get('away', '')
                score = match.get('score', {})
                if not score:
                    continue
                full_time = score.get('fullTime', {})
                if not full_time:
                    continue
                home_score = int(full_time.get('home', 0))
                away_score = int(full_time.get('away', 0))
                self.update_ratings(home_team, away_team, home_score, away_score)
                processed += 1
            except:
                continue
        print(f"✅ {processed} maçtan ELO ratingler hesaplandı")
    
    def update_ratings(self, home_team, away_team, home_score, away_score):
        home_rating = self.get_team_rating(home_team)
        away_rating = self.get_team_rating(away_team)
        
        if home_score > away_score:
            home_actual = 1.0
        elif home_score == away_score:
            home_actual = 0.5
        else:
            home_actual = 0.0
        
        home_expected = self.expected_score(home_rating + 50, away_rating)
        
        home_new = home_rating + self.k_factor * (home_actual - home_expected)
        away_new = away_rating + self.k_factor * ((1 - home_actual) - (1 - home_expected))
        
        self.team_ratings[home_team] = home_new
        self.team_ratings[away_team] = away_new
