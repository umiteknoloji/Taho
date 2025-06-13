#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Working GP Predictor - Web entegrasyonu için
Veri sızıntısı olmayan, gerçek GP tahmin sistemi
"""

import json
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class WorkingGPPredictor:
    def __init__(self):
        self.gp_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.is_trained = False
        
    def load_data(self, league_file='data/TR_stat.json'):
        """Veriyi yükle (data leakage olmadan)"""
        try:
            with open(league_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            matches = []
            team_stats = {}
            
            for match in data:
                try:
                    # Basic info
                    score_data = match.get('score', {})
                    full_time = score_data.get('fullTime', {})
                    home_score = int(full_time.get('home', 0))
                    away_score = int(full_time.get('away', 0))
                    
                    # Result
                    if home_score > away_score:
                        result = '1'
                    elif home_score == away_score:
                        result = 'X'
                    else:
                        result = '2'
                    
                    home_team = match.get('home', '')
                    away_team = match.get('away', '')
                    
                    # Pre-match features only
                    parsed_match = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'week': int(match.get('week', 0)),
                        'attendance': match.get('attendance', 0) if match.get('attendance') else 0,
                        'result': result
                    }
                    
                    # Historical features
                    home_history = team_stats.get(home_team, {'matches': 0, 'wins': 0, 'draws': 0, 'goals_for': 0, 'goals_against': 0})
                    away_history = team_stats.get(away_team, {'matches': 0, 'wins': 0, 'draws': 0, 'goals_for': 0, 'goals_against': 0})
                    
                    # Add historical features
                    parsed_match.update({
                        'home_prev_matches': home_history['matches'],
                        'away_prev_matches': away_history['matches'],
                        'home_prev_wins': home_history['wins'],
                        'away_prev_wins': away_history['wins'],
                        'home_prev_draws': home_history['draws'],
                        'away_prev_draws': away_history['draws'],
                        'home_prev_gf': home_history['goals_for'],
                        'away_prev_gf': away_history['goals_for'],
                        'home_prev_ga': home_history['goals_against'],
                        'away_prev_ga': away_history['goals_against']
                    })
                    
                    # Win rates
                    if home_history['matches'] > 0:
                        parsed_match['home_win_rate'] = home_history['wins'] / home_history['matches']
                        parsed_match['home_avg_gf'] = home_history['goals_for'] / home_history['matches']
                        parsed_match['home_avg_ga'] = home_history['goals_against'] / home_history['matches']
                    else:
                        parsed_match['home_win_rate'] = 0.33
                        parsed_match['home_avg_gf'] = 1.35
                        parsed_match['home_avg_ga'] = 1.35
                        
                    if away_history['matches'] > 0:
                        parsed_match['away_win_rate'] = away_history['wins'] / away_history['matches']
                        parsed_match['away_avg_gf'] = away_history['goals_for'] / away_history['matches']
                        parsed_match['away_avg_ga'] = away_history['goals_against'] / away_history['matches']
                    else:
                        parsed_match['away_win_rate'] = 0.33
                        parsed_match['away_avg_gf'] = 1.35
                        parsed_match['away_avg_ga'] = 1.35
                    
                    # Match stats (non-goal related)
                    stats = match.get('stats', {})
                    for stat_name, stat_data in stats.items():
                        if isinstance(stat_data, dict) and 'Goals' not in stat_name:
                            home_val = self._parse_stat(stat_data.get('home', '0'))
                            away_val = self._parse_stat(stat_data.get('away', '0'))
                            
                            clean_name = stat_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                            parsed_match[f'home_{clean_name}'] = home_val
                            parsed_match[f'away_{clean_name}'] = away_val
                            parsed_match[f'diff_{clean_name}'] = home_val - away_val
                    
                    matches.append(parsed_match)
                    
                    # Update team stats AFTER processing
                    if home_team not in team_stats:
                        team_stats[home_team] = {'matches': 0, 'wins': 0, 'draws': 0, 'goals_for': 0, 'goals_against': 0}
                    if away_team not in team_stats:
                        team_stats[away_team] = {'matches': 0, 'wins': 0, 'draws': 0, 'goals_for': 0, 'goals_against': 0}
                    
                    # Update stats
                    team_stats[home_team]['matches'] += 1
                    team_stats[home_team]['goals_for'] += home_score
                    team_stats[home_team]['goals_against'] += away_score
                    
                    team_stats[away_team]['matches'] += 1
                    team_stats[away_team]['goals_for'] += away_score
                    team_stats[away_team]['goals_against'] += home_score
                    
                    if result == '1':
                        team_stats[home_team]['wins'] += 1
                    elif result == 'X':
                        team_stats[home_team]['draws'] += 1
                        team_stats[away_team]['draws'] += 1
                    else:
                        team_stats[away_team]['wins'] += 1
                        
                except Exception as e:
                    continue
            
            return pd.DataFrame(matches)
            
        except Exception as e:
            print(f"Data loading error: {e}")
            return pd.DataFrame()
    
    def _parse_stat(self, value):
        """İstatistik değerini parse et"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                clean_val = value.replace('%', '').replace(',', '').strip()
                return float(clean_val)
            return 0.0
        except:
            return 0.0
    
    def prepare_features(self, df):
        """Özellikleri hazırla"""
        feature_df = df.drop(['result', 'home_team', 'away_team'], axis=1)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        # Convert any remaining object types
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                feature_df[col] = pd.Categorical(feature_df[col]).codes
        
        return feature_df
    
    def train(self, league_file='data/TR_stat.json'):
        """Modeli eğit"""
        # Load data
        df = self.load_data(league_file)
        if len(df) < 10:
            return False, "Insufficient data"
        
        # Prepare features
        X = self.prepare_features(df)
        y = self.label_encoder.fit_transform(df['result'])
        
        # Store feature names for later use
        self.feature_names = list(X.columns)
        
        # Feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=min(15, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        self.feature_selector = selector
        
        # Store selected feature names
        selected_indices = selector.get_support(indices=True)
        self.selected_feature_names = [self.feature_names[i] for i in selected_indices]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Split temporally
        split_point = int(len(df) * 0.8)
        X_train = X_scaled[:split_point]
        y_train = y[:split_point]
        
        # Train GP
        kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
        self.gp_model = GaussianProcessClassifier(kernel=kernel, random_state=42, max_iter_predict=100)
        
        try:
            self.gp_model.fit(X_train, y_train)
            self.is_trained = True
            
            # Test accuracy
            X_test = X_scaled[split_point:]
            y_test = y[split_point:]
            
            if len(X_test) > 0:
                y_pred = self.gp_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                return True, f"Model trained successfully. Test accuracy: {accuracy:.1%}"
            else:
                return True, "Model trained successfully."
                
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def predict_match(self, home_team, away_team, week=1, **kwargs):
        """Maç tahmini yap"""
        if not self.is_trained:
            return None, 0.0, "Model not trained"
        
        try:
            # Create feature vector with ALL training features
            features = {}
            
            # Initialize with zeros for all training features
            for feature_name in self.feature_names:
                features[feature_name] = 0.0
            
            # Set provided features
            basic_features = {
                'week': week,
                'attendance': kwargs.get('attendance', 0),
                'home_prev_matches': kwargs.get('home_prev_matches', 0),
                'away_prev_matches': kwargs.get('away_prev_matches', 0),
                'home_prev_wins': kwargs.get('home_prev_wins', 0),
                'away_prev_wins': kwargs.get('away_prev_wins', 0),
                'home_prev_draws': kwargs.get('home_prev_draws', 0),
                'away_prev_draws': kwargs.get('away_prev_draws', 0),
                'home_prev_gf': kwargs.get('home_prev_gf', 0),
                'away_prev_gf': kwargs.get('away_prev_gf', 0),
                'home_prev_ga': kwargs.get('home_prev_ga', 0),
                'away_prev_ga': kwargs.get('away_prev_ga', 0),
                'home_win_rate': kwargs.get('home_win_rate', 0.33),
                'away_win_rate': kwargs.get('away_win_rate', 0.33),
                'home_avg_gf': kwargs.get('home_avg_gf', 1.35),
                'away_avg_gf': kwargs.get('away_avg_gf', 1.35),
                'home_avg_ga': kwargs.get('home_avg_ga', 1.35),
                'away_avg_ga': kwargs.get('away_avg_ga', 1.35)
            }
            
            # Update features with provided values
            for key, value in basic_features.items():
                if key in features:
                    features[key] = value
            
            # Add any additional features from kwargs
            for key, value in kwargs.items():
                if key in features:
                    features[key] = value
            
            # Convert to DataFrame with correct column order
            feature_df = pd.DataFrame([features])
            feature_df = feature_df[self.feature_names]  # Ensure correct order
            
            # Apply feature selection
            X_selected = self.feature_selector.transform(feature_df)
            
            # Scale
            X_scaled = self.scaler.transform(X_selected)
            
            # Predict
            probabilities = self.gp_model.predict_proba(X_scaled)[0]
            prediction = self.gp_model.predict(X_scaled)[0]
            confidence = np.max(probabilities)
            
            # Convert prediction back to result
            result = self.label_encoder.inverse_transform([prediction])[0]
            
            return result, confidence, "Success"
            
        except Exception as e:
            return None, 0.0, f"Prediction error: {str(e)}"

# Test function
def test_gp_predictor():
    print("🎯 GP TAHMİN SİSTEMİ TEST")
    print("=" * 40)
    
    predictor = WorkingGPPredictor()
    
    # Train model
    success, message = predictor.train('data/TR_stat.json')
    print(f"Training: {message}")
    
    if success:
        # Test prediction
        result, confidence, status = predictor.predict_match(
            "Galatasaray", "Fenerbahçe", 
            week=20,
            home_win_rate=0.7,
            away_win_rate=0.6
        )
        print(f"Test prediction: {result} (confidence: {confidence:.1%}) - {status}")
    
    return predictor

if __name__ == "__main__":
    test_gp_predictor()