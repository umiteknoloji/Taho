#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced GP Football Analysis System
Tüm eksik özelliklerin tamamlandığı gelişmiş analiz sistemi
"""

import json
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel, RationalQuadratic
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import warnings
import sqlite3
from datetime import datetime, timedelta
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class AdvancedGPPredictor:
    def __init__(self):
        self.gp_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.is_trained = False
        self.feature_names = []
        self.selected_feature_names = []
        self.prediction_history = []
        self.multi_league_models = {}
        self.init_database()
        
    def init_database(self):
        """Tahmin geçmişi için SQLite database başlat"""
        self.conn = sqlite3.connect('predictions.db', check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                week INTEGER,
                prediction TEXT,
                confidence REAL,
                actual_result TEXT,
                correct INTEGER,
                entropy_score REAL,
                risk_level TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                league TEXT,
                accuracy REAL,
                high_conf_accuracy REAL,
                total_predictions INTEGER,
                correct_predictions INTEGER
            )
        ''')
        self.conn.commit()
    
    def load_data_with_advanced_features(self, league_file='data/TR_stat.json'):
        """Gelişmiş özellikler ile veriyi yükle"""
        try:
            with open(league_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            matches = []
            team_stats = {}
            team_form = {}  # Son N maç formu
            h2h_history = {}  # Head-to-head geçmiş
            
            for match_idx, match in enumerate(data):
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
                    week = int(match.get('week', 0))
                    
                    # Pre-match features
                    parsed_match = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'week': week,
                        'attendance': match.get('attendance', 0) if match.get('attendance') else 0,
                        'result': result,
                        'match_index': match_idx
                    }
                    
                    # Historical features
                    home_history = team_stats.get(home_team, self._init_team_stats())
                    away_history = team_stats.get(away_team, self._init_team_stats())
                    
                    # Form analysis (son 5 maç)
                    home_form = team_form.get(home_team, [])
                    away_form = team_form.get(away_team, [])
                    
                    parsed_match.update(self._calculate_form_features(home_form, away_form))
                    parsed_match.update(self._calculate_historical_features(home_history, away_history))
                    
                    # Head-to-head features
                    h2h_key = f"{home_team}_vs_{away_team}"
                    h2h_rev_key = f"{away_team}_vs_{home_team}"
                    h2h_data = h2h_history.get(h2h_key, []) + h2h_history.get(h2h_rev_key, [])
                    parsed_match.update(self._calculate_h2h_features(h2h_data, home_team))
                    
                    # Ev sahibi avantajı
                    parsed_match.update(self._calculate_home_advantage(home_history))
                    
                    # Match stats (non-goal related)
                    stats = match.get('stats', {})
                    for stat_name, stat_data in stats.items():
                        if isinstance(stat_data, dict) and 'Goals' not in stat_name and 'Gol' not in stat_name:
                            home_val = self._parse_stat(stat_data.get('home', '0'))
                            away_val = self._parse_stat(stat_data.get('away', '0'))
                            
                            clean_name = stat_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                            parsed_match[f'home_{clean_name}'] = home_val
                            parsed_match[f'away_{clean_name}'] = away_val
                            parsed_match[f'diff_{clean_name}'] = home_val - away_val
                            
                            if away_val != 0:
                                parsed_match[f'ratio_{clean_name}'] = home_val / away_val
                            else:
                                parsed_match[f'ratio_{clean_name}'] = home_val if home_val > 0 else 1.0
                    
                    matches.append(parsed_match)
                    
                    # Update histories AFTER processing
                    self._update_team_stats(team_stats, home_team, away_team, home_score, away_score, result)
                    self._update_team_form(team_form, home_team, away_team, result)
                    self._update_h2h_history(h2h_history, home_team, away_team, result)
                    
                except Exception as e:
                    continue
            
            return pd.DataFrame(matches)
            
        except Exception as e:
            print(f"Data loading error: {e}")
            return pd.DataFrame()
    
    def _init_team_stats(self):
        """Takım istatistikleri başlangıç değerleri"""
        return {
            'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0, 'goal_diff': 0,
            'home_matches': 0, 'home_wins': 0, 'home_draws': 0,
            'away_matches': 0, 'away_wins': 0, 'away_draws': 0,
            'recent_form_points': 0, 'recent_form_matches': 0
        }
    
    def _calculate_form_features(self, home_form, away_form):
        """Son N maç form analizi"""
        features = {}
        
        # Son 5 maç formu
        for n in [3, 5, 10]:
            home_recent = home_form[-n:] if len(home_form) >= n else home_form
            away_recent = away_form[-n:] if len(away_form) >= n else away_form
            
            # Form puanı hesapla (3-1-0 sistemle)
            home_form_points = sum([3 if x == 'W' else 1 if x == 'D' else 0 for x in home_recent])
            away_form_points = sum([3 if x == 'W' else 1 if x == 'D' else 0 for x in away_recent])
            
            features[f'home_form_{n}'] = home_form_points / (n * 3) if n > 0 else 0
            features[f'away_form_{n}'] = away_form_points / (n * 3) if n > 0 else 0
            features[f'form_diff_{n}'] = features[f'home_form_{n}'] - features[f'away_form_{n}']
        
        return features
    
    def _calculate_historical_features(self, home_history, away_history):
        """Geçmiş performans özellikleri"""
        features = {}
        
        # Temel istatistikler
        for prefix, history in [('home', home_history), ('away', away_history)]:
            if history['matches'] > 0:
                features[f'{prefix}_win_rate'] = history['wins'] / history['matches']
                features[f'{prefix}_draw_rate'] = history['draws'] / history['matches']
                features[f'{prefix}_avg_gf'] = history['goals_for'] / history['matches']
                features[f'{prefix}_avg_ga'] = history['goals_against'] / history['matches']
                features[f'{prefix}_avg_gd'] = history['goal_diff'] / history['matches']
            else:
                features[f'{prefix}_win_rate'] = 0.33
                features[f'{prefix}_draw_rate'] = 0.27
                features[f'{prefix}_avg_gf'] = 1.35
                features[f'{prefix}_avg_ga'] = 1.35
                features[f'{prefix}_avg_gd'] = 0.0
            
            # Ev/Deplasman özel istatistikler
            if prefix == 'home' and history['home_matches'] > 0:
                features['home_home_win_rate'] = history['home_wins'] / history['home_matches']
                features['home_home_draw_rate'] = history['home_draws'] / history['home_matches']
            elif prefix == 'away' and history['away_matches'] > 0:
                features['away_away_win_rate'] = history['away_wins'] / history['away_matches']
                features['away_away_draw_rate'] = history['away_draws'] / history['away_matches']
        
        # Karşılaştırmalı özellikler
        features['win_rate_diff'] = features['home_win_rate'] - features['away_win_rate']
        features['gf_diff'] = features['home_avg_gf'] - features['away_avg_gf']
        features['ga_diff'] = features['away_avg_ga'] - features['home_avg_ga']  # Tersine çevir
        
        return features
    
    def _calculate_h2h_features(self, h2h_data, home_team):
        """Head-to-head özellikler"""
        features = {
            'h2h_matches': len(h2h_data),
            'h2h_home_wins': 0,
            'h2h_draws': 0,
            'h2h_away_wins': 0,
            'h2h_home_advantage': 0
        }
        
        if h2h_data:
            for result_info in h2h_data:
                if result_info['home'] == home_team:
                    if result_info['result'] == '1':
                        features['h2h_home_wins'] += 1
                    elif result_info['result'] == 'X':
                        features['h2h_draws'] += 1
                    else:
                        features['h2h_away_wins'] += 1
            
            # H2H win rate
            if features['h2h_matches'] > 0:
                features['h2h_home_win_rate'] = features['h2h_home_wins'] / features['h2h_matches']
                features['h2h_draw_rate'] = features['h2h_draws'] / features['h2h_matches']
        
        return features
    
    def _calculate_home_advantage(self, home_history):
        """Ev sahibi avantajı hesaplama"""
        features = {'home_advantage_factor': 0.1}  # Default
        
        if home_history['home_matches'] > 5:
            home_win_rate = home_history['home_wins'] / home_history['home_matches']
            overall_win_rate = home_history['wins'] / home_history['matches'] if home_history['matches'] > 0 else 0
            features['home_advantage_factor'] = max(0, home_win_rate - overall_win_rate)
        
        return features
    
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
    
    def _update_team_stats(self, team_stats, home_team, away_team, home_score, away_score, result):
        """Takım istatistiklerini güncelle"""
        for team in [home_team, away_team]:
            if team not in team_stats:
                team_stats[team] = self._init_team_stats()
        
        # Home team update
        team_stats[home_team]['matches'] += 1
        team_stats[home_team]['goals_for'] += home_score
        team_stats[home_team]['goals_against'] += away_score
        team_stats[home_team]['goal_diff'] += (home_score - away_score)
        team_stats[home_team]['home_matches'] += 1
        
        # Away team update
        team_stats[away_team]['matches'] += 1
        team_stats[away_team]['goals_for'] += away_score
        team_stats[away_team]['goals_against'] += home_score
        team_stats[away_team]['goal_diff'] += (away_score - home_score)
        team_stats[away_team]['away_matches'] += 1
        
        # Result updates
        if result == '1':
            team_stats[home_team]['wins'] += 1
            team_stats[home_team]['home_wins'] += 1
            team_stats[away_team]['losses'] += 1
        elif result == 'X':
            team_stats[home_team]['draws'] += 1
            team_stats[home_team]['home_draws'] += 1
            team_stats[away_team]['draws'] += 1
            team_stats[away_team]['away_draws'] += 1
        else:
            team_stats[home_team]['losses'] += 1
            team_stats[away_team]['wins'] += 1
            team_stats[away_team]['away_wins'] += 1
    
    def _update_team_form(self, team_form, home_team, away_team, result):
        """Takım form geçmişini güncelle"""
        for team in [home_team, away_team]:
            if team not in team_form:
                team_form[team] = []
        
        # Form letters: W-Win, D-Draw, L-Loss
        if result == '1':
            team_form[home_team].append('W')
            team_form[away_team].append('L')
        elif result == 'X':
            team_form[home_team].append('D')
            team_form[away_team].append('D')
        else:
            team_form[home_team].append('L')
            team_form[away_team].append('W')
        
        # Son 10 maçı tut
        for team in [home_team, away_team]:
            if len(team_form[team]) > 10:
                team_form[team] = team_form[team][-10:]
    
    def _update_h2h_history(self, h2h_history, home_team, away_team, result):
        """Head-to-head geçmişi güncelle"""
        h2h_key = f"{home_team}_vs_{away_team}"
        
        if h2h_key not in h2h_history:
            h2h_history[h2h_key] = []
        
        h2h_history[h2h_key].append({
            'home': home_team,
            'away': away_team,
            'result': result
        })
        
        # Son 10 karşılaşmayı tut
        if len(h2h_history[h2h_key]) > 10:
            h2h_history[h2h_key] = h2h_history[h2h_key][-10:]
    
    def train_with_kernel_optimization(self, league_file='data/TR_stat.json'):
        """Kernel optimizasyonu ile model eğitimi"""
        print("🚀 Gelişmiş GP Eğitimi Başlıyor...")
        
        # Load data with advanced features
        df = self.load_data_with_advanced_features(league_file)
        if len(df) < 10:
            return False, "Insufficient data"
        
        print(f"📊 {len(df)} maç, {df.shape[1]-4} özellik yüklendi")
        
        # Prepare features
        X = self.prepare_features(df)
        y = self.label_encoder.fit_transform(df['result'])
        
        self.feature_names = list(X.columns)
        
        # Advanced feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=min(25, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        self.feature_selector = selector
        
        # Get selected feature names and scores
        selected_indices = selector.get_support(indices=True)
        feature_scores = selector.scores_
        self.selected_feature_names = [
            (self.feature_names[i], feature_scores[i]) 
            for i in selected_indices
        ]
        self.selected_feature_names.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🎯 En iyi {len(self.selected_feature_names)} özellik seçildi:")
        for i, (name, score) in enumerate(self.selected_feature_names[:10], 1):
            print(f"   {i:2d}. {name:<30} ({score:.2f})")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Temporal split
        split_point = int(len(df) * 0.8)
        X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Kernel optimization
        print("⚙️ Kernel optimizasyonu...")
        kernels = [
            ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-5, 1e1)),
            ConstantKernel(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5) + WhiteKernel(0.1, (1e-5, 1e1)),
            ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(1.0, 0.1, (1e-5, 1e5)) + WhiteKernel(0.1, (1e-5, 1e1)),
            ConstantKernel(1.0) * (RBF(1.0) + Matern(1.0, nu=2.5)) + WhiteKernel(0.1),
        ]
        
        best_score = 0
        best_kernel = None
        
        for i, kernel in enumerate(kernels, 1):
            try:
                gp = GaussianProcessClassifier(kernel=kernel, random_state=42, max_iter_predict=100)
                scores = cross_val_score(gp, X_train, y_train, cv=3, scoring='accuracy')
                mean_score = scores.mean()
                
                print(f"   Kernel {i}: {mean_score:.3f} ± {scores.std():.3f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_kernel = kernel
                    
            except Exception as e:
                print(f"   Kernel {i}: Hata - {str(e)[:50]}")
                continue
        
        if best_kernel is None:
            best_kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
            print("   🔄 Varsayılan kernel kullanılıyor")
        
        print(f"🏆 En iyi CV skoru: {best_score:.3f}")
        
        # Train final model
        self.gp_model = GaussianProcessClassifier(kernel=best_kernel, random_state=42, max_iter_predict=100)
        self.gp_model.fit(X_train, y_train)
        self.is_trained = True
        
        # Test accuracy
        if len(X_test) > 0:
            y_pred = self.gp_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log performance
            self._log_model_performance(league_file, accuracy, len(y_test), np.sum(y_pred == y_test))
            
            return True, f"Model trained successfully. CV: {best_score:.1%}, Test: {accuracy:.1%}"
        else:
            return True, f"Model trained successfully. CV: {best_score:.1%}"
    
    def predict_with_risk_analysis(self, home_team, away_team, week=1, **kwargs):
        """Risk analizi ile tahmin"""
        if not self.is_trained:
            return None, 0.0, 0.0, "High", "Model not trained"
        
        try:
            # Create feature vector with ALL training features
            features = {}
            
            # Initialize with zeros
            for feature_name in self.feature_names:
                features[feature_name] = 0.0
            
            # Set provided features
            basic_features = {
                'week': week,
                'attendance': kwargs.get('attendance', 0),
                'home_win_rate': kwargs.get('home_win_rate', 0.33),
                'away_win_rate': kwargs.get('away_win_rate', 0.33),
                'home_avg_gf': kwargs.get('home_avg_gf', 1.35),
                'away_avg_gf': kwargs.get('away_avg_gf', 1.35),
                'home_avg_ga': kwargs.get('home_avg_ga', 1.35),
                'away_avg_ga': kwargs.get('away_avg_ga', 1.35),
                'home_form_5': kwargs.get('home_form_5', 0.5),
                'away_form_5': kwargs.get('away_form_5', 0.5),
                'h2h_matches': kwargs.get('h2h_matches', 0),
                'home_advantage_factor': kwargs.get('home_advantage_factor', 0.1)
            }
            
            # Update features
            for key, value in basic_features.items():
                if key in features:
                    features[key] = value
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            feature_df = feature_df[self.feature_names]
            
            # Apply feature selection and scaling
            X_selected = self.feature_selector.transform(feature_df)
            X_scaled = self.scaler.transform(X_selected)
            
            # Predict with probabilities
            probabilities = self.gp_model.predict_proba(X_scaled)[0]
            prediction = self.gp_model.predict(X_scaled)[0]
            confidence = np.max(probabilities)
            
            # Calculate entropy (uncertainty)
            entropy_score = entropy(probabilities)
            
            # Risk level assessment
            if confidence >= 0.7 and entropy_score < 0.8:
                risk_level = "Low"
            elif confidence >= 0.5 and entropy_score < 1.2:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            # Convert prediction back
            result = self.label_encoder.inverse_transform([prediction])[0]
            
            # Log prediction
            self._log_prediction(home_team, away_team, week, result, confidence, entropy_score, risk_level)
            
            return result, confidence, entropy_score, risk_level, "Success"
            
        except Exception as e:
            return None, 0.0, 0.0, "High", f"Prediction error: {str(e)}"
    
    def detect_upset_potential(self, home_team, away_team, **kwargs):
        """Sürpriz sonuç potansiyeli tespiti"""
        try:
            # Basic strength indicators
            home_strength = kwargs.get('home_win_rate', 0.33) * 0.4 + kwargs.get('home_form_5', 0.5) * 0.6
            away_strength = kwargs.get('away_win_rate', 0.33) * 0.4 + kwargs.get('away_form_5', 0.5) * 0.6
            
            strength_diff = abs(home_strength - away_strength)
            
            # Get prediction
            result, confidence, entropy_score, risk_level, status = self.predict_with_risk_analysis(
                home_team, away_team, **kwargs
            )
            
            if status != "Success":
                return False, 0.0, "Analysis failed"
            
            # Upset potential criteria
            upset_potential = False
            upset_score = 0.0
            
            # High entropy + low confidence = upset potential
            if entropy_score > 1.0 and confidence < 0.6:
                upset_potential = True
                upset_score = entropy_score * (1 - confidence)
            
            # Strong team predicted to lose/draw
            if strength_diff > 0.2:
                stronger_team = home_team if home_strength > away_strength else away_team
                if (stronger_team == home_team and result in ['X', '2']) or \
                   (stronger_team == away_team and result in ['X', '1']):
                    upset_potential = True
                    upset_score = max(upset_score, strength_diff * (1 - confidence))
            
            return upset_potential, upset_score, f"Entropy: {entropy_score:.2f}, Confidence: {confidence:.2f}"
            
        except Exception as e:
            return False, 0.0, f"Error: {str(e)}"
    
    def get_prediction_history(self, limit=50):
        """Tahmin geçmişini getir"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        
        return [dict(zip(columns, row)) for row in results]
    
    def get_performance_over_time(self, league=None):
        """Zaman içinde performans analizi"""
        cursor = self.conn.cursor()
        
        if league:
            cursor.execute('''
                SELECT * FROM model_performance 
                WHERE league = ?
                ORDER BY timestamp
            ''', (league,))
        else:
            cursor.execute('''
                SELECT * FROM model_performance 
                ORDER BY timestamp
            ''')
        
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        
        return [dict(zip(columns, row)) for row in results]
    
    def calculate_feature_importance_ranking(self):
        """Özellik önem sıralaması"""
        if not self.selected_feature_names:
            return []
        
        # Normalize scores
        scores = [score for _, score in self.selected_feature_names]
        max_score = max(scores) if scores else 1
        
        ranking = []
        for i, (name, score) in enumerate(self.selected_feature_names, 1):
            ranking.append({
                'rank': i,
                'feature': name,
                'score': score,
                'normalized_score': score / max_score,
                'importance_level': self._get_importance_level(score / max_score)
            })
        
        return ranking
    
    def _get_importance_level(self, normalized_score):
        """Önem seviyesi belirle"""
        if normalized_score >= 0.8:
            return "Critical"
        elif normalized_score >= 0.6:
            return "High"
        elif normalized_score >= 0.4:
            return "Medium"
        elif normalized_score >= 0.2:
            return "Low"
        else:
            return "Minimal"
    
    def _log_prediction(self, home_team, away_team, week, prediction, confidence, entropy_score, risk_level):
        """Tahmini veritabanına kaydet"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, league, home_team, away_team, week, prediction, confidence, entropy_score, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            "Unknown",
            home_team,
            away_team,
            week,
            prediction,
            confidence,
            entropy_score,
            risk_level
        ))
        self.conn.commit()
    
    def _log_model_performance(self, league, accuracy, total_predictions, correct_predictions):
        """Model performansını kaydet"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO model_performance 
            (timestamp, league, accuracy, total_predictions, correct_predictions)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            league,
            accuracy,
            total_predictions,
            correct_predictions
        ))
        self.conn.commit()
    
    def prepare_features(self, df):
        """Özellikleri hazırla"""
        feature_df = df.drop(['result', 'home_team', 'away_team', 'match_index'], axis=1)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        # Convert any remaining object types
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                feature_df[col] = pd.Categorical(feature_df[col]).codes
        
        return feature_df

# Test function
def test_advanced_gp():
    print("🚀 ADVANCED GP TEST")
    print("=" * 50)
    
    predictor = AdvancedGPPredictor()
    
    # Train with kernel optimization
    success, message = predictor.train_with_kernel_optimization('data/TR_stat.json')
    print(f"Training: {message}")
    
    if success:
        # Test prediction with risk analysis
        result, confidence, entropy_score, risk_level, status = predictor.predict_with_risk_analysis(
            "Galatasaray", "Fenerbahçe", 
            week=20,
            home_win_rate=0.7,
            away_win_rate=0.6,
            home_form_5=0.8,
            away_form_5=0.7
        )
        print(f"\n🎯 Test Prediction:")
        print(f"   Result: {result}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Entropy: {entropy_score:.3f}")
        print(f"   Risk Level: {risk_level}")
        
        # Test upset detection
        upset_potential, upset_score, upset_info = predictor.detect_upset_potential(
            "Galatasaray", "Fenerbahçe",
            home_win_rate=0.7, away_win_rate=0.6,
            home_form_5=0.8, away_form_5=0.7
        )
        print(f"\n🚨 Upset Analysis:")
        print(f"   Potential: {upset_potential}")
        print(f"   Score: {upset_score:.3f}")
        print(f"   Info: {upset_info}")
        
        # Feature importance
        importance = predictor.calculate_feature_importance_ranking()
        print(f"\n📊 Top 5 Features:")
        for feature in importance[:5]:
            print(f"   {feature['rank']}. {feature['feature']} ({feature['importance_level']})")
    
    return predictor

if __name__ == "__main__":
    test_advanced_gp()
