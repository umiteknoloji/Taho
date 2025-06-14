#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced No H2H GP Predictor with Dynamic Thresholds
Tüm statik threshold değerleri gerçek veri analiziyle dinamik olarak hesaplanır
"""

import json
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from dynamic_threshold_calculator import DynamicThresholdCalculator
import warnings
warnings.filterwarnings('ignore')

class EnhancedNoH2HGP:
    def __init__(self):
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.is_trained = False
        self.feature_names = []
        self.team_stats = {}
        self.threshold_calculator = DynamicThresholdCalculator()
        self.dynamic_thresholds = {}
        
    def load_and_analyze_data(self, league_file, target_week):
        """Veriyi yükle ve dynamic threshold'ları hesapla"""
        with open(league_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 {league_file} yüklendi: {len(data)} maç")
        
        # Dynamic thresholds hesapla
        self.threshold_calculator.load_data(league_file, target_week)
        self.dynamic_thresholds = self.threshold_calculator.calculate_all_thresholds()
        
        print("🎯 Dynamic Thresholds Uygulandı:")
        for key, value in self.dynamic_thresholds.items():
            print(f"   {key}: {value}")
        
        # Team statistics hesapla
        self.calculate_team_statistics(data, target_week)
        
        # Training data (target week öncesi)
        train_data = [match for match in data if int(match.get('week', 0)) < target_week]
        test_data = [match for match in data if int(match.get('week', 0)) == target_week]
        
        print(f"🎯 Eğitim: {len(train_data)} maç, Test: {len(test_data)} maç")
        return train_data, test_data
    
    def calculate_team_statistics(self, data, target_week):
        """Takım istatistiklerini hesapla - Dynamic Temporal Weights ile"""
        team_stats = {}
        
        # Chronological sorting by week
        sorted_data = sorted(data, key=lambda x: int(x.get('week', 0)))
        
        for match in sorted_data:
            week = int(match.get('week', 0))
            if week >= target_week:
                continue
                
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            # Score
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            # Initialize team stats
            for team in [home_team, away_team]:
                if team not in team_stats:
                    team_stats[team] = {
                        'total_matches': 0,
                        'total_wins': 0, 'total_draws': 0, 'total_losses': 0,
                        'total_goals_for': 0, 'total_goals_against': 0,
                        
                        # HOME SPECIFIC
                        'home_matches': 0,
                        'home_wins': 0, 'home_draws': 0, 'home_losses': 0,
                        'home_goals_for': 0, 'home_goals_against': 0,
                        
                        # AWAY SPECIFIC  
                        'away_matches': 0,
                        'away_wins': 0, 'away_draws': 0, 'away_losses': 0,
                        'away_goals_for': 0, 'away_goals_against': 0,
                        
                        # RECENT FORM (Last 5 matches - chronological)
                        'last_5_results': [],  # [3,1,0,3,1] -> [W,D,L,W,D]
                        'last_5_home_results': [],
                        'last_5_away_results': [],
                        
                        # TEMPORAL WEIGHTING (Recent matches matter more)
                        'weighted_form': 0.0,
                        'weighted_home_form': 0.0, 
                        'weighted_away_form': 0.0,
                        
                        # MOMENTUM (Form trend)
                        'momentum': 0.0,
                        'home_momentum': 0.0,
                        'away_momentum': 0.0
                    }
            
            # Update statistics (same as before)
            # ... [Previous update logic remains the same]
            
            # Update home team
            team_stats[home_team]['total_matches'] += 1
            team_stats[home_team]['home_matches'] += 1
            team_stats[home_team]['total_goals_for'] += home_score
            team_stats[home_team]['total_goals_against'] += away_score
            team_stats[home_team]['home_goals_for'] += home_score
            team_stats[home_team]['home_goals_against'] += away_score
            
            # Update away team
            team_stats[away_team]['total_matches'] += 1
            team_stats[away_team]['away_matches'] += 1
            team_stats[away_team]['total_goals_for'] += away_score
            team_stats[away_team]['total_goals_against'] += home_score
            team_stats[away_team]['away_goals_for'] += away_score
            team_stats[away_team]['away_goals_against'] += home_score
            
            # RESULT ANALYSIS
            if home_score > away_score:
                # HOME WIN
                team_stats[home_team]['total_wins'] += 1
                team_stats[home_team]['home_wins'] += 1
                team_stats[away_team]['total_losses'] += 1
                team_stats[away_team]['away_losses'] += 1
                
                # Form updates
                team_stats[home_team]['last_5_results'].append(3)  # Win = 3 points
                team_stats[home_team]['last_5_home_results'].append(3)
                team_stats[away_team]['last_5_results'].append(0)  # Loss = 0 points
                team_stats[away_team]['last_5_away_results'].append(0)
                
            elif home_score == away_score:
                # DRAW
                team_stats[home_team]['total_draws'] += 1
                team_stats[home_team]['home_draws'] += 1
                team_stats[away_team]['total_draws'] += 1
                team_stats[away_team]['away_draws'] += 1
                
                # Form updates
                team_stats[home_team]['last_5_results'].append(1)  # Draw = 1 point
                team_stats[home_team]['last_5_home_results'].append(1)
                team_stats[away_team]['last_5_results'].append(1)
                team_stats[away_team]['last_5_away_results'].append(1)
                
            else:
                # AWAY WIN
                team_stats[away_team]['total_wins'] += 1
                team_stats[away_team]['away_wins'] += 1
                team_stats[home_team]['total_losses'] += 1
                team_stats[home_team]['home_losses'] += 1
                
                # Form updates
                team_stats[away_team]['last_5_results'].append(3)
                team_stats[away_team]['last_5_away_results'].append(3)
                team_stats[home_team]['last_5_results'].append(0)
                team_stats[home_team]['last_5_home_results'].append(0)
            
            # Keep only last 5 matches
            for team in [home_team, away_team]:
                team_stats[team]['last_5_results'] = team_stats[team]['last_5_results'][-5:]
                team_stats[team]['last_5_home_results'] = team_stats[team]['last_5_home_results'][-5:]
                team_stats[team]['last_5_away_results'] = team_stats[team]['last_5_away_results'][-5:]
        
        # Calculate weighted forms and momentum with DYNAMIC weights
        self._calculate_advanced_metrics_dynamic(team_stats)
        
        self.team_stats = team_stats
        print(f"📈 {len(team_stats)} takım istatistiği hesaplandı (Dynamic Thresholds)")
    
    def _calculate_advanced_metrics_dynamic(self, team_stats):
        """Gelişmiş metrikler hesapla - DYNAMIC TEMPORAL WEIGHTS ile"""
        # Dynamic temporal weights al
        weights = self.dynamic_thresholds.get('temporal_weights', [0.35, 0.25, 0.20, 0.15, 0.05])
        
        for team, stats in team_stats.items():
            # WEIGHTED FORM (Dynamic weights ile)
            if stats['last_5_results']:
                results = stats['last_5_results']
                
                # Dynamic weights kullan
                weighted_sum = sum(w * r for w, r in zip(weights, reversed(results)))
                max_possible = sum(w * 3 for w in weights)  # Max = all wins
                stats['weighted_form'] = weighted_sum / max_possible if max_possible > 0 else 0
            
            # HOME WEIGHTED FORM
            if stats['last_5_home_results']:
                home_weights = weights[:min(len(weights), len(stats['last_5_home_results']))]
                results = stats['last_5_home_results']
                
                weighted_sum = sum(w * r for w, r in zip(home_weights, reversed(results)))
                max_possible = sum(w * 3 for w in home_weights)
                stats['weighted_home_form'] = weighted_sum / max_possible if max_possible > 0 else 0
            
            # AWAY WEIGHTED FORM
            if stats['last_5_away_results']:
                away_weights = weights[:min(len(weights), len(stats['last_5_away_results']))]
                results = stats['last_5_away_results']
                
                weighted_sum = sum(w * r for w, r in zip(away_weights, reversed(results)))
                max_possible = sum(w * 3 for w in away_weights)
                stats['weighted_away_form'] = weighted_sum / max_possible if max_possible > 0 else 0
            
            # MOMENTUM hesaplamaları aynı kalıyor
            if len(stats['last_5_results']) >= 3:
                recent_3 = stats['last_5_results'][-3:]
                older_2 = stats['last_5_results'][-5:-3] if len(stats['last_5_results']) >= 5 else []
                
                recent_avg = sum(recent_3) / len(recent_3) if recent_3 else 0
                older_avg = sum(older_2) / len(older_2) if older_2 else recent_avg
                
                stats['momentum'] = recent_avg - older_avg
            
            # HOME/AWAY MOMENTUM aynı kalıyor
            if len(stats['last_5_home_results']) >= 2:
                recent_home = stats['last_5_home_results'][-2:]
                older_home = stats['last_5_home_results'][-4:-2] if len(stats['last_5_home_results']) >= 4 else []
                
                recent_avg = sum(recent_home) / len(recent_home)
                older_avg = sum(older_home) / len(older_home) if older_home else recent_avg
                
                stats['home_momentum'] = recent_avg - older_avg
            
            if len(stats['last_5_away_results']) >= 2:
                recent_away = stats['last_5_away_results'][-2:]
                older_away = stats['last_5_away_results'][-4:-2] if len(stats['last_5_away_results']) >= 4 else []
                
                recent_avg = sum(recent_away) / len(recent_away)
                older_avg = sum(older_away) / len(older_away) if older_away else recent_avg
                
                stats['away_momentum'] = recent_avg - older_avg
    
    def create_pure_form_features_dynamic(self, match_data):
        """Dynamic threshold'larla feature'lar oluştur"""
        home_team = match_data.get('home', '')
        away_team = match_data.get('away', '')
        
        home_stats = self.team_stats.get(home_team, {})
        away_stats = self.team_stats.get(away_team, {})
        
        features = {}
        
        # Dynamic threshold'ları al
        defensive_factor = self.dynamic_thresholds.get('defensive_factor', 0.5)
        balance_threshold = self.dynamic_thresholds.get('balance_threshold', 0.1)
        home_advantage = self.dynamic_thresholds.get('home_advantage', 0.4)
        desperation_threshold = self.dynamic_thresholds.get('desperation_threshold', 1.0)
        upset_threshold = self.dynamic_thresholds.get('upset_threshold', -0.5)
        
        # VENUE-SPECIFIC PERFORMANCE (Aynı kalıyor)
        home_home_win_rate = home_stats.get('home_wins', 0) / max(home_stats.get('home_matches', 1), 1)
        home_home_draw_rate = home_stats.get('home_draws', 0) / max(home_stats.get('home_matches', 1), 1)
        home_home_points_per_match = (home_stats.get('home_wins', 0) * 3 + home_stats.get('home_draws', 0)) / max(home_stats.get('home_matches', 1), 1)
        
        away_away_win_rate = away_stats.get('away_wins', 0) / max(away_stats.get('away_matches', 1), 1)
        away_away_draw_rate = away_stats.get('away_draws', 0) / max(away_stats.get('away_matches', 1), 1)
        away_away_points_per_match = (away_stats.get('away_wins', 0) * 3 + away_stats.get('away_draws', 0)) / max(away_stats.get('away_matches', 1), 1)
        
        # VENUE-SPECIFIC MAIN FEATURES
        features['home_venue_strength'] = home_home_points_per_match
        features['away_venue_strength'] = away_away_points_per_match
        features['venue_strength_diff'] = home_home_points_per_match - away_away_points_per_match
        
        # WIN RATE COMPARISON (VENUE-SPECIFIC)
        features['home_venue_win_rate'] = home_home_win_rate
        features['away_venue_win_rate'] = away_away_win_rate
        features['venue_win_rate_diff'] = home_home_win_rate - away_away_win_rate
        
        # DRAW TENDENCY (VENUE-SPECIFIC)
        features['home_venue_draw_rate'] = home_home_draw_rate
        features['away_venue_draw_rate'] = away_away_draw_rate
        features['venue_draw_tendency'] = (home_home_draw_rate + away_away_draw_rate) / 2
        
        # VENUE-SPECIFIC RECENT FORM (Dynamic Weighted)
        features['home_venue_form'] = home_stats.get('weighted_home_form', 0.33)
        features['away_venue_form'] = away_stats.get('weighted_away_form', 0.33)
        features['venue_form_diff'] = features['home_venue_form'] - features['away_venue_form']
        
        # VENUE-SPECIFIC MOMENTUM
        features['home_venue_momentum'] = home_stats.get('home_momentum', 0)
        features['away_venue_momentum'] = away_stats.get('away_momentum', 0)
        features['venue_momentum_diff'] = features['home_venue_momentum'] - features['away_venue_momentum']
        
        # VENUE-SPECIFIC GOAL PERFORMANCE (DYNAMIC DEFENSIVE FACTOR)
        home_home_goals_per_match = home_stats.get('home_goals_for', 0) / max(home_stats.get('home_matches', 1), 1)
        home_home_conceded_per_match = home_stats.get('home_goals_against', 0) / max(home_stats.get('home_matches', 1), 1)
        away_away_goals_per_match = away_stats.get('away_goals_for', 0) / max(away_stats.get('away_matches', 1), 1)
        away_away_conceded_per_match = away_stats.get('away_goals_against', 0) / max(away_stats.get('away_matches', 1), 1)
        
        features['home_venue_attack'] = home_home_goals_per_match
        features['away_venue_attack'] = away_away_goals_per_match
        features['venue_attack_diff'] = home_home_goals_per_match - away_away_goals_per_match
        
        # DYNAMIC DEFENSIVE FACTOR kullan
        features['home_venue_defense'] = 1.0 / (home_home_conceded_per_match + defensive_factor)
        features['away_venue_defense'] = 1.0 / (away_away_conceded_per_match + defensive_factor)
        features['venue_defense_diff'] = features['home_venue_defense'] - features['away_venue_defense']
        
        # SECONDARY: GENEL FORM
        features['home_total_form'] = home_stats.get('weighted_form', 0.33)
        features['away_total_form'] = away_stats.get('weighted_form', 0.33)
        features['total_form_diff'] = features['home_total_form'] - features['away_total_form']
        
        # BALANCE INDICATORS (DYNAMIC BALANCE THRESHOLD)
        features['venue_strength_balance'] = 1.0 / (abs(features['venue_strength_diff']) + balance_threshold)
        features['venue_form_balance'] = 1.0 / (abs(features['venue_form_diff']) + balance_threshold)
        features['venue_attack_balance'] = 1.0 / (abs(features['venue_attack_diff']) + balance_threshold)
        
        # COMBINED VENUE INDICATOR
        venue_weight = 0.5 * features['venue_strength_diff'] + 0.3 * features['venue_form_diff'] + 0.2 * features['venue_momentum_diff']
        features['master_venue_indicator'] = venue_weight
        
        # DYNAMIC HOME ADVANTAGE
        features['home_advantage_factor'] = home_advantage * (1 + features['home_venue_strength'] - 1.5)
        
        # ACCURACY IMPROVEMENT FEATURES (DYNAMIC THRESHOLDS)
        
        # 1. DYNAMIC DESPERATION FACTOR
        home_desperation = 0.0
        if features['home_venue_strength'] < desperation_threshold:
            home_desperation = (desperation_threshold - features['home_venue_strength']) * 0.3
        features['home_desperation_factor'] = home_desperation
        
        # 2. DEFENSIVE VULNERABILITY (DYNAMIC)
        home_vulnerability = 1.0 / (features['home_venue_defense'] + balance_threshold)
        away_vulnerability = 1.0 / (features['away_venue_defense'] + balance_threshold)
        features['combined_vulnerability'] = (home_vulnerability + away_vulnerability) / 2
        
        # 3. GOAL EXPECTANCY
        expected_goals = features['home_venue_attack'] + features['away_venue_attack']
        features['goal_expectancy'] = expected_goals
        
        # 4. UNPREDICTABILITY FACTOR (DYNAMIC)
        home_consistency = 1.0 / (abs(features['home_venue_momentum']) + balance_threshold)
        away_consistency = 1.0 / (abs(features['away_venue_momentum']) + balance_threshold)
        features['unpredictability'] = 2.0 / (home_consistency + away_consistency)
        
        # 5. RECENT TREND REVERSAL
        home_trend_change = abs(features['home_venue_momentum'] - features['home_total_form'])
        away_trend_change = abs(features['away_venue_momentum'] - features['away_total_form'])
        features['trend_reversal'] = (home_trend_change + away_trend_change) / 2
        
        # 6. DYNAMIC UPSET POTENTIAL
        upset_potential = 0.0
        if features['venue_strength_diff'] < upset_threshold:  # Dynamic threshold
            upset_potential = features['home_desperation_factor'] + features['home_advantage_factor']
        features['upset_potential'] = min(upset_potential, 1.0)
        
        return features
    
    def create_dataset(self, matches, feature_function=None):
        """Dataset oluştur - DYNAMIC FEATURE COUNT ile"""
        if feature_function is None:
            feature_function = self.create_pure_form_features_dynamic
        
        X, y = [], []
        feature_names_set = set()
        
        for match in matches:
            features = feature_function(match)
            if features:
                X.append(features)
                y.append(self._get_match_result(match))
                feature_names_set.update(features.keys())
        
        # Convert to DataFrame
        df = pd.DataFrame(X)
        df = df.fillna(0)
        
        # Dynamic feature selection
        max_features = self.dynamic_thresholds.get('max_features', 25)
        if len(df.columns) > max_features:
            print(f"🎯 Feature selection: {len(df.columns)} -> {max_features} (Dynamic)")
            
            # Select best features
            if len(set(y)) > 1:  # Multiple classes
                selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
                X_selected = selector.fit_transform(df.values, y)
                selected_features = df.columns[selector.get_support()]
                df = pd.DataFrame(X_selected, columns=selected_features)
        
        self.feature_names = list(df.columns)
        return df.values, np.array(y)
    
    def create_ensemble_model(self):
        """DYNAMIC KERNEL NOISE ile ensemble model oluştur"""
        # Dynamic kernel noise al
        noise_level = self.dynamic_thresholds.get('kernel_noise', 1e-3)
        
        # Enhanced kernels with dynamic noise
        rbf_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=noise_level)
        matern_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=noise_level)
        
        models = [
            ('gp_rbf', GaussianProcessClassifier(kernel=rbf_kernel, random_state=42, max_iter_predict=100)),
            ('gp_matern', GaussianProcessClassifier(kernel=matern_kernel, random_state=42, max_iter_predict=100))
        ]
        
        return VotingClassifier(models, voting='soft')
    
    def train(self, train_matches):
        """Model eğitimi - Dynamic thresholds ile"""
        print("🤖 Enhanced Model Training (Dynamic Thresholds)...")
        
        # Dataset oluştur
        X_train, y_train = self.create_dataset(train_matches)
        
        if len(X_train) == 0:
            print("❌ Training data yok!")
            return False
        
        print(f"📊 Training Data: {len(X_train)} samples, {len(self.feature_names)} features")
        print(f"🎯 Dynamic Thresholds Applied: {len(self.dynamic_thresholds)} values")
        
        # Normalize
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Create and train ensemble
        self.ensemble_model = self.create_ensemble_model()
        self.ensemble_model.fit(X_train_scaled, y_train_encoded)
        
        # Cross validation
        cv_scores = cross_val_score(self.ensemble_model, X_train_scaled, y_train_encoded, cv=3, scoring='accuracy')
        print(f"📈 CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        
        self.is_trained = True
        return True
    
    def _get_match_result(self, match):
        """Maç sonucunu al"""
        score = match.get('score', {}).get('fullTime', {})
        home_score = int(score.get('home', 0)) if score.get('home') else 0
        away_score = int(score.get('away', 0)) if score.get('away') else 0
        
        if home_score > away_score:
            return 'H'
        elif home_score == away_score:
            return 'D'
        else:
            return 'A'
    
    def predict_match(self, match_data):
        """Tek maç tahmini - Dynamic confidence ile"""
        if not self.is_trained:
            return None, None
        
        features = self.create_pure_form_features_dynamic(match_data)
        if not features:
            return None, None
        
        # Feature'ları model formatına dönüştür
        feature_vector = []
        for fname in self.feature_names:
            feature_vector.append(features.get(fname, 0))
        
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.ensemble_model.predict(X_scaled)[0]
        probabilities = self.ensemble_model.predict_proba(X_scaled)[0]
        
        # Convert back
        result_classes = self.label_encoder.classes_
        result = result_classes[prediction]
        
        # Probability mapping
        prob_dict = {cls: prob for cls, prob in zip(result_classes, probabilities)}
        confidence = max(probabilities)
        
        return result, prob_dict, confidence

# Test function
def test_enhanced_predictor():
    """Enhanced predictor'ı test et"""
    predictor = EnhancedNoH2HGP()
    
    # Test data
    league_file = "/Users/umitduman/Taho/data/ALM_stat.json"
    target_week = 11
    
    # Load and train
    train_data, test_data = predictor.load_and_analyze_data(league_file, target_week)
    predictor.train(train_data)
    
    # Test predictions
    correct = 0
    total = 0
    
    print(f"\n🎯 Testing Enhanced Predictor (Week {target_week}):")
    print("=" * 60)
    
    for match in test_data:
        actual_result = predictor._get_match_result(match)
        predicted_result, prob_dict, confidence = predictor.predict_match(match)
        
        home = match.get('home', '')
        away = match.get('away', '')
        
        if predicted_result:
            total += 1
            if predicted_result == actual_result:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} {home} vs {away}")
            print(f"   Actual: {actual_result}, Predicted: {predicted_result}, Confidence: {confidence:.3f}")
            print(f"   Probabilities: H:{prob_dict.get('H', 0):.3f} D:{prob_dict.get('D', 0):.3f} A:{prob_dict.get('A', 0):.3f}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n📊 ENHANCED RESULTS:")
    print(f"🎯 Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"🔧 Dynamic Thresholds Used: {len(predictor.dynamic_thresholds)}")
    
    return predictor, accuracy

if __name__ == "__main__":
    predictor, accuracy = test_enhanced_predictor()
