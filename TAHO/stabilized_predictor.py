#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Stabilized Enhanced Predictor
Optimize edilmiş, istikrarlı dynamic threshold sistemi
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
import warnings
warnings.filterwarnings('ignore')

class StabilizedPredictor:
    def __init__(self):
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.is_trained = False
        self.feature_names = []
        self.team_stats = {}
        self.stabilized_thresholds = {}
        
    def load_and_analyze_data(self, league_file, target_week):
        """Veriyi yükle ve STABILIZED threshold'ları hesapla"""
        with open(league_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 {league_file} yüklendi: {len(data)} maç")
        
        # Training data
        train_data = [match for match in data if int(match.get('week', 0)) < target_week]
        test_data = [match for match in data if int(match.get('week', 0)) == target_week]
        
        # STABILIZED threshold calculation
        self.stabilized_thresholds = self._calculate_stabilized_thresholds(train_data)
        
        print("🎯 Stabilized Thresholds:")
        for key, value in self.stabilized_thresholds.items():
            if isinstance(value, list):
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value:.4f}")
        
        # Team statistics hesapla
        self.calculate_team_statistics(data, target_week)
        
        print(f"🎯 Eğitim: {len(train_data)} maç, Test: {len(test_data)} maç")
        return train_data, test_data
    
    def _calculate_stabilized_thresholds(self, train_data):
        """STABILIZED threshold hesaplamaları"""
        thresholds = {}
        
        # 1. STABILIZED Defensive factor (median-based)
        all_goals = []
        for match in train_data:
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            all_goals.extend([home_score, away_score])
        
        if all_goals:
            median_goals = np.median(all_goals)
            q75 = np.percentile(all_goals, 75)
            thresholds['defensive_factor'] = max(0.8, min(2.0, median_goals + 0.3))
        else:
            thresholds['defensive_factor'] = 1.0
        
        # 2. STABILIZED Home advantage (confidence-based)
        home_wins = total_matches = 0
        for match in train_data:
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            total_matches += 1
            if home_score > away_score:
                home_wins += 1
        
        if total_matches >= 20:
            home_win_rate = home_wins / total_matches
            # Conservative home advantage
            thresholds['home_advantage'] = max(0.1, min(0.3, (home_win_rate - 0.33) * 1.2))
        else:
            thresholds['home_advantage'] = 0.15
        
        # 3. STABILIZED Balance threshold (robust)
        team_strengths = self._calculate_team_strengths(train_data)
        if len(team_strengths) > 3:
            strengths = list(team_strengths.values())
            median_strength = np.median(strengths)
            mad = np.median([abs(s - median_strength) for s in strengths])
            thresholds['balance_threshold'] = max(0.08, min(0.2, mad / 2))
        else:
            thresholds['balance_threshold'] = 0.1
        
        # 4. CONSERVATIVE Feature count
        data_size = len(train_data)
        if data_size < 50:
            thresholds['max_features'] = 6
        elif data_size < 80:
            thresholds['max_features'] = 8
        elif data_size < 100:
            thresholds['max_features'] = 10
        else:
            thresholds['max_features'] = min(12, data_size // 8)
        
        # 5. STABILIZED Other thresholds
        if team_strengths:
            strengths = list(team_strengths.values())
            thresholds['desperation_threshold'] = max(0.9, np.percentile(strengths, 35))
            thresholds['upset_threshold'] = max(-0.8, -0.7 * np.std(strengths))
        else:
            thresholds['desperation_threshold'] = 1.0
            thresholds['upset_threshold'] = -0.5
        
        # 6. SMOOTH Temporal weights
        thresholds['temporal_weights'] = [0.28, 0.24, 0.20, 0.16, 0.12]
        
        # 7. ADAPTIVE Kernel noise
        if data_size < 60:
            thresholds['kernel_noise'] = 4e-3
        elif data_size < 100:
            thresholds['kernel_noise'] = 3e-3
        else:
            thresholds['kernel_noise'] = 2e-3
        
        return thresholds
    
    def _calculate_team_strengths(self, train_data):
        """Team strength hesaplama"""
        team_stats = {}
        
        for match in train_data:
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            for team in [home_team, away_team]:
                if team not in team_stats:
                    team_stats[team] = {'points': 0, 'matches': 0}
            
            team_stats[home_team]['matches'] += 1
            team_stats[away_team]['matches'] += 1
            
            if home_score > away_score:
                team_stats[home_team]['points'] += 3
            elif home_score == away_score:
                team_stats[home_team]['points'] += 1
                team_stats[away_team]['points'] += 1
            else:
                team_stats[away_team]['points'] += 3
        
        strengths = {}
        for team, stats in team_stats.items():
            if stats['matches'] >= 3:
                strengths[team] = stats['points'] / stats['matches']
        
        return strengths
    
    def calculate_team_statistics(self, data, target_week):
        """Team statistics (existing logic with stabilized weights)"""
        team_stats = {}
        
        sorted_data = sorted(data, key=lambda x: int(x.get('week', 0)))
        
        for match in sorted_data:
            week = int(match.get('week', 0))
            if week >= target_week:
                continue
                
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            # Initialize team stats
            for team in [home_team, away_team]:
                if team not in team_stats:
                    team_stats[team] = {
                        'total_matches': 0, 'total_wins': 0, 'total_draws': 0, 'total_losses': 0,
                        'total_goals_for': 0, 'total_goals_against': 0,
                        'home_matches': 0, 'home_wins': 0, 'home_draws': 0, 'home_losses': 0,
                        'home_goals_for': 0, 'home_goals_against': 0,
                        'away_matches': 0, 'away_wins': 0, 'away_draws': 0, 'away_losses': 0,
                        'away_goals_for': 0, 'away_goals_against': 0,
                        'last_5_results': [], 'last_5_home_results': [], 'last_5_away_results': [],
                        'weighted_form': 0.0, 'weighted_home_form': 0.0, 'weighted_away_form': 0.0,
                        'momentum': 0.0, 'home_momentum': 0.0, 'away_momentum': 0.0
                    }
            
            # Update statistics (same logic as before)
            team_stats[home_team]['total_matches'] += 1
            team_stats[home_team]['home_matches'] += 1
            team_stats[home_team]['total_goals_for'] += home_score
            team_stats[home_team]['total_goals_against'] += away_score
            team_stats[home_team]['home_goals_for'] += home_score
            team_stats[home_team]['home_goals_against'] += away_score
            
            team_stats[away_team]['total_matches'] += 1
            team_stats[away_team]['away_matches'] += 1
            team_stats[away_team]['total_goals_for'] += away_score
            team_stats[away_team]['total_goals_against'] += home_score
            team_stats[away_team]['away_goals_for'] += away_score
            team_stats[away_team]['away_goals_against'] += home_score
            
            # Result updates
            if home_score > away_score:
                team_stats[home_team]['total_wins'] += 1
                team_stats[home_team]['home_wins'] += 1
                team_stats[away_team]['total_losses'] += 1
                team_stats[away_team]['away_losses'] += 1
                
                team_stats[home_team]['last_5_results'].append(3)
                team_stats[home_team]['last_5_home_results'].append(3)
                team_stats[away_team]['last_5_results'].append(0)
                team_stats[away_team]['last_5_away_results'].append(0)
                
            elif home_score == away_score:
                team_stats[home_team]['total_draws'] += 1
                team_stats[home_team]['home_draws'] += 1
                team_stats[away_team]['total_draws'] += 1
                team_stats[away_team]['away_draws'] += 1
                
                team_stats[home_team]['last_5_results'].append(1)
                team_stats[home_team]['last_5_home_results'].append(1)
                team_stats[away_team]['last_5_results'].append(1)
                team_stats[away_team]['last_5_away_results'].append(1)
                
            else:
                team_stats[away_team]['total_wins'] += 1
                team_stats[away_team]['away_wins'] += 1
                team_stats[home_team]['total_losses'] += 1
                team_stats[home_team]['home_losses'] += 1
                
                team_stats[away_team]['last_5_results'].append(3)
                team_stats[away_team]['last_5_away_results'].append(3)
                team_stats[home_team]['last_5_results'].append(0)
                team_stats[home_team]['last_5_home_results'].append(0)
            
            # Keep only last 5
            for team in [home_team, away_team]:
                team_stats[team]['last_5_results'] = team_stats[team]['last_5_results'][-5:]
                team_stats[team]['last_5_home_results'] = team_stats[team]['last_5_home_results'][-5:]
                team_stats[team]['last_5_away_results'] = team_stats[team]['last_5_away_results'][-5:]
        
        # Calculate weighted forms with STABILIZED weights
        self._calculate_stabilized_forms(team_stats)
        
        self.team_stats = team_stats
        print(f"📈 {len(team_stats)} takım istatistiği hesaplandı (Stabilized)")
    
    def _calculate_stabilized_forms(self, team_stats):
        """STABILIZED weighted form calculations"""
        weights = self.stabilized_thresholds.get('temporal_weights', [0.28, 0.24, 0.20, 0.16, 0.12])
        
        for team, stats in team_stats.items():
            # Weighted form
            if stats['last_5_results']:
                results = stats['last_5_results']
                weighted_sum = sum(w * r for w, r in zip(weights, reversed(results)))
                max_possible = sum(w * 3 for w in weights)
                stats['weighted_form'] = weighted_sum / max_possible if max_possible > 0 else 0
            
            # Home/Away forms
            if stats['last_5_home_results']:
                home_weights = weights[:min(len(weights), len(stats['last_5_home_results']))]
                results = stats['last_5_home_results']
                weighted_sum = sum(w * r for w, r in zip(home_weights, reversed(results)))
                max_possible = sum(w * 3 for w in home_weights)
                stats['weighted_home_form'] = weighted_sum / max_possible if max_possible > 0 else 0
            
            if stats['last_5_away_results']:
                away_weights = weights[:min(len(weights), len(stats['last_5_away_results']))]
                results = stats['last_5_away_results']
                weighted_sum = sum(w * r for w, r in zip(away_weights, reversed(results)))
                max_possible = sum(w * 3 for w in away_weights)
                stats['weighted_away_form'] = weighted_sum / max_possible if max_possible > 0 else 0
            
            # Momentum calculations (same as before)
            if len(stats['last_5_results']) >= 3:
                recent_3 = stats['last_5_results'][-3:]
                older_2 = stats['last_5_results'][-5:-3] if len(stats['last_5_results']) >= 5 else []
                recent_avg = sum(recent_3) / len(recent_3) if recent_3 else 0
                older_avg = sum(older_2) / len(older_2) if older_2 else recent_avg
                stats['momentum'] = recent_avg - older_avg
    
    def create_stabilized_features(self, match_data):
        """STABILIZED feature creation"""
        home_team = match_data.get('home', '')
        away_team = match_data.get('away', '')
        
        home_stats = self.team_stats.get(home_team, {})
        away_stats = self.team_stats.get(away_team, {})
        
        features = {}
        
        # Get stabilized thresholds
        defensive_factor = self.stabilized_thresholds.get('defensive_factor', 1.0)
        balance_threshold = self.stabilized_thresholds.get('balance_threshold', 0.1)
        home_advantage = self.stabilized_thresholds.get('home_advantage', 0.15)
        desperation_threshold = self.stabilized_thresholds.get('desperation_threshold', 1.0)
        upset_threshold = self.stabilized_thresholds.get('upset_threshold', -0.5)
        
        # Venue-specific performance
        home_home_points_per_match = (home_stats.get('home_wins', 0) * 3 + home_stats.get('home_draws', 0)) / max(home_stats.get('home_matches', 1), 1)
        away_away_points_per_match = (away_stats.get('away_wins', 0) * 3 + away_stats.get('away_draws', 0)) / max(away_stats.get('away_matches', 1), 1)
        
        features['home_venue_strength'] = home_home_points_per_match
        features['away_venue_strength'] = away_away_points_per_match
        features['venue_strength_diff'] = home_home_points_per_match - away_away_points_per_match
        
        # Venue-specific forms
        features['home_venue_form'] = home_stats.get('weighted_home_form', 0.33)
        features['away_venue_form'] = away_stats.get('weighted_away_form', 0.33)
        features['venue_form_diff'] = features['home_venue_form'] - features['away_venue_form']
        
        # Goal performance with STABILIZED defensive factor
        home_home_goals_per_match = home_stats.get('home_goals_for', 0) / max(home_stats.get('home_matches', 1), 1)
        home_home_conceded_per_match = home_stats.get('home_goals_against', 0) / max(home_stats.get('home_matches', 1), 1)
        away_away_goals_per_match = away_stats.get('away_goals_for', 0) / max(away_stats.get('away_matches', 1), 1)
        away_away_conceded_per_match = away_stats.get('away_goals_against', 0) / max(away_stats.get('away_matches', 1), 1)
        
        features['home_venue_attack'] = home_home_goals_per_match
        features['away_venue_attack'] = away_away_goals_per_match
        features['venue_attack_diff'] = home_home_goals_per_match - away_away_goals_per_match
        
        features['home_venue_defense'] = 1.0 / (home_home_conceded_per_match + defensive_factor)
        features['away_venue_defense'] = 1.0 / (away_away_conceded_per_match + defensive_factor)
        features['venue_defense_diff'] = features['home_venue_defense'] - features['away_venue_defense']
        
        # Total forms
        features['home_total_form'] = home_stats.get('weighted_form', 0.33)
        features['away_total_form'] = away_stats.get('weighted_form', 0.33)
        features['total_form_diff'] = features['home_total_form'] - features['away_total_form']
        
        # STABILIZED balance indicators
        features['venue_strength_balance'] = 1.0 / (abs(features['venue_strength_diff']) + balance_threshold)
        features['venue_form_balance'] = 1.0 / (abs(features['venue_form_diff']) + balance_threshold)
        
        # STABILIZED home advantage
        features['home_advantage_factor'] = home_advantage * (1 + features['home_venue_strength'] - 1.5)
        
        # STABILIZED desperation factor
        home_desperation = 0.0
        if features['home_venue_strength'] < desperation_threshold:
            home_desperation = (desperation_threshold - features['home_venue_strength']) * 0.25
        features['home_desperation_factor'] = home_desperation
        
        # STABILIZED upset potential
        upset_potential = 0.0
        if features['venue_strength_diff'] < upset_threshold:
            upset_potential = features['home_desperation_factor'] + features['home_advantage_factor'] * 0.5
        features['upset_potential'] = min(upset_potential, 0.8)
        
        return features
    
    def create_dataset(self, matches):
        """Dataset creation with STABILIZED feature selection"""
        X, y = [], []
        
        for match in matches:
            features = self.create_stabilized_features(match)
            if features:
                X.append(features)
                y.append(self._get_match_result(match))
        
        df = pd.DataFrame(X)
        df = df.fillna(0)
        
        # STABILIZED feature selection
        max_features = self.stabilized_thresholds.get('max_features', 10)
        if len(df.columns) > max_features:
            print(f"🎯 Stabilized Feature Selection: {len(df.columns)} -> {max_features}")
            
            if len(set(y)) > 1:
                selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
                X_selected = selector.fit_transform(df.values, y)
                selected_features = df.columns[selector.get_support()]
                df = pd.DataFrame(X_selected, columns=selected_features)
        
        self.feature_names = list(df.columns)
        return df.values, np.array(y)
    
    def train(self, train_matches):
        """Model training with STABILIZED parameters"""
        print("🤖 Stabilized Model Training...")
        
        X_train, y_train = self.create_dataset(train_matches)
        
        if len(X_train) == 0:
            print("❌ Training data yok!")
            return False
        
        print(f"📊 Training Data: {len(X_train)} samples, {len(self.feature_names)} features")
        
        # Normalize
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # STABILIZED ensemble model
        noise_level = self.stabilized_thresholds.get('kernel_noise', 3e-3)
        
        rbf_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=noise_level)
        matern_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=noise_level)
        
        models = [
            ('gp_rbf', GaussianProcessClassifier(kernel=rbf_kernel, random_state=42, max_iter_predict=100)),
            ('gp_matern', GaussianProcessClassifier(kernel=matern_kernel, random_state=42, max_iter_predict=100))
        ]
        
        self.ensemble_model = VotingClassifier(models, voting='soft')
        self.ensemble_model.fit(X_train_scaled, y_train_encoded)
        
        # Cross validation
        cv_scores = cross_val_score(self.ensemble_model, X_train_scaled, y_train_encoded, cv=3, scoring='accuracy')
        print(f"📈 CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        
        self.is_trained = True
        return True
    
    def predict_match(self, match_data):
        """Match prediction"""
        if not self.is_trained:
            return None, None, None
        
        features = self.create_stabilized_features(match_data)
        if not features:
            return None, None, None
        
        feature_vector = []
        for fname in self.feature_names:
            feature_vector.append(features.get(fname, 0))
        
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.ensemble_model.predict(X_scaled)[0]
        probabilities = self.ensemble_model.predict_proba(X_scaled)[0]
        
        result_classes = self.label_encoder.classes_
        result = result_classes[prediction]
        
        prob_dict = {cls: prob for cls, prob in zip(result_classes, probabilities)}
        confidence = max(probabilities)
        
        return result, prob_dict, confidence
    
    def _get_match_result(self, match):
        """Match result extraction"""
        score = match.get('score', {}).get('fullTime', {})
        home_score = int(score.get('home', 0)) if score.get('home') else 0
        away_score = int(score.get('away', 0)) if score.get('away') else 0
        
        if home_score > away_score:
            return 'H'
        elif home_score == away_score:
            return 'D'
        else:
            return 'A'

# Test function
def test_stabilized_predictor():
    """Stabilized predictor test"""
    print("🎯 STABILIZED DYNAMIC PREDICTOR TEST")
    print("=" * 60)
    
    league_file = "/Users/umitduman/Taho/data/ALM_stat.json"
    test_weeks = [10, 11, 12]
    
    results = {}
    
    for week in test_weeks:
        print(f"\n📅 Testing Week {week}...")
        
        predictor = StabilizedPredictor()
        train_data, test_data = predictor.load_and_analyze_data(league_file, week)
        
        if predictor.train(train_data):
            correct = total = 0
            
            for match in test_data:
                actual = predictor._get_match_result(match)
                predicted, prob_dict, confidence = predictor.predict_match(match)
                
                if predicted:
                    total += 1
                    if predicted == actual:
                        correct += 1
            
            accuracy = correct / total if total > 0 else 0
            results[week] = accuracy
            
            print(f"✅ Week {week}: {accuracy:.3f} ({correct}/{total})")
        else:
            print(f"❌ Week {week}: Training failed")
    
    # Summary
    if results:
        accuracies = list(results.values())
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f"\n📊 STABILIZED RESULTS SUMMARY:")
        print(f"   Average Accuracy: {avg_accuracy:.3f}")
        print(f"   Std Deviation: {std_accuracy:.3f}")
        print(f"   Range: {min(accuracies):.3f} - {max(accuracies):.3f}")
        print(f"   Consistency: {'GOOD' if std_accuracy < 0.15 else 'MODERATE' if std_accuracy < 0.25 else 'POOR'}")
        
        # Improvement analysis
        print(f"\n💡 STABILIZATION IMPACT:")
        print(f"   ✅ Reduced threshold volatility with robust calculations")
        print(f"   ✅ Conservative feature selection prevents overfitting")
        print(f"   ✅ Confidence-based home advantage calculation")
        print(f"   ✅ Median-based defensive factors reduce outlier impact")
        
        if std_accuracy < 0.2:
            print(f"   🎉 STABILIZATION SUCCESSFUL!")
        else:
            print(f"   ⚠️  Further stabilization needed")
    
    return results

if __name__ == "__main__":
    results = test_stabilized_predictor()
