#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perfect Accuracy GP Predictor
11. hafta Almanya ligi %100 doğruluk hedefli GP sistemi
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

class PerfectAccuracyGP:
    def __init__(self):
        self.gp_model = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.is_trained = False
        self.feature_names = []
        self.team_stats = {}
        
    def load_and_analyze_data(self, league_file, target_week):
        """Veriyi yükle ve analiz et"""
        with open(league_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 {league_file} yüklendi: {len(data)} maç")
        
        # Team statistics hesapla
        self.calculate_team_statistics(data, target_week)
        
        # Training data (target week öncesi)
        train_data = [match for match in data if int(match.get('week', 0)) < target_week]
        test_data = [match for match in data if int(match.get('week', 0)) == target_week]
        
        print(f"🎯 Eğitim: {len(train_data)} maç, Test: {len(test_data)} maç")
        return train_data, test_data
    
    def calculate_team_statistics(self, data, target_week):
        """Takım istatistiklerini hesapla"""
        team_stats = {}
        
        for match in data:
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
                        'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                        'home_matches': 0, 'home_wins': 0, 'home_draws': 0,
                        'away_matches': 0, 'away_wins': 0, 'away_draws': 0,
                        'goals_for': 0, 'goals_against': 0,
                        'home_goals_for': 0, 'home_goals_against': 0,
                        'away_goals_for': 0, 'away_goals_against': 0,
                        'recent_form': [],  # Son 5 maç
                        'recent_home_form': [],  # Son 5 ev sahibi maçı
                        'recent_away_form': []   # Son 5 deplasman maçı
                    }
            
            # Update home team stats
            team_stats[home_team]['matches'] += 1
            team_stats[home_team]['home_matches'] += 1
            team_stats[home_team]['goals_for'] += home_score
            team_stats[home_team]['goals_against'] += away_score
            team_stats[home_team]['home_goals_for'] += home_score
            team_stats[home_team]['home_goals_against'] += away_score
            
            # Update away team stats
            team_stats[away_team]['matches'] += 1
            team_stats[away_team]['away_matches'] += 1
            team_stats[away_team]['goals_for'] += away_score
            team_stats[away_team]['goals_against'] += home_score
            team_stats[away_team]['away_goals_for'] += away_score
            team_stats[away_team]['away_goals_against'] += home_score
            
            # Result için form update
            if home_score > away_score:
                # Home win
                team_stats[home_team]['wins'] += 1
                team_stats[home_team]['home_wins'] += 1
                team_stats[away_team]['losses'] += 1
                
                team_stats[home_team]['recent_form'].append(3)  # Win = 3 points
                team_stats[home_team]['recent_home_form'].append(3)
                team_stats[away_team]['recent_form'].append(0)   # Loss = 0 points
                team_stats[away_team]['recent_away_form'].append(0)
                
            elif home_score == away_score:
                # Draw
                team_stats[home_team]['draws'] += 1
                team_stats[home_team]['home_draws'] += 1
                team_stats[away_team]['draws'] += 1
                team_stats[away_team]['away_draws'] += 1
                
                team_stats[home_team]['recent_form'].append(1)  # Draw = 1 point
                team_stats[home_team]['recent_home_form'].append(1)
                team_stats[away_team]['recent_form'].append(1)
                team_stats[away_team]['recent_away_form'].append(1)
                
            else:
                # Away win
                team_stats[away_team]['wins'] += 1
                team_stats[away_team]['away_wins'] += 1
                team_stats[home_team]['losses'] += 1
                
                team_stats[away_team]['recent_form'].append(3)
                team_stats[away_team]['recent_away_form'].append(3)
                team_stats[home_team]['recent_form'].append(0)
                team_stats[home_team]['recent_home_form'].append(0)
            
            # Keep only last 5 matches
            for team in [home_team, away_team]:
                team_stats[team]['recent_form'] = team_stats[team]['recent_form'][-5:]
                team_stats[team]['recent_home_form'] = team_stats[team]['recent_home_form'][-5:]
                team_stats[team]['recent_away_form'] = team_stats[team]['recent_away_form'][-5:]
        
        self.team_stats = team_stats
        print(f"📈 {len(team_stats)} takım istatistiği hesaplandı")
    
    def create_enhanced_features(self, match_list):
        """Gelişmiş feature'lar oluştur"""
        if not isinstance(match_list, list):
            match_list = [match_list]
        
        features_list = []
        labels = []
        
        for match_data in match_list:
            features = {}
            
            home_team = match_data.get('home', '')
            away_team = match_data.get('away', '')
            
            home_stats = self.team_stats.get(home_team, {})
            away_stats = self.team_stats.get(away_team, {})
            
            # Basic team strength features
            features['home_win_rate'] = home_stats.get('wins', 0) / max(home_stats.get('matches', 1), 1)
            features['away_win_rate'] = away_stats.get('wins', 0) / max(away_stats.get('matches', 1), 1)
            features['win_rate_diff'] = features['home_win_rate'] - features['away_win_rate']
            
            # HOME ADVANTAGE FEATURES (Kritik!)
            features['home_advantage'] = home_stats.get('home_wins', 0) / max(home_stats.get('home_matches', 1), 1)
            features['away_disadvantage'] = away_stats.get('away_wins', 0) / max(away_stats.get('away_matches', 1), 1)
            features['home_advantage_diff'] = features['home_advantage'] - features['away_disadvantage']
            
            # Home specific performance
            features['home_goals_avg'] = home_stats.get('home_goals_for', 0) / max(home_stats.get('home_matches', 1), 1)
            features['home_conceded_avg'] = home_stats.get('home_goals_against', 0) / max(home_stats.get('home_matches', 1), 1)
            features['away_goals_avg'] = away_stats.get('away_goals_for', 0) / max(away_stats.get('away_matches', 1), 1)
            features['away_conceded_avg'] = away_stats.get('away_goals_against', 0) / max(away_stats.get('away_matches', 1), 1)
            
            # Goal difference
            features['home_goal_diff'] = home_stats.get('goals_for', 0) - home_stats.get('goals_against', 0)
            features['away_goal_diff'] = away_stats.get('goals_for', 0) - away_stats.get('goals_against', 0)
            features['goal_diff_advantage'] = features['home_goal_diff'] - features['away_goal_diff']
            
            # RECENT FORM FEATURES (Son 5 maç)
            home_form = home_stats.get('recent_form', [])
            away_form = away_stats.get('recent_form', [])
            home_form_avg = sum(home_form) / len(home_form) if home_form else 1.0
            away_form_avg = sum(away_form) / len(away_form) if away_form else 1.0
            
            features['home_recent_form'] = home_form_avg
            features['away_recent_form'] = away_form_avg
            features['form_diff'] = home_form_avg - away_form_avg
            
            # HOME/AWAY SPECIFIC RECENT FORM
            home_home_form = home_stats.get('recent_home_form', [])
            away_away_form = away_stats.get('recent_away_form', [])
            
            features['home_home_form'] = sum(home_home_form) / len(home_home_form) if home_home_form else 1.0
            features['away_away_form'] = sum(away_away_form) / len(away_away_form) if away_away_form else 1.0
            features['home_away_form_diff'] = features['home_home_form'] - features['away_away_form']
            
            # DEFENSIVE STRENGTH (Beraberlik için önemli)
            features['home_defensive'] = 1.0 / (features['home_conceded_avg'] + 0.1)  # Avoid division by zero
            features['away_defensive'] = 1.0 / (features['away_conceded_avg'] + 0.1)
            features['defensive_balance'] = abs(features['home_defensive'] - features['away_defensive'])
            
            # ATTACKING STRENGTH
            features['home_attacking'] = features['home_goals_avg']
            features['away_attacking'] = features['away_goals_avg']
            features['attacking_balance'] = abs(features['home_attacking'] - features['away_attacking'])
            
            # DRAW TENDENCY (Beraberlik eğilimi)
            home_draw_rate = home_stats.get('draws', 0) / max(home_stats.get('matches', 1), 1)
            away_draw_rate = away_stats.get('draws', 0) / max(away_stats.get('matches', 1), 1)
            features['home_draw_tendency'] = home_draw_rate
            features['away_draw_tendency'] = away_draw_rate
            features['combined_draw_tendency'] = (home_draw_rate + away_draw_rate) / 2
            
            # BALANCE INDICATORS (Beraberlik için)
            features['strength_balance'] = 1.0 / (abs(features['win_rate_diff']) + 0.1)
            features['goal_balance'] = 1.0 / (abs(features['home_goals_avg'] - features['away_goals_avg']) + 0.1)
            
            features_list.append(list(features.values()))
            
            # Label extraction
            score = match_data.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            if home_score > away_score:
                labels.append('1')  # Home win
            elif home_score < away_score:
                labels.append('2')  # Away win  
            else:
                labels.append('X')  # Draw
        
        if len(features_list) == 1:
            return np.array(features_list), np.array(labels)
        else:
            return np.array(features_list), np.array(labels)
    
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
    
    def train_enhanced_model(self, train_data):
        """Gelişmiş GP modeli eğit"""
        print("🚀 Gelişmiş GP Modeli Eğitiliyor...")
        
        # Feature extraction
        X = []
        y = []
        
        for match in train_data:
            features = self.create_enhanced_features(match)
            X.append(list(features.values()))
            
            # Result
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0))
            away_score = int(score.get('away', 0))
            
            if home_score > away_score:
                result = '1'
            elif home_score == away_score:
                result = 'X'
            else:
                result = '2'
                
            y.append(result)
        
        X = np.array(X)
        self.feature_names = list(features.keys())
        
        print(f"📊 Feature sayısı: {X.shape[1]}")
        print(f"📊 Eğitim veri sayısı: {X.shape[0]}")
        
        # Feature selection
        self.feature_selector = SelectKBest(mutual_info_classif, k=min(25, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Scaling
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Label encoding
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Multiple GP models with different kernels
        gp1 = GaussianProcessClassifier(
            kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3),
            random_state=42
        )
        
        gp2 = GaussianProcessClassifier(
            kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-3),
            random_state=42
        )
        
        gp3 = GaussianProcessClassifier(
            kernel=ConstantKernel(1.0) * RBF(1.0) * Matern(1.0, nu=2.5),
            random_state=42
        )
        
        # Ensemble model
        self.ensemble_model = VotingClassifier([
            ('gp1', gp1),
            ('gp2', gp2), 
            ('gp3', gp3)
        ], voting='soft')
        
        # Train ensemble
        self.ensemble_model.fit(X_scaled, y_encoded)
        
        # Cross validation
        cv_scores = cross_val_score(self.ensemble_model, X_scaled, y_encoded, cv=5)
        
        print(f"✅ Model eğitildi")
        print(f"📊 CV Ortalama: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.is_trained = True
        return cv_scores.mean()
    
    def train_ensemble_model(self, X, y):
        """Ensemble GP modelini eğit"""
        print("🚀 Ensemble GP Modeli Eğitiliyor...")
        
        # Feature selection if needed
        if X.shape[1] > 25:
            self.feature_selector = SelectKBest(mutual_info_classif, k=25)
            X_selected = self.feature_selector.fit_transform(X, y)
        else:
            X_selected = X
            
        # Scaling 
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Label encoding
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Multiple GP models with different kernels
        gp1 = GaussianProcessClassifier(
            kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3),
            random_state=42
        )
        
        gp2 = GaussianProcessClassifier(
            kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-3),
            random_state=42
        )
        
        gp3 = GaussianProcessClassifier(
            kernel=ConstantKernel(1.0) * RBF(1.0) * Matern(1.0, nu=2.5),
            random_state=42
        )
        
        # Ensemble model
        self.ensemble_model = VotingClassifier([
            ('gp1', gp1),
            ('gp2', gp2), 
            ('gp3', gp3)
        ], voting='soft')
        
        # Train ensemble
        self.ensemble_model.fit(X_scaled, y_encoded)
        
        # Cross validation
        cv_scores = cross_val_score(self.ensemble_model, X_scaled, y_encoded, cv=5)
        
        print(f"✅ Ensemble Model eğitildi")
        print(f"📊 CV Ortalama: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.is_trained = True
        return cv_scores.mean()
    
    def predict_with_uncertainty(self, test_data):
        """Belirsizlik ile tahmin yap"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        results = []
        
        for match in test_data:
            features = self.create_enhanced_features(match)
            X = np.array([list(features.values())])
            
            # Feature selection and scaling
            X_selected = self.feature_selector.transform(X)
            X_scaled = self.scaler.transform(X_selected)
            
            # Ensemble prediction
            probabilities = self.ensemble_model.predict_proba(X_scaled)[0]
            predicted_class = self.ensemble_model.predict(X_scaled)[0]
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Confidence (max probability)
            confidence = np.max(probabilities)
            
            # Uncertainty measures
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            uncertainty = entropy / np.log(len(probabilities))  # Normalized
            
            # Actual result
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0))
            away_score = int(score.get('away', 0))
            
            if home_score > away_score:
                actual = '1'
            elif home_score == away_score:
                actual = 'X'
            else:
                actual = '2'
            
            result = {
                'match': f"{match.get('home_team', '')} vs {match.get('away_team', '')}",
                'predicted': predicted_label,
                'actual': actual,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'probabilities': {
                    '1': probabilities[0] if '1' in self.label_encoder.classes_ else 0,
                    'X': probabilities[1] if 'X' in self.label_encoder.classes_ else 0, 
                    '2': probabilities[2] if '2' in self.label_encoder.classes_ else 0
                },
                'is_correct': predicted_label == actual,
                'home_score': home_score,
                'away_score': away_score
            }
            
            results.append(result)
        
        return results
    
    def predict_with_confidence(self, X):
        """Güven ile tahmin yap"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        results = []
        
        # Feature selection if used during training
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X)
        else:
            X_selected = X
            
        # Scaling
        X_scaled = self.scaler.transform(X_selected)
        
        # Predictions
        probabilities = self.ensemble_model.predict_proba(X_scaled)
        predicted_classes = self.ensemble_model.predict(X_scaled)
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        for i, (probs, pred_label) in enumerate(zip(probabilities, predicted_labels)):
            confidence = np.max(probs)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            uncertainty = entropy / np.log(len(probs))
            
            results.append({
                'prediction': pred_label,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'probabilities': {
                    label: prob for label, prob in zip(self.label_encoder.classes_, probs)
                }
            })
        
        return results

    def analyze_failures(self, results):
        """Başarısız tahminleri analiz et"""
        failures = [r for r in results if not r['is_correct']]
        
        if not failures:
            print("🏆 TÜM TAHMİNLER DOĞRU!")
            return
        
        print(f"\n🔍 {len(failures)} BAŞARISIZ TAHMİN ANALİZİ:")
        print("=" * 60)
        
        for i, failure in enumerate(failures, 1):
            print(f"{i}. {failure['match']}")
            print(f"   Tahmin: {failure['predicted']} | Gerçek: {failure['actual']}")
            print(f"   Güven: {failure['confidence']:.1%} | Belirsizlik: {failure['uncertainty']:.3f}")
            print(f"   Skor: {failure['home_score']}-{failure['away_score']}")
            print(f"   Olasılıklar: 1={failure['probabilities']['1']:.2f}, X={failure['probabilities']['X']:.2f}, 2={failure['probabilities']['2']:.2f}")
            print()
        
        # Failure pattern analysis
        print("📊 BAŞARISIZLIK PATTERN ANALİZİ:")
        
        # High confidence failures
        high_conf_failures = [f for f in failures if f['confidence'] > 0.8]
        if high_conf_failures:
            print(f"⚠️ Yüksek güvenli yanlışlar: {len(high_conf_failures)}")
            
        # Prediction type errors
        pred_errors = {}
        for f in failures:
            key = f"{f['predicted']} -> {f['actual']}"
            pred_errors[key] = pred_errors.get(key, 0) + 1
        
        print("🎯 Tahmin hatası türleri:")
        for error_type, count in pred_errors.items():
            print(f"   {error_type}: {count} kez")

def test_perfect_accuracy():
    """11. hafta Almanya ligi için mükemmellik testi"""
    predictor = PerfectAccuracyGP()
    
    # Data yükle
    train_data, test_data = predictor.load_and_analyze_data('data/ALM_stat.json', 11)
    
    # Model eğit
    cv_score = predictor.train_enhanced_model(train_data)
    
    # Tahmin yap
    results = predictor.predict_with_uncertainty(test_data)
    
    # Sonuçları analiz et
    correct = sum(r['is_correct'] for r in results)
    total = len(results)
    accuracy = correct / total * 100
    
    print(f"\n🎯 SONUÇ RAPORU:")
    print(f"Doğru tahmin: {correct}/{total}")
    print(f"Başarı oranı: {accuracy:.1f}%")
    
    if accuracy == 100:
        print("🏆 MÜKEMMEL! %100 BAŞARI ELDE EDİLDİ!")
    else:
        predictor.analyze_failures(results)
        
        print(f"\n💡 İYİLEŞTİRME ÖNERİLERİ:")
        print("1. Daha fazla feature engineering")
        print("2. Ensemble model ağırlıklarını optimize et")
        print("3. Takım-spesifik modeller oluştur")
        print("4. Temporal patterns'i daha iyi yakala")
    
    return predictor, results

if __name__ == "__main__":
    test_perfect_accuracy()
