#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No H2H GP Predictor
H2H analizi olmadan pure form-based tahmin sistemi
Sadece güncel form, home advantage ve team strength odaklı
"""

import json
import pandas as pd
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class NoH2HGP:
    def __init__(self):
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.is_trained = False
        self.feature_names = []
        self.team_stats = {}
        
    def load_and_analyze_data(self, league_file, target_week):
        """Veriyi yükle ve analiz et"""
        # Dosya yolunu kontrol et ve düzelt
        if not os.path.isabs(league_file):
            # Göreli yolsa, script dizinine göre ayarla
            script_dir = os.path.dirname(os.path.abspath(__file__))
            league_file = os.path.join(script_dir, league_file)
        
        if not os.path.exists(league_file):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {league_file}")
            
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
        """Takım istatistiklerini hesapla - SADECE GÜNCEL FORM"""
        team_stats = {}
        
        # Chronological sorting by week
        sorted_data = sorted(data, key=lambda x: int(x.get('week', 0)))
        
        for match in sorted_data:
            week = int(match.get('week', 0))
            if week >= target_week:
                continue
                
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            # Score - Güvenli erişim
            score_data = match.get('score', {})
            if score_data is None:
                score_data = {}
            
            full_time = score_data.get('fullTime', {})
            if full_time is None:
                full_time = {}
                
            home_score = int(full_time.get('home', 0)) if full_time.get('home') else 0
            away_score = int(full_time.get('away', 0)) if full_time.get('away') else 0
            
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
        
        # Calculate weighted forms and momentum
        self._calculate_advanced_metrics(team_stats)
        
        self.team_stats = team_stats
        print(f"📈 {len(team_stats)} takım istatistiği hesaplandı (H2H yok)")
    
    def _calculate_advanced_metrics(self, team_stats):
        """Gelişmiş metrikler hesapla"""
        for team, stats in team_stats.items():
            # WEIGHTED FORM (Recent matches more important)
            if stats['last_5_results']:
                weights = [0.35, 0.25, 0.20, 0.15, 0.05]  # Most recent = highest weight
                results = stats['last_5_results']
                
                weighted_sum = sum(w * r for w, r in zip(weights, reversed(results)))
                max_possible = sum(w * 3 for w in weights)  # Max = all wins
                stats['weighted_form'] = weighted_sum / max_possible if max_possible > 0 else 0
            
            # HOME WEIGHTED FORM
            if stats['last_5_home_results']:
                weights = [0.4, 0.3, 0.2, 0.1]  # Adjust for fewer home matches
                results = stats['last_5_home_results']
                
                weighted_sum = sum(w * r for w, r in zip(weights, reversed(results)))
                max_possible = sum(w * 3 for w in weights)
                stats['weighted_home_form'] = weighted_sum / max_possible if max_possible > 0 else 0
            
            # AWAY WEIGHTED FORM
            if stats['last_5_away_results']:
                weights = [0.4, 0.3, 0.2, 0.1]
                results = stats['last_5_away_results']
                
                weighted_sum = sum(w * r for w, r in zip(weights, reversed(results)))
                max_possible = sum(w * 3 for w in weights)
                stats['weighted_away_form'] = weighted_sum / max_possible if max_possible > 0 else 0
            
            # MOMENTUM (Form trend - improving or declining?)
            if len(stats['last_5_results']) >= 3:
                recent_3 = stats['last_5_results'][-3:]
                older_2 = stats['last_5_results'][-5:-3] if len(stats['last_5_results']) >= 5 else []
                
                recent_avg = sum(recent_3) / len(recent_3) if recent_3 else 0
                older_avg = sum(older_2) / len(older_2) if older_2 else recent_avg
                
                stats['momentum'] = recent_avg - older_avg  # Positive = improving
            
            # HOME/AWAY MOMENTUM
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
    
    def create_pure_form_features(self, match_data):
        """Sadece venue-specific form-based feature'lar oluştur - H2H YOK"""
        home_team = match_data.get('home', '')
        away_team = match_data.get('away', '')
        
        home_stats = self.team_stats.get(home_team, {})
        away_stats = self.team_stats.get(away_team, {})
        
        features = {}
        
        # 🎯 VENUE-SPECIFIC PERFORMANCE (ANA FAKTÖR!)
        # Home team sadece kendi evindeki performansına göre değerlendirilmeli
        # Away team sadece kendi deplasman performansına göre değerlendirilmeli
        
        # HOME TEAM'IN EVİNDEKİ BAŞARI ORANI
        home_home_win_rate = home_stats.get('home_wins', 0) / max(home_stats.get('home_matches', 1), 1)
        home_home_draw_rate = home_stats.get('home_draws', 0) / max(home_stats.get('home_matches', 1), 1)
        home_home_points_per_match = (home_stats.get('home_wins', 0) * 3 + home_stats.get('home_draws', 0)) / max(home_stats.get('home_matches', 1), 1)
        
        # AWAY TEAM'IN DEPLASMANINDAKI BAŞARI ORANI  
        away_away_win_rate = away_stats.get('away_wins', 0) / max(away_stats.get('away_matches', 1), 1)
        away_away_draw_rate = away_stats.get('away_draws', 0) / max(away_stats.get('away_matches', 1), 1)
        away_away_points_per_match = (away_stats.get('away_wins', 0) * 3 + away_stats.get('away_draws', 0)) / max(away_stats.get('away_matches', 1), 1)
        
        # VENUE-SPECIFIC MAIN FEATURES
        features['home_venue_strength'] = home_home_points_per_match  # 0-3 arası
        features['away_venue_strength'] = away_away_points_per_match  # 0-3 arası  
        features['venue_strength_diff'] = home_home_points_per_match - away_away_points_per_match  # -3 ile +3 arası
        
        # WIN RATE COMPARISON (VENUE-SPECIFIC)
        features['home_venue_win_rate'] = home_home_win_rate
        features['away_venue_win_rate'] = away_away_win_rate
        features['venue_win_rate_diff'] = home_home_win_rate - away_away_win_rate
        
        # DRAW TENDENCY (VENUE-SPECIFIC)
        features['home_venue_draw_rate'] = home_home_draw_rate
        features['away_venue_draw_rate'] = away_away_draw_rate
        features['venue_draw_tendency'] = (home_home_draw_rate + away_away_draw_rate) / 2
        
        # VENUE-SPECIFIC RECENT FORM (WEIGHTED)
        features['home_venue_form'] = home_stats.get('weighted_home_form', 0.33)
        features['away_venue_form'] = away_stats.get('weighted_away_form', 0.33)
        features['venue_form_diff'] = features['home_venue_form'] - features['away_venue_form']
        
        # VENUE-SPECIFIC MOMENTUM
        features['home_venue_momentum'] = home_stats.get('home_momentum', 0)
        features['away_venue_momentum'] = away_stats.get('away_momentum', 0)
        features['venue_momentum_diff'] = features['home_venue_momentum'] - features['away_venue_momentum']
        
        # VENUE-SPECIFIC GOAL PERFORMANCE
        home_home_goals_per_match = home_stats.get('home_goals_for', 0) / max(home_stats.get('home_matches', 1), 1)
        home_home_conceded_per_match = home_stats.get('home_goals_against', 0) / max(home_stats.get('home_matches', 1), 1)
        away_away_goals_per_match = away_stats.get('away_goals_for', 0) / max(away_stats.get('away_matches', 1), 1)
        away_away_conceded_per_match = away_stats.get('away_goals_against', 0) / max(away_stats.get('away_matches', 1), 1)
        
        features['home_venue_attack'] = home_home_goals_per_match
        features['away_venue_attack'] = away_away_goals_per_match
        features['venue_attack_diff'] = home_home_goals_per_match - away_away_goals_per_match
        
        features['home_venue_defense'] = 1.0 / (home_home_conceded_per_match + 0.5)  # Inverse - yüksek iyi
        features['away_venue_defense'] = 1.0 / (away_away_conceded_per_match + 0.5)
        features['venue_defense_diff'] = features['home_venue_defense'] - features['away_venue_defense']
        
        # SECONDARY: GENEL FORM (Sadece referans için)
        features['home_total_form'] = home_stats.get('weighted_form', 0.33)
        features['away_total_form'] = away_stats.get('weighted_form', 0.33)
        features['total_form_diff'] = features['home_total_form'] - features['away_total_form']
        
        # BALANCE INDICATORS (Draw prediction için)
        features['venue_strength_balance'] = 1.0 / (abs(features['venue_strength_diff']) + 0.1)
        features['venue_form_balance'] = 1.0 / (abs(features['venue_form_diff']) + 0.1)
        features['venue_attack_balance'] = 1.0 / (abs(features['venue_attack_diff']) + 0.1)
        
        # COMBINED VENUE INDICATOR (En önemli feature)
        # Venue strength, form ve momentum'un ağırlıklı kombinasyonu
        venue_weight = 0.5 * features['venue_strength_diff'] + 0.3 * features['venue_form_diff'] + 0.2 * features['venue_momentum_diff']
        features['master_venue_indicator'] = venue_weight
        
        # HOME ADVANTAGE MULTIPLIER
        # Ev sahipliği genelde 0.3-0.5 puan avantaj sağlar
        home_advantage_base = 0.4  # Liga ortalaması
        features['home_advantage_factor'] = home_advantage_base * (1 + features['home_venue_strength'] - 1.5)  # 1.5 ortalama puan
        
        # 🚀 ACCURACY IMPROVEMENT FEATURES
        
        # 1. HOME DESPERATION FACTOR (Evde kötü takımların desperate win potansiyeli)
        # Evde çok kötü olan takımlar bazen sürpriz galibiyetler alabilir
        home_desperation = 0.0
        if features['home_venue_strength'] < 1.0:  # Evde çok kötü performans
            # Ne kadar kötüyse o kadar desperate
            home_desperation = (1.0 - features['home_venue_strength']) * 0.3
        features['home_desperation_factor'] = home_desperation
        
        # 2. DEFENSIVE VULNERABILITY (Savunma açığı faktörü)
        # Çok gol yiyen takımlar sürpriz sonuçlara açık
        home_vulnerability = 1.0 / (features['home_venue_defense'] + 0.1)  # Tersi - yüksek değer = savunma sorunu
        away_vulnerability = 1.0 / (features['away_venue_defense'] + 0.1)
        features['combined_vulnerability'] = (home_vulnerability + away_vulnerability) / 2
        
        # 3. GOAL EXPECTANCY (Beklenen gol sayısı)
        expected_goals = features['home_venue_attack'] + features['away_venue_attack']
        features['goal_expectancy'] = expected_goals
        
        # 4. UNPREDICTABILITY FACTOR (Öngörülmezlik faktörü)
        # Her iki takım da değişken performans gösteriyorsa beklenmedik sonuçlar çıkabilir
        home_consistency = 1.0 / (abs(features['home_venue_momentum']) + 0.1)  # Düşük momentum = tutarsız
        away_consistency = 1.0 / (abs(features['away_venue_momentum']) + 0.1)
        features['unpredictability'] = 2.0 / (home_consistency + away_consistency)
        
        # 5. RECENT TREND REVERSAL (Son trend değişimi)
        # Son maçlarda trend değişimi olmuş mu?
        home_trend_change = abs(features['home_venue_momentum'] - features['home_total_form'])
        away_trend_change = abs(features['away_venue_momentum'] - features['away_total_form'])  
        features['trend_reversal'] = (home_trend_change + away_trend_change) / 2
        
        # 6. UPSET POTENTIAL (Sürpriz sonuç potansiyeli)
        # Zayıf takımın güçlü takımı yenme potansiyeli
        upset_potential = 0.0
        if features['venue_strength_diff'] < -0.5:  # Güçlü fark varsa
            # Ev avantajı + desperate factor upset yaratabilir
            upset_potential = features['home_desperation_factor'] + features['home_advantage_factor']
        features['upset_potential'] = min(upset_potential, 1.0)  # Max 1.0
        
        return features
    
    def train_model(self, train_data):
        """Pure form-based GP modeli eğit"""
        print("🚀 Pure Form GP Modeli Eğitiliyor (H2H yok)...")
        
        # Feature extraction
        X = []
        y = []
        
        for match in train_data:
            features = self.create_pure_form_features(match)
            X.append(list(features.values()))
            
            # Result - Güvenli erişim
            score_data = match.get('score', {})
            if score_data is None:
                score_data = {}
            
            full_time = score_data.get('fullTime', {})
            if full_time is None:
                full_time = {}
                
            home_score = int(full_time.get('home', 0)) if full_time.get('home') else 0
            away_score = int(full_time.get('away', 0)) if full_time.get('away') else 0
            
            if home_score > away_score:
                result = '1'  # Home win
            elif home_score == away_score:
                result = 'X'  # Draw
            else:
                result = '2'  # Away win
                
            y.append(result)
        
        X = np.array(X)
        self.feature_names = list(features.keys())
        
        print(f"📊 Feature sayısı: {X.shape[1]}")
        print(f"📊 Eğitim veri sayısı: {X.shape[0]}")
        
        # Feature selection - Venue-specific features'a öncelik ver
        # Venue-specific feature'ların isimlerini bul
        venue_features = [name for name in self.feature_names if 'venue' in name.lower()]
        print(f"🎯 {len(venue_features)} venue-specific feature tespit edildi")
        
        # Daha fazla feature seç ama venue-specific'lere odakla
        k_best = min(25, X.shape[1])  # Daha fazla feature kullan
        self.feature_selector = SelectKBest(mutual_info_classif, k=k_best)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Seçilen feature'ları göster
        selected_mask = self.feature_selector.get_support()
        selected_features = [self.feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]]
        print(f"📊 Seçilen {len(selected_features)} feature:")
        for i, feature in enumerate(selected_features):
            if 'venue' in feature.lower():
                print(f"  🎯 {feature}")  # Venue-specific features'ı vurgula
            else:
                print(f"     {feature}")
        
        print(f"📊 Venue-specific features oranı: {sum(1 for f in selected_features if 'venue' in f.lower())}/{len(selected_features)}")
        
        # Scaling
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Label encoding
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Multiple GP models
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
        
        print(f"✅ Pure Form Model eğitildi")
        print(f"📊 CV Ortalama: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.is_trained = True
        return cv_scores.mean()
    
    def predict_matches(self, test_data):
        """Maçları tahmin et"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        results = []
        
        for match in test_data:
            features = self.create_pure_form_features(match)
            X = np.array([list(features.values())])
            
            # Feature selection and scaling
            X_selected = self.feature_selector.transform(X)
            X_scaled = self.scaler.transform(X_selected)
            
            # Prediction
            probabilities = self.ensemble_model.predict_proba(X_scaled)[0]
            predicted_class = self.ensemble_model.predict(X_scaled)[0]
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Confidence
            confidence = np.max(probabilities)
            
            # Actual result - Güvenli erişim
            score_data = match.get('score', {})
            if score_data is None:
                score_data = {}
            
            full_time = score_data.get('fullTime', {})
            if full_time is None:
                full_time = {}
                
            home_score = int(full_time.get('home', 0)) if full_time.get('home') else 0
            away_score = int(full_time.get('away', 0)) if full_time.get('away') else 0
            
            if home_score > away_score:
                actual = '1'
            elif home_score == away_score:
                actual = 'X'
            else:
                actual = '2'
            
            result = {
                'match': f"{match.get('home', '')} vs {match.get('away', '')}",
                'home_team': match.get('home', ''),
                'away_team': match.get('away', ''),
                'predicted': predicted_label,
                'actual': actual,
                'confidence': confidence,
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
    
    def analyze_results(self, results):
        """Sonuçları analiz et"""
        correct = sum(r['is_correct'] for r in results)
        total = len(results)
        accuracy = correct / total * 100
        
        print(f"\n🎯 PURE FORM MODEL SONUÇLARI:")
        print(f"Doğru tahmin: {correct}/{total}")
        print(f"Başarı oranı: {accuracy:.1f}%")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            status = "✅" if result['is_correct'] else "❌"
            print(f"{i}. {status} {result['match']}")
            print(f"   Tahmin: {result['predicted']} | Gerçek: {result['actual']} | Güven: {result['confidence']:.1%}")
            print(f"   Skor: {result['home_score']}-{result['away_score']}")
            
            if not result['is_correct']:
                print(f"   🔍 Olasılıklar: 1={result['probabilities']['1']:.2f}, X={result['probabilities']['X']:.2f}, 2={result['probabilities']['2']:.2f}")
            print()
        
        # Failure analysis
        failures = [r for r in results if not r['is_correct']]
        if failures:
            print(f"\n🔍 {len(failures)} BAŞARISIZLIK ANALİZİ:")
            
            # Prediction patterns
            pred_patterns = {}
            for f in failures:
                pattern = f"{f['predicted']} → {f['actual']}"
                pred_patterns[pattern] = pred_patterns.get(pattern, 0) + 1
            
            print("📊 Hata Türleri:")
            for pattern, count in pred_patterns.items():
                print(f"   {pattern}: {count} kez")
        
        return accuracy
    
    def evaluate_predictions(self, predictions, min_confidence=0.7, min_matches=3):
        """Web arayüzü için tahmin değerlendirme metodu"""
        if len(predictions) < min_matches:
            return {
                'accuracy': 0,
                'total_predictions': 0,
                'correct_predictions': 0,
                'predictions': []
            }
        
        # Güven filtreleme
        filtered_predictions = [
            p for p in predictions 
            if p.get('confidence', 0) >= min_confidence
        ]
        
        if len(filtered_predictions) < min_matches:
            return {
                'accuracy': 0,
                'total_predictions': 0,
                'correct_predictions': 0,
                'predictions': []
            }
        
        # Doğruluk hesaplama
        correct_count = sum(1 for p in filtered_predictions if p.get('is_correct', False))
        total_count = len(filtered_predictions)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        # Web arayüzü formatında sonuç
        web_predictions = []
        for p in filtered_predictions:
            web_predictions.append({
                'home_team': p.get('home_team', ''),
                'away_team': p.get('away_team', ''),
                'predicted': p.get('predicted', ''),
                'actual': p.get('actual', ''),
                'confidence': p.get('confidence', 0)
            })
        
        return {
            'accuracy': round(accuracy, 1),
            'total_predictions': total_count,
            'correct_predictions': correct_count,
            'predictions': web_predictions
        }
    
    def analyze_venue_specific_failures(self, results):
        """Başarısız tahminlerin venue-specific performance analizini yapar"""
        
        failed_matches = [r for r in results if not r['is_correct']]
        
        print("\n🔍 BAŞARISIZ TAHMİNLERİN VENUE-SPECIFIC ANALİZİ")
        print("=" * 60)
        
        for match in failed_matches:
            home_team = match['home_team']
            away_team = match['away_team']
            
            print(f"\n❌ {home_team} vs {away_team}")
            print(f"Tahmin: {match['predicted']} | Gerçek: {match['actual']}")
            print(f"Skor: {match['home_score']}-{match['away_score']}")
            
            # Get team stats
            home_stats = self.team_stats.get(home_team, {})
            away_stats = self.team_stats.get(away_team, {})
            
            # HOME TEAM ANALYSIS
            print(f"\n🏠 {home_team} (Ev Sahibi):")
            home_matches = home_stats.get('home_matches', 0)
            home_wins = home_stats.get('home_wins', 0)
            home_draws = home_stats.get('home_draws', 0)
            home_losses = home_stats.get('home_losses', 0)
            home_goals_for = home_stats.get('home_goals_for', 0)
            home_goals_against = home_stats.get('home_goals_against', 0)
            
            if home_matches > 0:
                home_win_rate = (home_wins / home_matches) * 100
                home_points_per_match = (home_wins * 3 + home_draws) / home_matches
                home_goals_per_match = home_goals_for / home_matches
                home_conceded_per_match = home_goals_against / home_matches
                
                print(f"  Evde oynanan maç: {home_matches}")
                print(f"  Evde G-B-M: {home_wins}-{home_draws}-{home_losses}")
                print(f"  Evde kazanma oranı: {home_win_rate:.1f}%")
                print(f"  Evde maç başı puan: {home_points_per_match:.2f}")
                print(f"  Evde maç başı gol: {home_goals_per_match:.2f}")
                print(f"  Evde maç başı yediği: {home_conceded_per_match:.2f}")
                print(f"  Evde form: {home_stats.get('weighted_home_form', 0):.3f}")
            else:
                print("  Evde maç verisi yok!")
            
            # AWAY TEAM ANALYSIS
            print(f"\n✈️ {away_team} (Deplasman):")
            away_matches = away_stats.get('away_matches', 0)
            away_wins = away_stats.get('away_wins', 0)
            away_draws = away_stats.get('away_draws', 0)
            away_losses = away_stats.get('away_losses', 0)
            away_goals_for = away_stats.get('away_goals_for', 0)
            away_goals_against = away_stats.get('away_goals_against', 0)
            
            if away_matches > 0:
                away_win_rate = (away_wins / away_matches) * 100
                away_points_per_match = (away_wins * 3 + away_draws) / away_matches
                away_goals_per_match = away_goals_for / away_matches
                away_conceded_per_match = away_goals_against / away_matches
                
                print(f"  Deplasmanda oynanan maç: {away_matches}")
                print(f"  Deplasmanda G-B-M: {away_wins}-{away_draws}-{away_losses}")
                print(f"  Deplasmanda kazanma oranı: {away_win_rate:.1f}%")
                print(f"  Deplasmanda maç başı puan: {away_points_per_match:.2f}")
                print(f"  Deplasmanda maç başı gol: {away_goals_per_match:.2f}")
                print(f"  Deplasmanda maç başı yediği: {away_conceded_per_match:.2f}")
                print(f"  Deplasman form: {away_stats.get('weighted_away_form', 0):.3f}")
            else:
                print("  Deplasman maç verisi yok!")
            
            # FEATURE ANALYSIS
            print(f"\n📊 Feature Değerleri:")
            features = self.create_pure_form_features({
                'home': home_team,
                'away': away_team
            })
            
            key_features = [
                'venue_strength_diff',
                'venue_form_diff', 
                'venue_win_rate_diff',
                'venue_draw_tendency',
                'master_venue_indicator'
            ]
            
            for feature in key_features:
                if feature in features:
                    print(f"  {feature}: {features[feature]:.3f}")
            
            # PREDICTION EXPLANATION
            print(f"\n💭 Tahmin Açıklaması:")
            venue_strength_diff = features.get('venue_strength_diff', 0)
            
            if venue_strength_diff > 0.5:
                print(f"  Model ev sahibini güçlü gördü (+{venue_strength_diff:.2f})")
            elif venue_strength_diff < -0.5:
                print(f"  Model deplasman takımını güçlü gördü ({venue_strength_diff:.2f})")
            else:
                print(f"  Model dengeli bir maç öngördü ({venue_strength_diff:.2f})")
            
            print("-" * 60)

def test_no_h2h_system():
    """H2H olmayan sistem testi"""
    predictor = NoH2HGP()
    
    # Data yükle
    train_data, test_data = predictor.load_and_analyze_data('data/ALM_stat.json', 11)
    
    # Model eğit
    cv_score = predictor.train_model(train_data)
    
    # Tahmin yap
    results = predictor.predict_matches(test_data)
    
    # Analiz et
    accuracy = predictor.analyze_results(results)
    
    # Venue-specific analysis
    predictor.analyze_venue_specific_failures(results)
    
    if accuracy == 100:
        print("🏆 MÜKEMMEL! H2H olmadan %100 başarı!")
    else:
        print(f"\n💡 PURE FORM İLE BAŞARI: {accuracy:.1f}%")
        print("H2H'sız model sonuçları...")
    
    return predictor, results

if __name__ == "__main__":
    test_no_h2h_system()
