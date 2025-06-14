#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Smart GP Predictor
H2H olmadan, akıllı takım sınıflandırması ve zamansal ağırlıklandırma ile
%100 doğruluk hedefli sistem
"""

import json
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class UltraSmartGP:
    def __init__(self):
        self.ensemble_model = None
        self.scaler = RobustScaler()  # RobustScaler outlier'lara daha dayanıklı
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.is_trained = False
        self.feature_names = []
        self.team_stats = {}
        self.team_rankings = {}
        self.league_averages = {}
        self.week_weights = {}  # Haftalara göre ağırlık
        
    def load_and_analyze_data(self, league_file, target_week):
        """Veriyi yükle ve akıllı analiz et"""
        with open(league_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 {league_file} yüklendi: {len(data)} maç")
        
        # Liga ortalamalarını hesapla
        self.calculate_league_averages(data, target_week)
        
        # Takım istatistiklerini zamansal ağırlıkla hesapla
        self.calculate_weighted_team_statistics(data, target_week)
        
        # Takım güç sıralaması oluştur
        self.create_team_power_rankings()
        
        # Training data (target week öncesi)
        train_data = [match for match in data if int(match.get('week', 0)) < target_week]
        test_data = [match for match in data if int(match.get('week', 0)) == target_week]
        
        print(f"🎯 Eğitim: {len(train_data)} maç, Test: {len(test_data)} maç")
        return train_data, test_data
    
    def calculate_league_averages(self, data, target_week):
        """Liga ortalamalarını hesapla (normalizasyon için)"""
        total_goals = 0
        total_matches = 0
        home_wins = 0
        draws = 0
        away_wins = 0
        
        for match in data:
            week = int(match.get('week', 0))
            if week >= target_week:
                continue
                
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            total_goals += home_score + away_score
            total_matches += 1
            
            if home_score > away_score:
                home_wins += 1
            elif home_score == away_score:
                draws += 1
            else:
                away_wins += 1
        
        if total_matches > 0:
            self.league_averages = {
                'goals_per_match': total_goals / total_matches,
                'home_win_rate': home_wins / total_matches,
                'draw_rate': draws / total_matches,
                'away_win_rate': away_wins / total_matches,
                'total_matches': total_matches
            }
        
        print(f"📈 Liga ortalamaları: Maç başı gol: {self.league_averages.get('goals_per_match', 0):.2f}")
    
    def get_week_weight(self, week, target_week):
        """Haftaya göre ağırlık hesapla (yakın haftalar daha ağır)"""
        week_diff = target_week - week
        if week_diff <= 0:
            return 0.0
        
        # Exponential decay - son 3 hafta çok ağır, sonra azalır
        if week_diff <= 3:
            return 1.0
        elif week_diff <= 6:
            return 0.8
        elif week_diff <= 10:
            return 0.5
        else:
            return 0.2
    
    def calculate_weighted_team_statistics(self, data, target_week):
        """Zamansal ağırlıklı takım istatistikleri"""
        team_stats = {}
        
        # Initialize team stats
        for match in data:
            week = int(match.get('week', 0))
            if week >= target_week:
                continue
                
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            for team in [home_team, away_team]:
                if team not in team_stats:
                    team_stats[team] = {
                        'total_weight': 0.0,
                        'weighted_points': 0.0,
                        'weighted_goals_for': 0.0,
                        'weighted_goals_against': 0.0,
                        'weighted_home_points': 0.0,
                        'weighted_away_points': 0.0,
                        'weighted_home_matches': 0.0,
                        'weighted_away_matches': 0.0,
                        'recent_form_weighted': [],  # (result, weight) tuples
                        'form_trend': 0.0,  # Pozitif = yükseliş, negatif = düşüş
                        'consistency': 0.0,  # Form tutarlılığı
                        'pressure_performance': 0.0,  # Kritik maçlarda performans
                    }
        
        # Calculate weighted stats
        for match in data:
            week = int(match.get('week', 0))
            if week >= target_week:
                continue
                
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            # Week weight
            weight = self.get_week_weight(week, target_week)
            if weight == 0:
                continue
            
            # Score
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            # Update weighted stats
            team_stats[home_team]['total_weight'] += weight
            team_stats[away_team]['total_weight'] += weight
            
            team_stats[home_team]['weighted_goals_for'] += home_score * weight
            team_stats[home_team]['weighted_goals_against'] += away_score * weight
            team_stats[away_team]['weighted_goals_for'] += away_score * weight
            team_stats[away_team]['weighted_goals_against'] += home_score * weight
            
            team_stats[home_team]['weighted_home_matches'] += weight
            team_stats[away_team]['weighted_away_matches'] += weight
            
            # Points calculation
            if home_score > away_score:
                # Home win
                home_points = 3
                away_points = 0
            elif home_score == away_score:
                # Draw
                home_points = 1
                away_points = 1
            else:
                # Away win
                home_points = 0
                away_points = 3
            
            team_stats[home_team]['weighted_points'] += home_points * weight
            team_stats[home_team]['weighted_home_points'] += home_points * weight
            team_stats[away_team]['weighted_points'] += away_points * weight
            team_stats[away_team]['weighted_away_points'] += away_points * weight
            
            # Recent form with weight
            team_stats[home_team]['recent_form_weighted'].append((home_points, weight, week))
            team_stats[away_team]['recent_form_weighted'].append((away_points, weight, week))
        
        # Calculate derived metrics
        for team, stats in team_stats.items():
            if stats['total_weight'] > 0:
                # Average per match metrics
                stats['avg_points_per_match'] = stats['weighted_points'] / stats['total_weight']
                stats['avg_goals_for'] = stats['weighted_goals_for'] / stats['total_weight']
                stats['avg_goals_against'] = stats['weighted_goals_against'] / stats['total_weight']
                stats['avg_goal_diff'] = stats['avg_goals_for'] - stats['avg_goals_against']
                
                # Home/Away specific
                if stats['weighted_home_matches'] > 0:
                    stats['avg_home_points'] = stats['weighted_home_points'] / stats['weighted_home_matches']
                else:
                    stats['avg_home_points'] = 0
                    
                if stats['weighted_away_matches'] > 0:
                    stats['avg_away_points'] = stats['weighted_away_points'] / stats['weighted_away_matches']
                else:
                    stats['avg_away_points'] = 0
                
                # Form trend calculation (son 5 maçın trend'i)
                recent_results = sorted(stats['recent_form_weighted'], key=lambda x: x[2])[-5:]
                if len(recent_results) >= 3:
                    early_avg = sum(r[0] * r[1] for r in recent_results[:2]) / sum(r[1] for r in recent_results[:2])
                    late_avg = sum(r[0] * r[1] for r in recent_results[-2:]) / sum(r[1] for r in recent_results[-2:])
                    stats['form_trend'] = late_avg - early_avg
                
                # Consistency (form'un standart sapması)
                recent_points = [r[0] for r in recent_results]
                if len(recent_points) > 1:
                    stats['consistency'] = 1.0 / (np.std(recent_points) + 0.1)  # Yüksek = tutarlı
        
        self.team_stats = team_stats
        print(f"📈 {len(team_stats)} takım için ağırlıklı istatistikler hesaplandı")
    
    def create_team_power_rankings(self):
        """Takım güç sıralaması oluştur"""
        if not self.team_stats:
            return
        
        # Multiple criteria için composite score
        team_scores = {}
        
        for team, stats in self.team_stats.items():
            if stats['total_weight'] == 0:
                continue
                
            # Composite power score
            power_score = (
                stats.get('avg_points_per_match', 0) * 0.4 +  # Points ağırlığı
                stats.get('avg_goal_diff', 0) * 0.2 +         # Goal difference
                stats.get('form_trend', 0) * 0.2 +            # Form trend
                stats.get('consistency', 0) * 0.1 +           # Consistency
                (stats.get('avg_home_points', 0) + stats.get('avg_away_points', 0)) * 0.1  # Home+Away balance
            )
            
            team_scores[team] = power_score
        
        # Ranking oluştur
        sorted_teams = sorted(team_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (team, score) in enumerate(sorted_teams, 1):
            self.team_rankings[team] = {
                'rank': rank,
                'power_score': score,
                'tier': self._get_team_tier(rank, len(sorted_teams))
            }
        
        print(f"🏆 Takım güç sıralaması oluşturuldu")
        
        # Top 5 ve bottom 5'i göster
        print("🔝 Top 5:")
        for i, (team, score) in enumerate(sorted_teams[:5], 1):
            print(f"   {i}. {team}: {score:.3f}")
        
        if len(sorted_teams) > 10:
            print("🔻 Bottom 5:")
            for i, (team, score) in enumerate(sorted_teams[-5:], len(sorted_teams)-4):
                print(f"   {i}. {team}: {score:.3f}")
    
    def _get_team_tier(self, rank, total_teams):
        """Takım tier'ını belirle"""
        if rank <= total_teams * 0.2:
            return 'ELITE'
        elif rank <= total_teams * 0.4:
            return 'STRONG'
        elif rank <= total_teams * 0.6:
            return 'MIDDLE'
        elif rank <= total_teams * 0.8:
            return 'WEAK'
        else:
            return 'BOTTOM'
    
    def create_ultra_smart_features(self, match_list):
        """Ultra akıllı feature'lar oluştur"""
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
            home_ranking = self.team_rankings.get(home_team, {})
            away_ranking = self.team_rankings.get(away_team, {})
            
            # 1. POWER RANKING FEATURES
            features['home_power_score'] = home_ranking.get('power_score', 0)
            features['away_power_score'] = away_ranking.get('power_score', 0)
            features['power_score_diff'] = features['home_power_score'] - features['away_power_score']
            
            home_rank = home_ranking.get('rank', 99)
            away_rank = away_ranking.get('rank', 99)
            features['rank_advantage'] = away_rank - home_rank  # Pozitif = home avantajlı
            
            # 2. TIER MATCHING
            home_tier = home_ranking.get('tier', 'MIDDLE')
            away_tier = away_ranking.get('tier', 'MIDDLE')
            tier_values = {'ELITE': 5, 'STRONG': 4, 'MIDDLE': 3, 'WEAK': 2, 'BOTTOM': 1}
            features['tier_diff'] = tier_values.get(home_tier, 3) - tier_values.get(away_tier, 3)
            
            # Tier mismatch (büyük fark = sürpriz potansiyeli)
            features['tier_mismatch'] = abs(features['tier_diff'])
            features['upset_potential'] = 1.0 if features['tier_diff'] < -2 else 0.0
            
            # 3. WEIGHTED PERFORMANCE METRICS
            features['home_avg_points'] = home_stats.get('avg_points_per_match', 1.0)
            features['away_avg_points'] = away_stats.get('avg_points_per_match', 1.0)
            features['points_diff'] = features['home_avg_points'] - features['away_avg_points']
            
            # Liga ortalaması ile karşılaştırma
            league_avg_points = self.league_averages.get('home_win_rate', 0.33) * 3 + \
                               self.league_averages.get('draw_rate', 0.33) * 1
            features['home_vs_league'] = features['home_avg_points'] - league_avg_points
            features['away_vs_league'] = features['away_avg_points'] - league_avg_points
            
            # 4. GOAL METRICS
            features['home_attack'] = home_stats.get('avg_goals_for', 1.0)
            features['away_attack'] = away_stats.get('avg_goals_for', 1.0)
            features['home_defense'] = 3.0 - home_stats.get('avg_goals_against', 1.5)
            features['away_defense'] = 3.0 - away_stats.get('avg_goals_against', 1.5)
            
            features['attack_diff'] = features['home_attack'] - features['away_attack']
            features['defense_diff'] = features['home_defense'] - features['away_defense']
            
            # Goal difference advantage
            features['goal_diff_advantage'] = home_stats.get('avg_goal_diff', 0) - away_stats.get('avg_goal_diff', 0)
            
            # 5. HOME/AWAY SPECIFIC PERFORMANCE
            features['home_home_strength'] = home_stats.get('avg_home_points', 1.5)
            features['away_away_strength'] = away_stats.get('avg_away_points', 1.0)
            features['venue_advantage'] = features['home_home_strength'] - features['away_away_strength']
            
            # 6. FORM AND MOMENTUM
            features['home_form_trend'] = home_stats.get('form_trend', 0.0)
            features['away_form_trend'] = away_stats.get('form_trend', 0.0)
            features['momentum_diff'] = features['home_form_trend'] - features['away_form_trend']
            
            # 7. CONSISTENCY AND RELIABILITY
            features['home_consistency'] = home_stats.get('consistency', 1.0)
            features['away_consistency'] = away_stats.get('consistency', 1.0)
            features['consistency_advantage'] = features['home_consistency'] - features['away_consistency']
            
            # 8. COMPOSITE INDICATORS
            # Overall home advantage (multiple factors)
            features['total_home_advantage'] = (
                features['venue_advantage'] * 0.4 +
                features['power_score_diff'] * 0.3 +
                features['goal_diff_advantage'] * 0.2 +
                features['momentum_diff'] * 0.1
            )
            
            # Draw likelihood indicators
            features['strength_balance'] = 1.0 / (abs(features['power_score_diff']) + 0.1)
            features['form_balance'] = 1.0 / (abs(features['momentum_diff']) + 0.1)
            features['draw_likelihood'] = (features['strength_balance'] + features['form_balance']) / 2
            
            # Surprise factor
            features['surprise_factor'] = features['upset_potential'] + features['tier_mismatch'] / 5.0
            
            features_list.append(list(features.values()))
            
            # Label extraction for training
            if 'score' in match_data:
                score = match_data.get('score', {}).get('fullTime', {})
                home_score = int(score.get('home', 0)) if score.get('home') else 0
                away_score = int(score.get('away', 0)) if score.get('away') else 0
                
                if home_score > away_score:
                    labels.append('1')  # Home win
                elif home_score < away_score:
                    labels.append('2')  # Away win  
                else:
                    labels.append('X')  # Draw
        
        self.feature_names = list(features.keys())
        return np.array(features_list), np.array(labels) if labels else None
    
    def train_ultra_smart_model(self, train_data):
        """Ultra akıllı ensemble GP modelini eğit"""
        print("🧠 Ultra Smart GP Modeli Eğitiliyor...")
        
        # Feature extraction
        X, y = self.create_ultra_smart_features(train_data)
        
        print(f"📊 Feature sayısı: {X.shape[1]}")
        print(f"📊 Eğitim veri sayısı: {X.shape[0]}")
        
        # Feature selection (en iyi 20 feature)
        self.feature_selector = SelectKBest(mutual_info_classif, k=min(20, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Robust scaling
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Label encoding
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Advanced GP ensemble with optimized kernels
        gp1 = GaussianProcessClassifier(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-5),
            random_state=42,
            n_restarts_optimizer=3
        )
        
        gp2 = GaussianProcessClassifier(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-5),
            random_state=42,
            n_restarts_optimizer=3
        )
        
        gp3 = GaussianProcessClassifier(
            kernel=ConstantKernel(0.5) * RBF(0.5) * Matern(1.0, nu=2.5) + WhiteKernel(1e-5),
            random_state=42,
            n_restarts_optimizer=3
        )
        
        # Weighted ensemble (GP1 daha ağır çünkü RBF genelde daha iyi)
        self.ensemble_model = VotingClassifier([
            ('gp1', gp1),
            ('gp2', gp2), 
            ('gp3', gp3)
        ], voting='soft')
        
        # Train ensemble
        self.ensemble_model.fit(X_scaled, y_encoded)
        
        # Cross validation
        cv_scores = cross_val_score(self.ensemble_model, X_scaled, y_encoded, cv=5, scoring='accuracy')
        
        print(f"✅ Ultra Smart Model eğitildi")
        print(f"📊 CV Ortalama: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"📊 En iyi CV: {cv_scores.max():.3f}")
        
        # Feature importance (approximate)
        selected_features = self.feature_selector.get_support()
        important_features = [self.feature_names[i] for i, selected in enumerate(selected_features) if selected]
        print(f"🔍 Seçilen önemli feature'lar: {len(important_features)}")
        
        self.is_trained = True
        return cv_scores.mean()
    
    def predict_with_ultra_intelligence(self, test_data):
        """Ultra akıllı tahmin sistemi"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        results = []
        
        for match in test_data:
            X, _ = self.create_ultra_smart_features([match])
            
            # Feature selection and scaling
            X_selected = self.feature_selector.transform(X)
            X_scaled = self.scaler.transform(X_selected)
            
            # Ensemble prediction
            probabilities = self.ensemble_model.predict_proba(X_scaled)[0]
            predicted_class = self.ensemble_model.predict(X_scaled)[0]
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Confidence measures
            confidence = np.max(probabilities)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            uncertainty = entropy / np.log(len(probabilities))
            
            # Risk assessment
            prob_dict = {label: prob for label, prob in zip(self.label_encoder.classes_, probabilities)}
            
            # Actual result
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            if home_score > away_score:
                actual = '1'
            elif home_score == away_score:
                actual = 'X'
            else:
                actual = '2'
            
            # Smart analysis
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            home_rank = self.team_rankings.get(home_team, {}).get('rank', 99)
            away_rank = self.team_rankings.get(away_team, {}).get('rank', 99)
            
            result = {
                'match': f"{home_team} vs {away_team}",
                'predicted': predicted_label,
                'actual': actual,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'probabilities': prob_dict,
                'is_correct': predicted_label == actual,
                'home_score': home_score,
                'away_score': away_score,
                'home_rank': home_rank,
                'away_rank': away_rank,
                'rank_diff': away_rank - home_rank,
                'is_upset': (predicted_label == '2' and home_rank < away_rank) or 
                           (predicted_label == '1' and away_rank < home_rank)
            }
            
            results.append(result)
        
        return results

def test_ultra_smart_system():
    """Ultra akıllı sistem testi"""
    predictor = UltraSmartGP()
    
    # Data yükle ve analiz et
    train_data, test_data = predictor.load_and_analyze_data('data/ALM_stat.json', 11)
    
    # Ultra smart model eğit
    cv_score = predictor.train_ultra_smart_model(train_data)
    
    # Ultra intelligent predictions
    results = predictor.predict_with_ultra_intelligence(test_data)
    
    # Sonuçları analiz et
    correct = sum(r['is_correct'] for r in results)
    total = len(results)
    accuracy = correct / total * 100
    
    print(f"\n🎯 ULTRA SMART SONUÇ RAPORU:")
    print(f"Doğru tahmin: {correct}/{total}")
    print(f"Başarı oranı: {accuracy:.1f}%")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['is_correct'] else "❌"
        conf_str = f"{result['confidence']:.1%}"
        
        print(f"{status} {result['match']}")
        print(f"   Tahmin: {result['predicted']} ({conf_str}) | Gerçek: {result['actual']}")
        print(f"   Skor: {result['home_score']}-{result['away_score']}")
        print(f"   Sıralama: {result['home_rank']} vs {result['away_rank']} (fark: {result['rank_diff']})")
        
        if result.get('is_upset'):
            print(f"   🚨 SÜRPRIZ TAHMİNİ!")
        print()
    
    if accuracy == 100:
        print("🏆 MÜKEMMEL! %100 ULTRA SMART BAŞARI!")
    else:
        # Failure analysis
        failures = [r for r in results if not r['is_correct']]
        print(f"\n🔍 İYİLEŞTİRME ANALİZİ ({len(failures)} hata):")
        
        for failure in failures:
            print(f"❌ {failure['match']}")
            print(f"   Problem: {failure['predicted']} tahmin, {failure['actual']} gerçek")
            print(f"   Güven: {failure['confidence']:.1%} | Belirsizlik: {failure['uncertainty']:.3f}")
            print(f"   Sıralama farkı: {failure['rank_diff']}")
            
            # Specific recommendations
            if failure['confidence'] > 0.7:
                print(f"   💡 Yüksek güvenli hata - feature engineering gözden geçir")
            if abs(failure['rank_diff']) > 5:
                print(f"   💡 Büyük sıralama farkı - sürpriz faktörü güçlendir")
        
        print(f"\n🎯 ÖNERİLER:")
        print("1. Başarısız tahminlerin özel pattern'lerini analiz et")
        print("2. Takım-spesifik feature'lar ekle")
        print("3. Ensemble ağırlıklarını optimize et")
        print("4. Temporal decay fonksiyonunu ayarla")
    
    return predictor, results

if __name__ == "__main__":
    test_ultra_smart_system()
