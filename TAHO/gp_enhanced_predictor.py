#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GP Tabanlı Gelişmiş Tahmin Sistemi
Bu sistem GP'yi ana model olarak kullanır ve doğruluğu artırmak için
gelişmiş özellik mühendisliği ve optimizasyonlar ekler.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, RationalQuadratic
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

class GPEnhancedPredictor:
    def __init__(self):
        self.gp_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        self.confidence_threshold = 0.6
        
    def create_advanced_features(self, df):
        """GP için optimize edilmiş gelişmiş özellikler"""
        print("🔧 GP için gelişmiş özellik mühendisliği...")
        
        # 1. Form-based features (GP için çok önemli)
        df = self._add_weighted_form_features(df)
        
        # 2. Head-to-head with decay (GP için temporal pattern)
        df = self._add_decayed_h2h_features(df)
        
        # 3. League context normalization
        df = self._add_normalized_league_features(df)
        
        # 4. Momentum and trend features
        df = self._add_momentum_features(df)
        
        # 5. GP-specific statistical features
        df = self._add_gp_statistical_features(df)
        
        return df
    
    def _add_weighted_form_features(self, df):
        """Ağırlıklı form özellikleri (son maçlar daha önemli)"""
        # Exponential weighted averages
        weights_3 = np.array([0.5, 0.3, 0.2])  # Son 3 maç
        weights_5 = np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # Son 5 maç
        
        # Home team form
        df['home_weighted_goals_3'] = self._calculate_weighted_average(df, 'home_team', 'home_score', weights_3)
        df['home_weighted_goals_5'] = self._calculate_weighted_average(df, 'home_team', 'home_score', weights_5)
        df['home_weighted_conceded_3'] = self._calculate_weighted_average(df, 'home_team', 'away_score', weights_3)
        
        # Away team form
        df['away_weighted_goals_3'] = self._calculate_weighted_average(df, 'away_team', 'away_score', weights_3)
        df['away_weighted_goals_5'] = self._calculate_weighted_average(df, 'away_team', 'away_score', weights_5)
        df['away_weighted_conceded_3'] = self._calculate_weighted_average(df, 'away_team', 'home_score', weights_3)
        
        # Form difference (GP için kritik)
        df['goal_difference_form_3'] = df['home_weighted_goals_3'] - df['away_weighted_goals_3']
        df['defensive_form_diff_3'] = df['away_weighted_conceded_3'] - df['home_weighted_conceded_3']
        
        return df
    
    def _add_decayed_h2h_features(self, df):
        """Zamanla azalan head-to-head özellikler"""
        df['h2h_home_advantage'] = 0.0
        df['h2h_goal_difference'] = 0.0
        df['h2h_over_under_pattern'] = 0.0
        
        for idx, row in df.iterrows():
            h2h_data = self._get_decayed_h2h(df, row['home_team'], row['away_team'], idx)
            df.at[idx, 'h2h_home_advantage'] = h2h_data['home_advantage']
            df.at[idx, 'h2h_goal_difference'] = h2h_data['goal_diff']
            df.at[idx, 'h2h_over_under_pattern'] = h2h_data['over_under']
        
        return df
    
    def _add_normalized_league_features(self, df):
        """Lig normalizedözellikleri"""
        # League-specific normalization
        df['home_goals_vs_league'] = df.groupby('league')['home_score'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        df['away_goals_vs_league'] = df.groupby('league')['away_score'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        # Team strength relative to league average
        df['home_team_strength'] = df.groupby(['league', 'home_team'])['home_score'].transform('mean') - \
                                   df.groupby('league')['home_score'].transform('mean')
        df['away_team_strength'] = df.groupby(['league', 'away_team'])['away_score'].transform('mean') - \
                                   df.groupby('league')['away_score'].transform('mean')
        
        # Relative strength difference (GP için çok güçlü feature)
        df['strength_difference'] = df['home_team_strength'] - df['away_team_strength']
        
        return df
    
    def _add_momentum_features(self, df):
        """Momentum ve trend özellikleri"""
        # Goal scoring momentum
        df['home_scoring_momentum'] = df.groupby('home_team')['home_score'].rolling(3).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 3 else 0
        ).reset_index(drop=True)
        
        df['away_scoring_momentum'] = df.groupby('away_team')['away_score'].rolling(3).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 3 else 0
        ).reset_index(drop=True)
        
        # Points form (3 points for win, 1 for draw)
        df['home_points_form'] = self._calculate_points_form(df, 'home')
        df['away_points_form'] = self._calculate_points_form(df, 'away')
        
        return df
    
    def _add_gp_statistical_features(self, df):
        """GP için optimize edilmiş istatistiksel özellikler"""
        # Goal variance (consistency indicator)
        df['home_goal_variance'] = df.groupby('home_team')['home_score'].rolling(5).var().reset_index(drop=True)
        df['away_goal_variance'] = df.groupby('away_team')['away_score'].rolling(5).var().reset_index(drop=True)
        
        # Attacking/Defensive balance
        df['home_attack_defense_ratio'] = (df['home_weighted_goals_3'] + 1) / (df['home_weighted_conceded_3'] + 1)
        df['away_attack_defense_ratio'] = (df['away_weighted_goals_3'] + 1) / (df['away_weighted_conceded_3'] + 1)
        
        # Match context features
        df['expected_total_goals'] = df['home_weighted_goals_3'] + df['away_weighted_goals_3']
        df['expected_goal_difference'] = df['home_weighted_goals_3'] - df['away_weighted_goals_3']
        
        return df
    
    def optimize_gp_kernels(self, X, y):
        """GP kernel optimizasyonu"""
        print("🎯 GP Kernel optimizasyonu başlıyor...")
        
        # Farklı kernel kombinasyonları
        kernels = {
            'rbf_optimized': ConstantKernel(1.0) * RBF(length_scale=1.0),
            'matern_52': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
            'matern_32': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
            'rbf_plus_noise': ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3),
            'football_special': (ConstantKernel(1.0) * RBF(1.0) + 
                                ConstantKernel(0.5) * Matern(nu=1.5) + 
                                WhiteKernel(1e-3)),
            'multi_scale': (ConstantKernel(1.0) * RBF(length_scale=1.0) +
                           ConstantKernel(0.5) * RBF(length_scale=0.1) +
                           WhiteKernel(1e-3))
        }
        
        best_kernel = None
        best_score = 0
        kernel_scores = {}
        
        for name, kernel in kernels.items():
            try:
                gp = GaussianProcessClassifier(
                    kernel=kernel,
                    random_state=42,
                    n_restarts_optimizer=3,
                    max_iter_predict=200
                )
                
                scores = cross_val_score(gp, X, y, cv=3, scoring='accuracy')
                avg_score = scores.mean()
                kernel_scores[name] = avg_score
                
                print(f"  {name}: %{avg_score*100:.1f} ± {scores.std()*100:.1f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_kernel = kernel
                    
            except Exception as e:
                print(f"  {name}: Hata - {str(e)[:50]}")
                kernel_scores[name] = 0
        
        print(f"\n🏆 En iyi kernel: %{best_score*100:.1f} doğruluk")
        return best_kernel, kernel_scores
    
    def train_enhanced_gp(self, X, y):
        """Gelişmiş GP eğitimi"""
        print("🚀 Gelişmiş GP eğitimi başlıyor...")
        
        # Veriyi normalize et
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Optimal kernel bul
        best_kernel, kernel_scores = self.optimize_gp_kernels(X_scaled, y_encoded)
        
        # En iyi kernel ile final model
        self.gp_model = GaussianProcessClassifier(
            kernel=best_kernel,
            random_state=42,
            n_restarts_optimizer=5,
            max_iter_predict=300
        )
        
        # Eğitim
        self.gp_model.fit(X_scaled, y_encoded)
        
        # Cross-validation ile değerlendirme
        cv_scores = cross_val_score(self.gp_model, X_scaled, y_encoded, cv=5, scoring='accuracy')
        
        print(f"✅ GP Eğitimi Tamamlandı!")
        print(f"📊 Cross-validation: %{cv_scores.mean()*100:.1f} ± {cv_scores.std()*100:.1f}")
        
        return cv_scores.mean(), kernel_scores
    
    def predict_with_enhanced_confidence(self, X):
        """Gelişmiş güven skorları ile tahmin"""
        X_scaled = self.scaler.transform(X)
        
        # Probability predictions
        probabilities = self.gp_model.predict_proba(X_scaled)
        predictions = self.gp_model.predict(X_scaled)
        
        # Gelişmiş güven hesaplama
        confidences = []
        for prob in probabilities:
            # Maximum probability
            max_prob = np.max(prob)
            
            # Entropy-based uncertainty
            entropy = -np.sum(prob * np.log(prob + 1e-8))
            normalized_entropy = entropy / np.log(len(prob))
            
            # Confidence with entropy adjustment
            confidence = max_prob * (1 - normalized_entropy)
            
            # Additional adjustment for close probabilities
            prob_spread = np.max(prob) - np.min(prob)
            confidence = confidence * (0.5 + 0.5 * prob_spread)
            
            confidences.append(confidence * 100)
        
        # Decode predictions
        predictions_decoded = self.label_encoder.inverse_transform(predictions)
        
        return predictions_decoded, probabilities, confidences
    
    def filter_high_confidence_predictions(self, predictions, probabilities, confidences, threshold=None):
        """Yüksek güvenli tahminleri filtrele"""
        if threshold is None:
            threshold = self.confidence_threshold * 100
        
        high_conf_indices = [i for i, conf in enumerate(confidences) if conf >= threshold]
        
        if high_conf_indices:
            filtered_predictions = [predictions[i] for i in high_conf_indices]
            filtered_probabilities = [probabilities[i] for i in high_conf_indices]
            filtered_confidences = [confidences[i] for i in high_conf_indices]
            
            print(f"🎯 Yüksek güvenli tahminler: {len(filtered_predictions)}/{len(predictions)}")
            print(f"📈 Ortalama güven: %{np.mean(filtered_confidences):.1f}")
            
            return filtered_predictions, filtered_probabilities, filtered_confidences, high_conf_indices
        else:
            print(f"⚠️ Eşik %{threshold} üzerinde tahmin yok!")
            return [], [], [], []
    
    def _calculate_weighted_average(self, df, team_col, score_col, weights):
        """Ağırlıklı ortalama hesapla"""
        result = []
        for idx, row in df.iterrows():
            team = row[team_col]
            team_matches = df[(df[team_col] == team) & (df.index < idx)]
            
            if len(team_matches) >= len(weights):
                recent_scores = team_matches[score_col].tail(len(weights)).values
                weighted_avg = np.average(recent_scores, weights=weights)
            else:
                weighted_avg = team_matches[score_col].mean() if len(team_matches) > 0 else 0
            
            result.append(weighted_avg)
        
        return result
    
    def _get_decayed_h2h(self, df, home_team, away_team, current_idx):
        """Zamanla azalan H2H verileri"""
        h2h_matches = df[
            ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
            ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ]
        h2h_matches = h2h_matches[h2h_matches.index < current_idx].tail(5)
        
        if len(h2h_matches) == 0:
            return {'home_advantage': 0, 'goal_diff': 0, 'over_under': 0}
        
        # Time decay weights (more recent = higher weight)
        weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05])[:len(h2h_matches)]
        weights = weights / weights.sum()
        
        home_advantages = []
        goal_diffs = []
        total_goals = []
        
        for _, match in h2h_matches.iterrows():
            if match['home_team'] == home_team:
                home_advantage = 1 if match['result'] == '1' else (0.5 if match['result'] == 'X' else 0)
                goal_diff = match['home_score'] - match['away_score']
            else:
                home_advantage = 1 if match['result'] == '2' else (0.5 if match['result'] == 'X' else 0)
                goal_diff = match['away_score'] - match['home_score']
            
            home_advantages.append(home_advantage)
            goal_diffs.append(goal_diff)
            total_goals.append(match['home_score'] + match['away_score'])
        
        weighted_home_adv = np.average(home_advantages, weights=weights)
        weighted_goal_diff = np.average(goal_diffs, weights=weights)
        weighted_total_goals = np.average(total_goals, weights=weights)
        
        return {
            'home_advantage': weighted_home_adv,
            'goal_diff': weighted_goal_diff,
            'over_under': weighted_total_goals
        }
    
    def _calculate_points_form(self, df, venue):
        """Puan formu hesapla"""
        result = []
        for idx, row in df.iterrows():
            if venue == 'home':
                team = row['home_team']
                team_matches = df[(df['home_team'] == team) & (df.index < idx)]
                points = team_matches['result'].map({'1': 3, 'X': 1, '2': 0}).tail(5).sum()
            else:
                team = row['away_team']
                team_matches = df[(df['away_team'] == team) & (df.index < idx)]
                points = team_matches['result'].map({'2': 3, 'X': 1, '1': 0}).tail(5).sum()
            
            result.append(points)
        
        return result

def main():
    """Test the enhanced GP predictor"""
    predictor = GPEnhancedPredictor()
    
    try:
        # Load test data
        with open('data/combined_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        all_matches = []
        for league, matches in data.items():
            for match in matches:
                match['league'] = league
                all_matches.append(match)
        
        df = pd.DataFrame(all_matches)
        print(f"📊 Toplam maç sayısı: {len(df)}")
        
        # Create enhanced features
        enhanced_df = predictor.create_advanced_features(df)
        
        # Prepare features and target
        feature_columns = [col for col in enhanced_df.columns 
                          if col not in ['home_team', 'away_team', 'result', 'league', 'date']]
        
        X = enhanced_df[feature_columns].fillna(0)
        y = enhanced_df['result'].fillna('X')
        
        print(f"🔧 Özellik sayısı: {len(feature_columns)}")
        print(f"🎯 Hedef dağılımı: {y.value_counts().to_dict()}")
        
        # Train enhanced GP
        cv_score, kernel_scores = predictor.train_enhanced_gp(X, y)
        
        # Test predictions
        test_X = X.tail(20)  # Son 20 maç
        predictions, probabilities, confidences = predictor.predict_with_enhanced_confidence(test_X)
        
        print(f"\n🔮 ÖRNEK TAHMİNLER:")
        for i in range(min(10, len(predictions))):
            print(f"  Tahmin {i+1}: {predictions[i]} (Güven: %{confidences[i]:.1f})")
            if len(probabilities[i]) == 3:
                print(f"    Olasılıklar: 1=%{probabilities[i][0]*100:.1f}, X=%{probabilities[i][1]*100:.1f}, 2=%{probabilities[i][2]*100:.1f}")
        
        # High confidence filtering
        high_conf_preds, high_conf_probs, high_conf_confs, indices = predictor.filter_high_confidence_predictions(
            predictions, probabilities, confidences, threshold=70
        )
        
        print(f"\n🏆 YÜKSEK GÜVENLİ TAHMİNLER (>%70):")
        for i, (pred, conf) in enumerate(zip(high_conf_preds, high_conf_confs)):
            print(f"  {i+1}. {pred} (%{conf:.1f})")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
