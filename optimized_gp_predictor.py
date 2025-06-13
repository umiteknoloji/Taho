#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gelişmiş GP Tabanlı Futbol Tahmin Sistemi
Gerçek veri yapısı ile çalışan optimized GP system
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class OptimizedGPPredictor:
    def __init__(self):
        self.gp_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def load_and_parse_data(self, data_files=None):
        """Gerçek veri formatını parse et"""
        print("📥 Futbol verileri yükleniyor ve parse ediliyor...")
        
        if data_files is None:
            data_files = ['data/TR_stat.json', 'data/ENG_stat.json']
        
        all_matches = []
        
        for file_path in data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                league_name = file_path.split('/')[-1].replace('_stat.json', '')
                
                for match in data:
                    parsed_match = self._parse_match(match, league_name)
                    if parsed_match:
                        all_matches.append(parsed_match)
                
                print(f"  ✅ {league_name}: {len(data)} maç yüklendi")
                
            except Exception as e:
                print(f"  ❌ {file_path}: {e}")
        
        return pd.DataFrame(all_matches)
    
    def _parse_match(self, match, league):
        """Tek maçı parse et"""
        try:
            # Skorları çıkar
            score_data = match.get('score', {})
            full_time = score_data.get('fullTime', {})
            home_score = int(full_time.get('home', 0))
            away_score = int(full_time.get('away', 0))
            
            # Maç sonucunu hesapla
            if home_score > away_score:
                result = '1'
            elif home_score == away_score:
                result = 'X'
            else:
                result = '2'
            
            # Statistikleri parse et
            stats = match.get('stats', {})
            
            parsed = {
                'league': league,
                'week': int(match.get('week', 0)),
                'date': match.get('date', ''),
                'home_team': match.get('home', ''),
                'away_team': match.get('away', ''),
                'home_score': home_score,
                'away_score': away_score,
                'total_goals': home_score + away_score,
                'goal_difference': abs(home_score - away_score),
                'result': result,
                'attendance': match.get('attendance', 0)
            }
            
            # İstatistikleri ekle
            for stat_name, stat_data in stats.items():
                if isinstance(stat_data, dict):
                    home_val = self._parse_stat_value(stat_data.get('home', '0'))
                    away_val = self._parse_stat_value(stat_data.get('away', '0'))
                    
                    stat_key = stat_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                    parsed[f'home_{stat_key}'] = home_val
                    parsed[f'away_{stat_key}'] = away_val
                    parsed[f'diff_{stat_key}'] = home_val - away_val
            
            return parsed
            
        except Exception as e:
            print(f"    ⚠️ Match parse hatası: {e}")
            return None
    
    def _parse_stat_value(self, value):
        """İstatistik değerini sayıya çevir"""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Yüzde işaretini kaldır
            value = value.replace('%', '')
            
            # Slash notation (örn: "8/39") 
            if '/' in value:
                parts = value.split('/')
                if len(parts) == 2 and parts[1] != '0':
                    return float(parts[0]) / float(parts[1])
                else:
                    return float(parts[0]) if parts[0].isdigit() else 0
            
            # Normal sayı
            try:
                return float(value)
            except:
                return 0
        
        return 0
    
    def create_advanced_features(self, df):
        """Gelişmiş özellik mühendisliği"""
        print("🔧 Gelişmiş özellik mühendisliği...")
        
        # Sort by league, then by week
        df = df.sort_values(['league', 'week']).reset_index(drop=True)
        
        # 1. Rolling averages (form)
        df = self._add_rolling_features(df)
        
        # 2. Head-to-head features
        df = self._add_h2h_features(df)
        
        # 3. League context features  
        df = self._add_league_context(df)
        
        # 4. Match context features
        df = self._add_match_context(df)
        
        # 5. Statistical ratios
        df = self._add_statistical_ratios(df)
        
        return df
    
    def _add_rolling_features(self, df):
        """Rolling ortalamalar (form)"""
        windows = [3, 5]
        
        for window in windows:
            # Home team rolling stats
            for stat in ['home_score', 'away_score', 'total_goals']:
                df[f'home_{stat}_avg_{window}'] = df.groupby('home_team')[stat].rolling(window).mean().reset_index(drop=True)
            
            # Away team rolling stats  
            for stat in ['home_score', 'away_score', 'total_goals']:
                df[f'away_{stat}_avg_{window}'] = df.groupby('away_team')[stat].rolling(window).mean().reset_index(drop=True)
        
        # Points form (last 5 games)
        df['home_points_form'] = df.groupby('home_team').apply(
            lambda x: x['result'].map({'1': 3, 'X': 1, '2': 0}).rolling(5).sum()
        ).reset_index(drop=True)
        
        df['away_points_form'] = df.groupby('away_team').apply(
            lambda x: x['result'].map({'2': 3, 'X': 1, '1': 0}).rolling(5).sum()
        ).reset_index(drop=True)
        
        return df
    
    def _add_h2h_features(self, df):
        """Head-to-head özellikler"""
        df['h2h_home_wins'] = 0
        df['h2h_draws'] = 0
        df['h2h_away_wins'] = 0
        df['h2h_avg_goals'] = 0
        
        for idx, row in df.iterrows():
            h2h_data = self._get_h2h_history(df, row['home_team'], row['away_team'], idx)
            df.at[idx, 'h2h_home_wins'] = h2h_data['home_wins']
            df.at[idx, 'h2h_draws'] = h2h_data['draws'] 
            df.at[idx, 'h2h_away_wins'] = h2h_data['away_wins']
            df.at[idx, 'h2h_avg_goals'] = h2h_data['avg_goals']
        
        return df
    
    def _add_league_context(self, df):
        """Lig konteksti özellikleri"""
        # League averages
        df['league_avg_goals'] = df.groupby('league')['total_goals'].transform('mean')
        df['league_avg_home_score'] = df.groupby('league')['home_score'].transform('mean')
        
        # Team strength vs league
        df['home_strength_vs_league'] = (
            df.groupby(['league', 'home_team'])['home_score'].transform('mean') - 
            df['league_avg_home_score']
        )
        
        df['away_strength_vs_league'] = (
            df.groupby(['league', 'away_team'])['away_score'].transform('mean') - 
            df.groupby('league')['away_score'].transform('mean')
        )
        
        return df
    
    def _add_match_context(self, df):
        """Maç konteksti özellikleri"""
        # Goal expectation based on form
        df['expected_home_goals'] = df.get('home_home_score_avg_5', 0)
        df['expected_away_goals'] = df.get('away_away_score_avg_5', 0) 
        df['expected_total'] = df['expected_home_goals'] + df['expected_away_goals']
        
        # Form difference
        df['form_difference'] = df.get('home_points_form', 0) - df.get('away_points_form', 0)
        
        return df
    
    def _add_statistical_ratios(self, df):
        """İstatistiksel oranlar"""
        # Attack/Defense ratios
        if 'home_home_score_avg_5' in df.columns and 'home_away_score_avg_5' in df.columns:
            df['home_attack_defense_ratio'] = (df['home_home_score_avg_5'] + 0.1) / (df['home_away_score_avg_5'] + 0.1)
            df['away_attack_defense_ratio'] = (df['away_away_score_avg_5'] + 0.1) / (df['away_home_score_avg_5'] + 0.1)
        
        return df
    
    def _get_h2h_history(self, df, home_team, away_team, current_idx, max_games=5):
        """Head-to-head geçmiş"""
        h2h_matches = df[
            ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
            ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ]
        h2h_matches = h2h_matches[h2h_matches.index < current_idx].tail(max_games)
        
        if len(h2h_matches) == 0:
            return {'home_wins': 0, 'draws': 0, 'away_wins': 0, 'avg_goals': 0}
        
        home_wins = draws = away_wins = 0
        total_goals = []
        
        for _, match in h2h_matches.iterrows():
            total_goals.append(match['total_goals'])
            
            if match['home_team'] == home_team:
                if match['result'] == '1':
                    home_wins += 1
                elif match['result'] == 'X':
                    draws += 1
                else:
                    away_wins += 1
            else:  # away team is home in this match
                if match['result'] == '2':
                    home_wins += 1
                elif match['result'] == 'X':
                    draws += 1
                else:
                    away_wins += 1
        
        return {
            'home_wins': home_wins,
            'draws': draws, 
            'away_wins': away_wins,
            'avg_goals': np.mean(total_goals) if total_goals else 0
        }
    
    def optimize_gp_model(self, X, y):
        """GP modelini optimize et"""
        print("🎯 GP model optimizasyonu...")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Test different kernels
        kernels = {
            'RBF': ConstantKernel(1.0) * RBF(length_scale=1.0),
            'Matern_2.5': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
            'RBF_Noise': ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3),
            'Football_Optimized': (
                ConstantKernel(1.0) * RBF(length_scale=1.0) +
                ConstantKernel(0.5) * Matern(length_scale=0.5, nu=1.5) +
                WhiteKernel(1e-3)
            )
        }
        
        best_kernel = None
        best_score = 0
        
        for name, kernel in kernels.items():
            try:
                gp = GaussianProcessClassifier(
                    kernel=kernel,
                    random_state=42,
                    n_restarts_optimizer=2,
                    max_iter_predict=100
                )
                
                scores = cross_val_score(gp, X_scaled, y_encoded, cv=3, scoring='accuracy')
                avg_score = scores.mean()
                
                print(f"  {name}: %{avg_score*100:.1f} ± {scores.std()*100:.1f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_kernel = kernel
                    
            except Exception as e:
                print(f"  {name}: Hata - {str(e)[:30]}")
        
        # Train final model with best kernel
        self.gp_model = GaussianProcessClassifier(
            kernel=best_kernel,
            random_state=42,
            n_restarts_optimizer=3,
            max_iter_predict=200
        )
        
        self.gp_model.fit(X_scaled, y_encoded)
        
        print(f"✅ En iyi model: %{best_score*100:.1f} CV doğruluk")
        return best_score
    
    def predict_with_confidence(self, X):
        """Güvenli tahmin"""
        X_scaled = self.scaler.transform(X)
        
        # Predictions and probabilities
        probabilities = self.gp_model.predict_proba(X_scaled)
        predictions = self.gp_model.predict(X_scaled)
        
        # Enhanced confidence calculation
        confidences = []
        for prob in probabilities:
            max_prob = np.max(prob)
            entropy = -np.sum(prob * np.log(prob + 1e-8))
            normalized_entropy = entropy / np.log(len(prob))
            confidence = max_prob * (1 - normalized_entropy * 0.5)
            confidences.append(confidence * 100)
        
        # Decode predictions
        predictions_decoded = self.label_encoder.inverse_transform(predictions)
        
        return predictions_decoded, probabilities, confidences
    
    def evaluate_model(self, X, y):
        """Model değerlendirmesi"""
        print("\n📊 MODEL DEĞERLENDIRMESI")
        print("="*40)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Model eğitimi
        best_score = self.optimize_gp_model(X_train, y_train)
        
        # Test predictions
        predictions, probabilities, confidences = self.predict_with_confidence(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"\n🎯 Test Doğruluğu: %{accuracy*100:.1f}")
        
        # High confidence accuracy
        high_conf_indices = [i for i, conf in enumerate(confidences) if conf >= 70]
        if high_conf_indices:
            high_conf_pred = [predictions[i] for i in high_conf_indices]
            high_conf_actual = [y_test.iloc[i] for i in high_conf_indices]
            high_conf_acc = accuracy_score(high_conf_actual, high_conf_pred)
            
            print(f"🔒 Yüksek Güvenli (%70+): %{high_conf_acc*100:.1f}")
            print(f"📈 Yüksek Güvenli Oran: {len(high_conf_indices)}/{len(predictions)} (%{len(high_conf_indices)/len(predictions)*100:.1f})")
        
        # Classification report
        print(f"\n📋 Detaylı Rapor:")
        print(classification_report(y_test, predictions))
        
        return {
            'cv_score': best_score,
            'test_accuracy': accuracy,
            'high_conf_accuracy': high_conf_acc if high_conf_indices else 0,
            'high_conf_ratio': len(high_conf_indices)/len(predictions) if high_conf_indices else 0
        }

def main():
    """Ana fonksiyon"""
    predictor = OptimizedGPPredictor()
    
    try:
        # Veriyi yükle
        df = predictor.load_and_parse_data(['data/TR_stat.json'])
        
        if len(df) == 0:
            print("❌ Veri yüklenemedi!")
            return
        
        print(f"\n📊 Yüklenen veri: {len(df)} maç")
        print(f"🎯 Sonuç dağılımı: {df['result'].value_counts().to_dict()}")
        
        # Gelişmiş özellikler ekle
        enhanced_df = predictor.create_advanced_features(df)
        
        # Özellikleri hazırla
        feature_columns = [col for col in enhanced_df.columns 
                          if col not in ['home_team', 'away_team', 'result', 'league', 'date']]
        
        X = enhanced_df[feature_columns].fillna(0)
        y = enhanced_df['result']
        
        print(f"🔧 Özellik sayısı: {len(feature_columns)}")
        
        # Model değerlendirmesi
        results = predictor.evaluate_model(X, y)
        
        print(f"\n✅ ANALİZ TAMAMLANDI!")
        print(f"📈 En İyi CV Skoru: %{results['cv_score']*100:.1f}")
        print(f"🎯 Test Doğruluğu: %{results['test_accuracy']*100:.1f}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
