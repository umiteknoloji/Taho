#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GP Futbol Tahmin Sistemi - Doğruluk Odaklı
Analiz sonuçlarına göre optimize edilmiş GP classifier
"""

import json
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class AccuracyFocusedGP:
    def __init__(self):
        self.gp_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.selected_features = []
        
    def load_and_enhance_data(self, league_file='data/TR_stat.json'):
        """Veriyi yükle ve gelişmiş özellikler ekle"""
        print("📥 Veri yükleniyor ve özellikler geliştiriliyor...")
        
        with open(league_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        matches = []
        for match in data:
            try:
                # Basic parsing
                score_data = match.get('score', {})
                full_time = score_data.get('fullTime', {})
                home_score = int(full_time.get('home', 0))
                away_score = int(full_time.get('away', 0))
                
                if home_score > away_score:
                    result = '1'
                elif home_score == away_score:
                    result = 'X'
                else:
                    result = '2'
                
                # Enhanced features from stats
                stats = match.get('stats', {})
                parsed_match = {
                    'home_team': match.get('home', ''),
                    'away_team': match.get('away', ''),
                    'home_score': home_score,
                    'away_score': away_score,
                    'total_goals': home_score + away_score,
                    'goal_difference': home_score - away_score,
                    'abs_goal_diff': abs(home_score - away_score),
                    'result': result,
                    'week': int(match.get('week', 0)),
                    'attendance': match.get('attendance', 0)
                }
                
                # Parse detailed stats
                for stat_name, stat_data in stats.items():
                    if isinstance(stat_data, dict):
                        home_val = self._parse_stat_value(stat_data.get('home', '0'))
                        away_val = self._parse_stat_value(stat_data.get('away', '0'))
                        
                        # Clean stat name
                        clean_name = stat_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                        parsed_match[f'home_{clean_name}'] = home_val
                        parsed_match[f'away_{clean_name}'] = away_val
                        parsed_match[f'diff_{clean_name}'] = home_val - away_val
                        
                        # Ratio features (avoid division by zero)
                        if away_val != 0:
                            parsed_match[f'ratio_{clean_name}'] = home_val / away_val
                        else:
                            parsed_match[f'ratio_{clean_name}'] = home_val if home_val > 0 else 1.0
                
                matches.append(parsed_match)
                
            except Exception as e:
                print(f"  ⚠️ Match parse error: {e}")
                continue
        
        df = pd.DataFrame(matches)
        print(f"  ✅ Parsed {len(df)} matches")
        
        # Advanced feature engineering
        enhanced_df = self._create_advanced_features(df)
        
        return enhanced_df
    
    def _parse_stat_value(self, value):
        """İstatistik değerini parse et"""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            value = value.replace('%', '')
            
            # Handle ratio format (e.g., "8/39")
            if '/' in value:
                parts = value.split('/')
                if len(parts) == 2:
                    try:
                        numerator = float(parts[0])
                        denominator = float(parts[1])
                        return (numerator / denominator * 100) if denominator != 0 else numerator
                    except:
                        return 0
            
            try:
                return float(value)
            except:
                return 0
        
        return 0
    
    def _create_advanced_features(self, df):
        """Gelişmiş özellik mühendisliği"""
        print("  🔧 Gelişmiş özellik mühendisliği...")
        
        # Sort by week for time-based features
        df = df.sort_values(['week']).reset_index(drop=True)
        
        # 1. Team form features (rolling averages)
        self._add_team_form_features(df)
        
        # 2. Head-to-head features
        self._add_h2h_features(df)
        
        # 3. Match context features
        self._add_match_context_features(df)
        
        # 4. Statistical momentum features
        self._add_momentum_features(df)
        
        # 5. Derived features
        self._add_derived_features(df)
        
        print(f"  📊 Total features: {len(df.columns)}")
        return df
    
    def _add_team_form_features(self, df):
        """Takım formu özellikleri"""
        # Last 3 and 5 game averages
        for window in [3, 5]:
            # Home team stats when playing at home
            df[f'home_goals_avg_{window}'] = df.groupby('home_team')['home_score'].rolling(window, min_periods=1).mean().reset_index(drop=True)
            df[f'home_conceded_avg_{window}'] = df.groupby('home_team')['away_score'].rolling(window, min_periods=1).mean().reset_index(drop=True)
            
            # Away team stats when playing away
            df[f'away_goals_avg_{window}'] = df.groupby('away_team')['away_score'].rolling(window, min_periods=1).mean().reset_index(drop=True)
            df[f'away_conceded_avg_{window}'] = df.groupby('away_team')['home_score'].rolling(window, min_periods=1).mean().reset_index(drop=True)
        
        # Points form (last 5 games)
        df['home_points_form'] = 0
        df['away_points_form'] = 0
        
        for idx, row in df.iterrows():
            # Home team points (considering all games)
            home_recent = df[(df['home_team'] == row['home_team']) & (df.index < idx)].tail(5)
            home_points = sum(3 if r == '1' else (1 if r == 'X' else 0) for r in home_recent['result'])
            df.at[idx, 'home_points_form'] = home_points
            
            # Away team points (considering all games)
            away_recent = df[(df['away_team'] == row['away_team']) & (df.index < idx)].tail(5)
            away_points = sum(3 if r == '2' else (1 if r == 'X' else 0) for r in away_recent['result'])
            df.at[idx, 'away_points_form'] = away_points
    
    def _add_h2h_features(self, df):
        """Head-to-head özellikleri"""
        df['h2h_games'] = 0
        df['h2h_home_wins'] = 0
        df['h2h_draws'] = 0
        df['h2h_away_wins'] = 0
        df['h2h_avg_goals'] = 0
        
        for idx, row in df.iterrows():
            h2h = self._get_h2h_history(df, row['home_team'], row['away_team'], idx)
            df.at[idx, 'h2h_games'] = h2h['total']
            df.at[idx, 'h2h_home_wins'] = h2h['home_wins']
            df.at[idx, 'h2h_draws'] = h2h['draws']
            df.at[idx, 'h2h_away_wins'] = h2h['away_wins']
            df.at[idx, 'h2h_avg_goals'] = h2h['avg_goals']
    
    def _add_match_context_features(self, df):
        """Maç konteksti özellikleri"""
        # Goal expectation based on form
        df['expected_home_goals'] = df.get('home_goals_avg_5', 0).fillna(0)
        df['expected_away_goals'] = df.get('away_goals_avg_5', 0).fillna(0) 
        df['expected_total_goals'] = df['expected_home_goals'] + df['expected_away_goals']
        df['expected_goal_diff'] = df['expected_home_goals'] - df['expected_away_goals']
        
        # Form differences
        df['form_difference'] = df.get('home_points_form', 0) - df.get('away_points_form', 0)
        df['goal_form_diff'] = df.get('home_goals_avg_3', 0) - df.get('away_goals_avg_3', 0)
        
        # Defensive strength difference
        df['defensive_diff'] = df.get('away_conceded_avg_3', 0) - df.get('home_conceded_avg_3', 0)
    
    def _add_momentum_features(self, df):
        """Momentum özellikleri"""
        # Scoring trend (improving/declining)
        df['home_scoring_trend'] = 0
        df['away_scoring_trend'] = 0
        
        for idx, row in df.iterrows():
            # Home team scoring trend
            home_recent = df[(df['home_team'] == row['home_team']) & (df.index < idx)].tail(3)
            if len(home_recent) >= 2:
                home_trend = home_recent['home_score'].iloc[-1] - home_recent['home_score'].iloc[0]
                df.at[idx, 'home_scoring_trend'] = home_trend
            
            # Away team scoring trend
            away_recent = df[(df['away_team'] == row['away_team']) & (df.index < idx)].tail(3)
            if len(away_recent) >= 2:
                away_trend = away_recent['away_score'].iloc[-1] - away_recent['away_score'].iloc[0]
                df.at[idx, 'away_scoring_trend'] = away_trend
    
    def _add_derived_features(self, df):
        """Türetilmiş özellikler"""
        # Attack vs Defense ratios
        df['home_attack_defense_ratio'] = (df.get('home_goals_avg_3', 0) + 0.1) / (df.get('home_conceded_avg_3', 0) + 0.1)
        df['away_attack_defense_ratio'] = (df.get('away_goals_avg_3', 0) + 0.1) / (df.get('away_conceded_avg_3', 0) + 0.1)
        
        # Overall team strength (combination of attack and defense)
        df['home_strength'] = df.get('home_goals_avg_5', 0) - df.get('home_conceded_avg_5', 0)
        df['away_strength'] = df.get('away_goals_avg_5', 0) - df.get('away_conceded_avg_5', 0)
        df['strength_difference'] = df['home_strength'] - df['away_strength']
        
        # Match competitiveness indicators
        df['likely_close_match'] = ((abs(df.get('form_difference', 0)) <= 3) & 
                                   (abs(df.get('strength_difference', 0)) <= 0.5)).astype(int)
    
    def _get_h2h_history(self, df, home_team, away_team, current_idx, max_games=5):
        """Head-to-head geçmiş al"""
        h2h_matches = df[
            ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
            ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ]
        h2h_matches = h2h_matches[h2h_matches.index < current_idx].tail(max_games)
        
        if len(h2h_matches) == 0:
            return {'total': 0, 'home_wins': 0, 'draws': 0, 'away_wins': 0, 'avg_goals': 0}
        
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
            else:
                if match['result'] == '2':
                    home_wins += 1
                elif match['result'] == 'X':
                    draws += 1
                else:
                    away_wins += 1
        
        return {
            'total': len(h2h_matches),
            'home_wins': home_wins,
            'draws': draws,
            'away_wins': away_wins,
            'avg_goals': np.mean(total_goals)
        }
    
    def select_best_features(self, X, y, max_features=25):
        """En iyi özellikleri seç"""
        print("🎯 Özellik seçimi yapılıyor...")
        
        # Remove non-numeric columns and handle infinite values
        numeric_X = X.select_dtypes(include=[np.number])
        numeric_X = numeric_X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if len(numeric_X.columns) == 0:
            print("  ❌ Hiç sayısal özellik bulunamadı!")
            return numeric_X, []
        
        # Feature selection using univariate analysis
        selector = SelectKBest(score_func=f_classif, k=min(max_features, len(numeric_X.columns)))
        
        try:
            X_selected = selector.fit_transform(numeric_X, y)
            selected_feature_names = numeric_X.columns[selector.get_support()].tolist()
            
            print(f"  ✅ {len(selected_feature_names)} özellik seçildi")
            print("  📊 Seçilen özellikler:")
            for i, feature in enumerate(selected_feature_names[:10]):  # İlk 10'u göster
                score = selector.scores_[selector.get_support()][i]
                print(f"    {i+1:2d}. {feature:<30} (skor: {score:.2f})")
            
            self.feature_selector = selector
            self.selected_features = selected_feature_names
            
            return pd.DataFrame(X_selected, columns=selected_feature_names), selected_feature_names
            
        except Exception as e:
            print(f"  ⚠️ Feature selection hatası: {e}")
            return numeric_X.iloc[:, :max_features], numeric_X.columns[:max_features].tolist()
    
    def optimize_gp_hyperparameters(self, X, y):
        """GP hiperparametrelerini optimize et"""
        print("⚙️ GP hiperparametre optimizasyonu...")
        
        # Normalize data
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Define kernels to test
        kernels = [
            ConstantKernel(1.0) * RBF(length_scale=1.0),
            ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
            ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
            ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3),
        ]
        
        best_score = 0
        best_params = None
        
        for i, kernel in enumerate(kernels):
            try:
                gp = GaussianProcessClassifier(
                    kernel=kernel,
                    random_state=42,
                    n_restarts_optimizer=2,
                    max_iter_predict=100
                )
                
                # Cross-validation
                scores = cross_val_score(gp, X_scaled, y_encoded, cv=3, scoring='accuracy')
                avg_score = scores.mean()
                
                print(f"  Kernel {i+1}: %{avg_score*100:.1f} ± {scores.std()*100:.1f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {
                        'kernel': kernel,
                        'random_state': 42,
                        'n_restarts_optimizer': 3,
                        'max_iter_predict': 200
                    }
                    
            except Exception as e:
                print(f"  Kernel {i+1}: Hata - {str(e)[:30]}")
        
        if best_params:
            print(f"  🏆 En iyi CV skoru: %{best_score*100:.1f}")
            return best_params, best_score
        else:
            print("  ❌ Hiçbir kernel çalışmadı!")
            return None, 0
    
    def train_final_model(self, X, y):
        """Final modeli eğit"""
        print("\n🚀 FINAL MODEL EĞİTİMİ")
        print("="*40)
        
        # Feature selection
        X_selected, feature_names = self.select_best_features(X, y)
        
        if len(X_selected) == 0:
            print("❌ Özellik seçimi başarısız!")
            return None
        
        # Hyperparameter optimization
        best_params, best_cv_score = self.optimize_gp_hyperparameters(X_selected, y)
        
        if not best_params:
            print("❌ Hiperparametre optimizasyonu başarısız!")
            return None
        
        # Train final model
        X_scaled = self.scaler.fit_transform(X_selected)
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.gp_model = GaussianProcessClassifier(**best_params)
        self.gp_model.fit(X_scaled, y_encoded)
        
        print(f"✅ Model eğitimi tamamlandı!")
        print(f"📊 CV Doğruluk: %{best_cv_score*100:.1f}")
        
        return {
            'cv_score': best_cv_score,
            'feature_count': len(feature_names),
            'selected_features': feature_names
        }
    
    def evaluate_model(self, X, y):
        """Modeli değerlendir"""
        print("\n📊 MODEL DEĞERLENDIRMESI")
        print("="*40)
        
        # Feature selection (same as training)
        X_selected, _ = self.select_best_features(X, y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train model
        training_result = self.train_final_model(X_train, y_train)
        
        if not training_result:
            return None
        
        # Test predictions
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_encoded = self.gp_model.predict(X_test_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Test accuracy
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 TEST SONUÇLARI:")
        print(f"  Test Doğruluğu: %{test_accuracy*100:.1f}")
        
        # Confidence analysis
        y_proba = self.gp_model.predict_proba(X_test_scaled)
        confidences = [np.max(prob) * 100 for prob in y_proba]
        avg_confidence = np.mean(confidences)
        
        # High confidence predictions
        high_conf_threshold = 70
        high_conf_indices = [i for i, conf in enumerate(confidences) if conf >= high_conf_threshold]
        
        if high_conf_indices:
            high_conf_pred = [y_pred[i] for i in high_conf_indices]
            high_conf_actual = [y_test.iloc[i] for i in high_conf_indices]
            high_conf_acc = accuracy_score(high_conf_actual, high_conf_pred)
            
            print(f"  Yüksek Güvenli (%{high_conf_threshold}+): %{high_conf_acc*100:.1f}")
            print(f"  Yüksek Güvenli Oran: {len(high_conf_indices)}/{len(y_pred)} (%{len(high_conf_indices)/len(y_pred)*100:.1f})")
        else:
            high_conf_acc = 0
        
        print(f"  Ortalama Güven: %{avg_confidence:.1f}")
        
        # Classification report
        print(f"\n📋 DETAYLI RAPOR:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        print(f"\n🔍 CONFUSION MATRIX:")
        cm = confusion_matrix(y_test, y_pred, labels=['1', 'X', '2'])
        print("      Pred:  1   X   2")
        for i, label in enumerate(['1', 'X', '2']):
            print(f"Actual {label}: {cm[i]}")
        
        return {
            'cv_score': training_result['cv_score'],
            'test_accuracy': test_accuracy,
            'high_conf_accuracy': high_conf_acc,
            'high_conf_ratio': len(high_conf_indices)/len(y_pred) if high_conf_indices else 0,
            'avg_confidence': avg_confidence,
            'feature_count': training_result['feature_count'],
            'confusion_matrix': cm.tolist()
        }

def main():
    """Ana fonksiyon"""
    predictor = AccuracyFocusedGP()
    
    try:
        print("🎯 GP FUTBOL TAHMİN SİSTEMİ - DOĞRULUK ODAKLI")
        print("="*60)
        
        # Load and enhance data
        df = predictor.load_and_enhance_data('data/TR_stat.json')
        
        print(f"\n📊 GENEL İSTATİSTİKLER:")
        print(f"  Toplam maç: {len(df)}")
        print(f"  Sonuç dağılımı: {df['result'].value_counts().to_dict()}")
        print(f"  Toplam özellik: {len(df.columns)}")
        
        # Prepare features and target
        feature_columns = [col for col in df.columns 
                          if col not in ['home_team', 'away_team', 'result']]
        
        X = df[feature_columns]
        y = df['result']
        
        # Evaluate model
        results = predictor.evaluate_model(X, y)
        
        if results:
            print(f"\n🏆 FINAL SONUÇLAR:")
            print(f"  CV Doğruluk: %{results['cv_score']*100:.1f}")
            print(f"  Test Doğruluk: %{results['test_accuracy']*100:.1f}")
            print(f"  Yüksek Güvenli Doğruluk: %{results['high_conf_accuracy']*100:.1f}")
            print(f"  Kullanılan Özellik: {results['feature_count']}")
            
            # Performance assessment
            baseline = 46.7  # From analysis
            improvement = results['test_accuracy'] * 100 - baseline
            
            print(f"\n📈 PERFORMANS DEĞERLENDİRMESİ:")
            print(f"  Baseline: %{baseline:.1f}")
            print(f"  İyileşme: +%{improvement:.1f}")
            
            if improvement >= 15:
                print("  🟢 MÜKEMMEL: Çok başarılı model!")
            elif improvement >= 10:
                print("  🟢 İYİ: Başarılı model!")
            elif improvement >= 5:
                print("  🟡 ORTA: Kabul edilebilir iyileşme")
            else:
                print("  🔴 ZAYIF: Daha fazla iyileştirme gerekli")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
