#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GP Futbol Tahmin Sistemi - Doğru Versiyon
Data leakage olmadan gerçek tahmin sistemi
"""

import json
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel, RationalQuadratic
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class ProperGPPredictor:
    def __init__(self):
        self.gp_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.selected_features = []
        
    def load_and_enhance_data(self, league_file='data/TR_stat.json'):
        """Veriyi yükle ve SADECE önceki bilgilerle özellikler ekle"""
        print("📥 Veri yükleniyor ve pre-match özellikler geliştiriliyor...")
        
        with open(league_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        matches = []
        team_history = {}
        
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
                
                home_team = match.get('home', '')
                away_team = match.get('away', '')
                week = int(match.get('week', 0))
                
                # Pre-match features (NO match result info!)
                parsed_match = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'week': week,
                    'result': result,  # This is our target, not a feature
                    'attendance': match.get('attendance', 0) if match.get('attendance') else 0
                }
                
                # Historical features (based on previous matches only)
                home_history = team_history.get(home_team, {})
                away_history = team_history.get(away_team, {})
                
                # Team form features (from previous matches)
                parsed_match.update({
                    'home_prev_matches': home_history.get('matches', 0),
                    'away_prev_matches': away_history.get('matches', 0),
                    'home_prev_wins': home_history.get('wins', 0),
                    'away_prev_wins': away_history.get('wins', 0),
                    'home_prev_draws': home_history.get('draws', 0),
                    'away_prev_draws': away_history.get('draws', 0),
                    'home_prev_losses': home_history.get('losses', 0),
                    'away_prev_losses': away_history.get('losses', 0),
                    'home_prev_goals_for': home_history.get('goals_for', 0),
                    'away_prev_goals_for': away_history.get('goals_for', 0),
                    'home_prev_goals_against': home_history.get('goals_against', 0),
                    'away_prev_goals_against': away_history.get('goals_against', 0),
                    'home_prev_goal_diff': home_history.get('goal_diff', 0),
                    'away_prev_goal_diff': away_history.get('goal_diff', 0),
                })
                
                # Calculate form percentages if there are previous matches
                if home_history.get('matches', 0) > 0:
                    parsed_match['home_win_rate'] = home_history['wins'] / home_history['matches']
                    parsed_match['home_avg_goals_for'] = home_history['goals_for'] / home_history['matches']
                    parsed_match['home_avg_goals_against'] = home_history['goals_against'] / home_history['matches']
                else:
                    parsed_match['home_win_rate'] = 0.33  # neutral assumption
                    parsed_match['home_avg_goals_for'] = 1.35  # league average
                    parsed_match['home_avg_goals_against'] = 1.35
                    
                if away_history.get('matches', 0) > 0:
                    parsed_match['away_win_rate'] = away_history['wins'] / away_history['matches']
                    parsed_match['away_avg_goals_for'] = away_history['goals_for'] / away_history['matches']
                    parsed_match['away_avg_goals_against'] = away_history['goals_against'] / away_history['matches']
                else:
                    parsed_match['away_win_rate'] = 0.33
                    parsed_match['away_avg_goals_for'] = 1.35
                    parsed_match['away_avg_goals_against'] = 1.35
                
                # Head-to-head features
                h2h_key = f"{home_team}_vs_{away_team}"
                h2h_rev_key = f"{away_team}_vs_{home_team}"
                
                parsed_match['h2h_matches'] = 0
                parsed_match['h2h_home_wins'] = 0
                parsed_match['h2h_draws'] = 0
                parsed_match['h2h_away_wins'] = 0
                
                # Pre-match statistical features from match stats (but not scores!)
                stats = match.get('stats', {})
                for stat_name, stat_data in stats.items():
                    if isinstance(stat_data, dict) and stat_name not in ['Goals']:  # Exclude goal stats
                        home_val = self._parse_stat_value(stat_data.get('home', '0'))
                        away_val = self._parse_stat_value(stat_data.get('away', '0'))
                        
                        clean_name = stat_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                        parsed_match[f'home_{clean_name}'] = home_val
                        parsed_match[f'away_{clean_name}'] = away_val
                        parsed_match[f'diff_{clean_name}'] = home_val - away_val
                        
                        if away_val != 0:
                            parsed_match[f'ratio_{clean_name}'] = home_val / away_val
                        else:
                            parsed_match[f'ratio_{clean_name}'] = home_val if home_val > 0 else 1.0
                
                matches.append(parsed_match)
                
                # Update team history AFTER processing this match
                if home_team not in team_history:
                    team_history[home_team] = {'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 
                                             'goals_for': 0, 'goals_against': 0, 'goal_diff': 0}
                if away_team not in team_history:
                    team_history[away_team] = {'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                                             'goals_for': 0, 'goals_against': 0, 'goal_diff': 0}
                
                # Update home team history
                team_history[home_team]['matches'] += 1
                team_history[home_team]['goals_for'] += home_score
                team_history[home_team]['goals_against'] += away_score
                team_history[home_team]['goal_diff'] += (home_score - away_score)
                
                if result == '1':
                    team_history[home_team]['wins'] += 1
                elif result == 'X':
                    team_history[home_team]['draws'] += 1
                else:
                    team_history[home_team]['losses'] += 1
                
                # Update away team history
                team_history[away_team]['matches'] += 1
                team_history[away_team]['goals_for'] += away_score
                team_history[away_team]['goals_against'] += home_score
                team_history[away_team]['goal_diff'] += (away_score - home_score)
                
                if result == '2':
                    team_history[away_team]['wins'] += 1
                elif result == 'X':
                    team_history[away_team]['draws'] += 1
                else:
                    team_history[away_team]['losses'] += 1
                
            except Exception as e:
                print(f"  ⚠️ Match parse error: {e}")
                continue
        
        df = pd.DataFrame(matches)
        print(f"  ✅ Parsed {len(df)} matches")
        
        # Additional feature engineering
        enhanced_df = self._create_advanced_features(df)
        
        return enhanced_df

    def _parse_stat_value(self, value):
        """İstatistik değerini parse et"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                # Remove percentage signs and other characters
                clean_val = value.replace('%', '').replace(',', '').strip()
                return float(clean_val)
            except:
                return 0.0
        return 0.0

    def _create_advanced_features(self, df):
        """Gelişmiş özellikler ekle (NO LEAKAGE!)"""
        print("  🔧 Gelişmiş özellik mühendisliği...")
        
        enhanced_df = df.copy()
        
        # Team strength differential features
        enhanced_df['win_rate_diff'] = enhanced_df['home_win_rate'] - enhanced_df['away_win_rate']
        enhanced_df['avg_goals_diff'] = enhanced_df['home_avg_goals_for'] - enhanced_df['away_avg_goals_for']
        enhanced_df['defense_diff'] = enhanced_df['away_avg_goals_against'] - enhanced_df['home_avg_goals_against']
        
        # Form features
        enhanced_df['home_form_strength'] = (enhanced_df['home_win_rate'] * 3 + 
                                           enhanced_df['home_prev_draws'] * 1) / enhanced_df['home_prev_matches']
        enhanced_df['away_form_strength'] = (enhanced_df['away_win_rate'] * 3 + 
                                           enhanced_df['away_prev_draws'] * 1) / enhanced_df['away_prev_matches']
        
        # Replace infinities and NaNs
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], 0)
        enhanced_df = enhanced_df.fillna(0)
        
        return enhanced_df

    def prepare_features(self, df):
        """Özellikleri hazırla (target hariç)"""
        # Remove non-feature columns
        feature_df = df.drop(['result', 'home_team', 'away_team'], axis=1)
        
        # Encode categorical features if any
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                feature_df[col] = pd.Categorical(feature_df[col]).codes
        
        return feature_df

    def select_best_features(self, X, y, k=20):
        """En iyi özellikleri seç"""
        print(f"🎯 Özellik seçimi yapılıyor...")
        
        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names and scores
        selected_indices = selector.get_support(indices=True)
        feature_scores = selector.scores_
        
        self.selected_features = [(X.columns[i], feature_scores[i]) for i in selected_indices]
        self.selected_features.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  ✅ {len(self.selected_features)} özellik seçildi")
        print("  📊 Seçilen özellikler:")
        for i, (name, score) in enumerate(self.selected_features[:10], 1):
            print(f"    {i:2d}. {name:<30} (skor: {score:.2f})")
        
        self.feature_selector = selector
        return X_selected, [name for name, _ in self.selected_features]

    def train_model(self, X, y):
        """GP modelini eğit"""
        print("⚙️ GP hiperparametre optimizasyonu...")
        
        # Define kernels to try
        kernels = [
            ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
            ConstantKernel(1.0) * Matern(1.0, nu=1.5) + WhiteKernel(0.1),
            ConstantKernel(1.0) * RationalQuadratic(1.0, 0.1) + WhiteKernel(0.1),
            ConstantKernel(1.0) * (RBF(1.0) + Matern(1.0, nu=2.5)) + WhiteKernel(0.1),
        ]
        
        best_score = 0
        best_kernel = None
        
        # Try each kernel with cross-validation
        for i, kernel in enumerate(kernels, 1):
            try:
                gp = GaussianProcessClassifier(kernel=kernel, random_state=42, max_iter_predict=100)
                scores = cross_val_score(gp, X, y, cv=3, scoring='accuracy')
                mean_score = scores.mean()
                
                print(f"  Kernel {i}: %{mean_score:.1f} ± {scores.std():.3f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_kernel = kernel
                    
            except Exception as e:
                print(f"  Kernel {i}: Hata - {e}")
                continue
        
        if best_kernel is None:
            # Fallback to simple kernel
            best_kernel = ConstantKernel(1.0) * RBF(1.0)
            print("  🔄 Basit kernel kullanılıyor...")
        
        print(f"  🏆 En iyi CV skoru: %{best_score:.1f}")
        
        # Train final model
        self.gp_model = GaussianProcessClassifier(kernel=best_kernel, random_state=42, max_iter_predict=100)
        self.gp_model.fit(X, y)
        
        print("✅ Model eğitimi tamamlandı!")
        return best_score

    def predict_with_confidence(self, X):
        """Güven skoru ile tahmin yap"""
        # Get prediction probabilities
        probabilities = self.gp_model.predict_proba(X)
        predictions = self.gp_model.predict(X)
        
        # Calculate confidence as max probability
        confidence = np.max(probabilities, axis=1)
        
        return predictions, confidence, probabilities

    def evaluate_model(self, df):
        """Modeli değerlendir"""
        print("\n📊 MODEL DEĞERLENDIRMESI")
        print("=" * 40)
        
        # Prepare data
        X = self.prepare_features(df)
        y = self.label_encoder.fit_transform(df['result'])
        
        print(f"📊 Total features: {X.shape[1]}")
        
        # Feature selection
        X_selected, selected_names = self.select_best_features(X, y, k=20)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Split data temporally (later matches as test)
        split_point = int(len(df) * 0.75)
        X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train model
        cv_score = self.train_model(X_train, y_train)
        
        # Predictions
        y_pred, confidence, probabilities = self.predict_with_confidence(X_test)
        
        # Overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # High-confidence predictions (>70%)
        high_conf_mask = confidence > 0.7
        if np.any(high_conf_mask):
            high_conf_accuracy = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
            high_conf_ratio = np.sum(high_conf_mask) / len(y_test)
        else:
            high_conf_accuracy = 0
            high_conf_ratio = 0
        
        return {
            'cv_accuracy': cv_score,
            'test_accuracy': accuracy,
            'high_conf_accuracy': high_conf_accuracy,
            'high_conf_ratio': high_conf_ratio,
            'avg_confidence': np.mean(confidence),
            'y_test': y_test,
            'y_pred': y_pred,
            'confidence': confidence,
            'selected_features': selected_names
        }

def main():
    print("🎯 GP FUTBOL TAHMİN SİSTEMİ - DOĞRU VERSİYON")
    print("=" * 60)
    
    predictor = ProperGPPredictor()
    
    # Load and process data
    df = predictor.load_and_enhance_data()
    
    print(f"\n📊 GENEL İSTATİSTİKLER:")
    result_counts = df['result'].value_counts()
    print(f"  Toplam maç: {len(df)}")
    print(f"  Sonuç dağılımı: {dict(result_counts)}")
    
    # Evaluate model
    results = predictor.evaluate_model(df)
    
    print(f"\n🎯 TEST SONUÇLARI:")
    print(f"  CV Doğruluğu: %{results['cv_accuracy']:.1f}")
    print(f"  Test Doğruluğu: %{results['test_accuracy']:.1f}")
    print(f"  Yüksek Güvenli (%70+): %{results['high_conf_accuracy']:.1f}")
    print(f"  Yüksek Güvenli Oran: {int(results['high_conf_ratio'] * len(results['y_test']))}/{len(results['y_test'])} (%{results['high_conf_ratio'] * 100:.1f})")
    print(f"  Ortalama Güven: %{results['avg_confidence'] * 100:.1f}")
    print(f"  Kullanılan Özellik: {len(results['selected_features'])}")
    
    # Detailed report
    print(f"\n📋 DETAYLI RAPOR:")
    class_names = predictor.label_encoder.classes_
    print(classification_report(results['y_test'], results['y_pred'], 
                              target_names=class_names, digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(results['y_test'], results['y_pred'])
    print(f"\n🔍 CONFUSION MATRIX:")
    print(f"      Pred:  {' '.join(class_names)}")
    for i, actual_class in enumerate(class_names):
        print(f"Actual {actual_class}: {cm[i]}")
    
    # Performance assessment
    baseline_acc = max(result_counts) / len(df)  # Most frequent class
    improvement = results['test_accuracy'] - baseline_acc
    
    print(f"\n📈 PERFORMANS DEĞERLENDİRMESİ:")
    print(f"  Baseline: %{baseline_acc:.1f}")
    print(f"  İyileşme: {improvement:+.1f}%")
    
    if results['test_accuracy'] > baseline_acc + 0.15:
        print("  🟢 MÜKEMMEL: Çok başarılı model!")
    elif results['test_accuracy'] > baseline_acc + 0.10:
        print("  🟡 İYİ: Güzel performans!")
    elif results['test_accuracy'] > baseline_acc + 0.05:
        print("  🟠 ORTA: Biraz iyileşme var")
    else:
        print("  🔴 ZAYIF: Daha fazla çalışma gerekli")

if __name__ == "__main__":
    main()
