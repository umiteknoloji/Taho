#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GP için Akıllı Özellik Seçici
Bu modül GP'nin performansını optimize etmek için
en iyi özellikleri seçer ve özellik önemlerini analiz eder.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

class GPFeatureSelector:
    def __init__(self):
        self.selected_features = []
        self.feature_scores = {}
        self.scaler = StandardScaler()
        
    def analyze_feature_importance_for_gp(self, X, y, feature_names):
        """GP için özellik önemlerini analiz et"""
        print("🔍 GP için özellik önem analizi...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Univariate feature selection
        print("  📊 Univariate analiz...")
        f_scores = self._univariate_analysis(X_scaled, y)
        
        # 2. Mutual information
        print("  🔗 Mutual information analizi...")
        mi_scores = self._mutual_information_analysis(X_scaled, y)
        
        # 3. GP-specific correlation analysis
        print("  🎯 GP-özel korelasyon analizi...")
        gp_scores = self._gp_specific_analysis(X_scaled, y)
        
        # 4. Recursive feature elimination with RF (proxy for GP)
        print("  🔄 Recursive feature elimination...")
        rfe_scores = self._recursive_elimination(X_scaled, y)
        
        # Combine scores
        combined_scores = self._combine_feature_scores(
            f_scores, mi_scores, gp_scores, rfe_scores, feature_names
        )
        
        return combined_scores
    
    def _univariate_analysis(self, X, y):
        """Univariate feature selection"""
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        return selector.scores_
    
    def _mutual_information_analysis(self, X, y):
        """Mutual information analysis"""
        mi_scores = mutual_info_classif(X, y, random_state=42)
        return mi_scores
    
    def _gp_specific_analysis(self, X, y):
        """GP-özel özellik analizi"""
        gp_scores = []
        
        # Basit GP ile her özelliğin bireysel performansını test et
        base_gp = GaussianProcessClassifier(
            kernel=ConstantKernel(1.0) * RBF(1.0),
            random_state=42,
            n_restarts_optimizer=1
        )
        
        for i in range(X.shape[1]):
            try:
                X_single = X[:, i].reshape(-1, 1)
                cv_scores = cross_val_score(base_gp, X_single, y, cv=3, scoring='accuracy')
                gp_scores.append(cv_scores.mean())
            except:
                gp_scores.append(0)
        
        return np.array(gp_scores)
    
    def _recursive_elimination(self, X, y):
        """Recursive feature elimination"""
        # RF as proxy for feature importance (faster than GP)
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        return rf.feature_importances_
    
    def _combine_feature_scores(self, f_scores, mi_scores, gp_scores, rfe_scores, feature_names):
        """Özellik skorlarını birleştir"""
        # Normalize scores
        f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
        mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
        gp_scores_norm = (gp_scores - gp_scores.min()) / (gp_scores.max() - gp_scores.min() + 1e-8)
        rfe_scores_norm = (rfe_scores - rfe_scores.min()) / (rfe_scores.max() - rfe_scores.min() + 1e-8)
        
        # Weighted combination (GP scores get higher weight)
        combined = (
            0.2 * f_scores_norm +
            0.3 * mi_scores_norm +
            0.4 * gp_scores_norm +  # GP'ye daha fazla ağırlık
            0.1 * rfe_scores_norm
        )
        
        # Create feature score dictionary
        feature_scores = {}
        for i, name in enumerate(feature_names):
            feature_scores[name] = {
                'combined_score': combined[i],
                'f_score': f_scores_norm[i],
                'mi_score': mi_scores_norm[i],
                'gp_score': gp_scores_norm[i],
                'rfe_score': rfe_scores_norm[i]
            }
        
        return feature_scores
    
    def select_optimal_features(self, X, y, feature_names, target_features=None):
        """Optimal özellik seçimi"""
        print("🎯 Optimal özellik seçimi...")
        
        # Feature importance analysis
        feature_scores = self.analyze_feature_importance_for_gp(X, y, feature_names)
        
        # Sort features by combined score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
        if target_features is None:
            # Progressive feature selection
            best_features, best_score = self._progressive_feature_selection(X, y, sorted_features)
        else:
            # Select top N features
            best_features = [name for name, _ in sorted_features[:target_features]]
            best_score = self._evaluate_feature_set(X, y, feature_names, best_features)
        
        self.selected_features = best_features
        self.feature_scores = feature_scores
        
        return best_features, best_score
    
    def _progressive_feature_selection(self, X, y, sorted_features, max_features=25):
        """Progresif özellik seçimi"""
        print("  🔄 Progresif seçim yapılıyor...")
        
        best_score = 0
        best_features = []
        current_features = []
        
        # Her seferinde bir özellik ekleyerek test et
        for i, (feature_name, scores) in enumerate(sorted_features[:max_features]):
            current_features.append(feature_name)
            
            # Bu özellik setiyle performans değerlendir
            score = self._evaluate_feature_set(X, y, [f[0] for f in sorted_features], current_features)
            
            print(f"    {len(current_features):2d} özellik: %{score*100:.1f}")
            
            if score > best_score:
                best_score = score
                best_features = current_features.copy()
            elif score < best_score - 0.02:  # Performance düşerse dur
                print(f"    ⚠️ Performans düştü, durduruluyor...")
                break
        
        print(f"  ✅ En iyi: {len(best_features)} özellik (%{best_score*100:.1f})")
        return best_features, best_score
    
    def _evaluate_feature_set(self, X, y, all_feature_names, selected_feature_names):
        """Özellik setini değerlendir"""
        # Feature indices
        feature_indices = [all_feature_names.index(name) for name in selected_feature_names if name in all_feature_names]
        
        if not feature_indices:
            return 0
        
        X_selected = X[:, feature_indices]
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # GP with optimized kernel
        gp = GaussianProcessClassifier(
            kernel=ConstantKernel(1.0) * RBF(1.0),
            random_state=42,
            n_restarts_optimizer=2
        )
        
        try:
            cv_scores = cross_val_score(gp, X_scaled, y, cv=3, scoring='accuracy')
            return cv_scores.mean()
        except:
            return 0
    
    def get_feature_report(self):
        """Özellik raporu oluştur"""
        if not self.feature_scores:
            return "Henüz özellik analizi yapılmadı."
        
        report = "\n🏆 ÖZELLİK SEÇIM RAPORU\n"
        report += "=" * 50 + "\n\n"
        
        # Top features
        top_features = sorted(self.feature_scores.items(), 
                            key=lambda x: x[1]['combined_score'], reverse=True)[:15]
        
        report += "📊 EN İYİ 15 ÖZELLİK:\n"
        report += "-" * 30 + "\n"
        for i, (name, scores) in enumerate(top_features, 1):
            report += f"{i:2d}. {name:<25} Skor: {scores['combined_score']:.3f}\n"
            report += f"    GP: {scores['gp_score']:.3f} | MI: {scores['mi_score']:.3f} | F: {scores['f_score']:.3f}\n\n"
        
        # Selected features
        if self.selected_features:
            report += f"✅ SEÇİLEN ÖZELLİKLER ({len(self.selected_features)}):\n"
            report += "-" * 30 + "\n"
            for i, feature in enumerate(self.selected_features, 1):
                score = self.feature_scores[feature]['combined_score']
                report += f"{i:2d}. {feature} (Skor: {score:.3f})\n"
        
        return report
    
    def save_selected_features(self, filename='selected_features.json'):
        """Seçilen özellikleri kaydet"""
        if self.selected_features:
            feature_data = {
                'selected_features': self.selected_features,
                'feature_scores': self.feature_scores,
                'selection_method': 'GP-optimized progressive selection'
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(feature_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Özellikler kaydedildi: {filename}")
            return True
        return False

def main():
    """Test the GP feature selector"""
    selector = GPFeatureSelector()
    
    try:
        # Test data (random for demonstration)
        np.random.seed(42)
        n_samples = 1000
        n_features = 30
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice(['1', 'X', '2'], n_samples)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        print(f"📊 Test verisi: {n_samples} örnek, {n_features} özellik")
        
        # Feature selection
        best_features, best_score = selector.select_optimal_features(
            X, y, feature_names, target_features=15
        )
        
        print(f"\n🏆 En iyi özellik seti:")
        print(f"  Özellik sayısı: {len(best_features)}")
        print(f"  Cross-validation skoru: %{best_score*100:.1f}")
        
        # Feature report
        report = selector.get_feature_report()
        print(report)
        
        # Save results
        selector.save_selected_features()
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
