#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GP İçin Gelişmiş Özellikler
Mevcut GP'ye uncertainty quantification, advanced kernels ve hyperparameter optimization ekler
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, RationalQuadratic, 
    ExpSineSquared, DotProduct, ConstantKernel
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")

class EnhancedGPFeatures:
    """GP için gelişmiş özellikler"""
    
    @staticmethod
    def get_advanced_kernels():
        """Gelişmiş kernel kombinasyonları"""
        kernels = {}
        
        # 1. Temel kernels - Farklı length scale'ler
        kernels['rbf_short'] = RBF(length_scale=0.1)
        kernels['rbf_medium'] = RBF(length_scale=1.0)
        kernels['rbf_long'] = RBF(length_scale=10.0)
        
        # 2. Matern kernels - Farklı smoothness
        kernels['matern_12'] = Matern(nu=0.5, length_scale=1.0)
        kernels['matern_32'] = Matern(nu=1.5, length_scale=1.0) 
        kernels['matern_52'] = Matern(nu=2.5, length_scale=1.0)
        
        # 3. Rational Quadratic (infinite mixture of RBF)
        kernels['rational_quadratic'] = RationalQuadratic(length_scale=1.0, alpha=1.0)
        
        # 4. Dot Product (polynomial-like)
        kernels['dot_product'] = DotProduct(sigma_0=1.0)
        
        # 5. Periodic patterns (for seasonal data)
        kernels['periodic'] = ExpSineSquared(length_scale=1.0, periodicity=1.0)
        
        # 6. Composite kernels with constant scaling
        c = ConstantKernel(1.0)
        kernels['scaled_rbf'] = c * RBF(1.0)
        kernels['scaled_matern'] = c * Matern(nu=2.5, length_scale=1.0)
        
        # 7. Additive combinations
        kernels['rbf_plus_white'] = RBF(1.0) + WhiteKernel(1e-3)
        kernels['matern_plus_white'] = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(1e-3)
        kernels['rbf_plus_periodic'] = RBF(1.0) + ExpSineSquared(1.0, 1.0)
        
        # 8. Multiplicative combinations
        kernels['rbf_times_periodic'] = RBF(1.0) * ExpSineSquared(1.0, 1.0)
        kernels['matern_times_rational'] = Matern(nu=1.5, length_scale=1.0) * RationalQuadratic(1.0, 1.0)
        
        # 9. Complex combinations
        kernels['complex_1'] = (ConstantKernel(1.0) * RBF(1.0) + 
                               WhiteKernel(1e-3) + 
                               Matern(nu=1.5, length_scale=1.0))
        
        kernels['complex_2'] = (RBF(1.0) * ExpSineSquared(1.0, 1.0) + 
                               RationalQuadratic(1.0, 1.0) + 
                               WhiteKernel(1e-3))
        
        # 10. Football-specific optimized
        kernels['football_optimized'] = (ConstantKernel(1.0) * 
                                        Matern(nu=2.5, length_scale=1.0) + 
                                        RBF(length_scale=0.5) + 
                                        WhiteKernel(1e-2))
        
        return kernels
    
    @staticmethod
    def get_hyperparameter_grid():
        """Hyperparameter optimization için grid"""
        return {
            'kernel': [
                RBF(length_scale=ls) for ls in [0.1, 0.5, 1.0, 2.0, 5.0]
            ] + [
                Matern(nu=nu, length_scale=ls) 
                for nu in [0.5, 1.5, 2.5] 
                for ls in [0.1, 1.0, 5.0]
            ] + [
                RBF(ls) + WhiteKernel(noise) 
                for ls in [0.5, 1.0, 2.0] 
                for noise in [1e-3, 1e-2, 1e-1]
            ],
            'n_restarts_optimizer': [1, 3, 5],
            'max_iter_predict': [50, 100, 200]
        }
    
    @staticmethod
    def compute_uncertainty_metrics(gp_model, X_test, y_test):
        """Uncertainty quantification metrikleri"""
        if not hasattr(gp_model, 'predict_proba'):
            return {}
        
        try:
            # Tahmin olasılıkları
            y_pred_proba = gp_model.predict_proba(X_test)
            
            # Entropy (uncertainty measure)
            entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-15), axis=1)
            
            # Maximum probability (confidence)
            max_prob = np.max(y_pred_proba, axis=1)
            
            # Predictive variance (if available)
            predictive_var = np.var(y_pred_proba, axis=1)
            
            # Log loss (calibration measure)
            log_loss_val = log_loss(y_test, y_pred_proba)
            
            return {
                'mean_entropy': np.mean(entropy),
                'std_entropy': np.std(entropy),
                'mean_confidence': np.mean(max_prob),
                'std_confidence': np.std(max_prob),
                'mean_predictive_var': np.mean(predictive_var),
                'log_loss': log_loss_val,
                'uncertainty_scores': entropy,
                'confidence_scores': max_prob
            }
        except Exception as e:
            print(f"⚠️ Uncertainty hesaplama hatası: {e}")
            return {}
    
    @staticmethod
    def optimize_hyperparameters(X_train, y_train, search_type='grid', cv=3):
        """Hyperparameter optimization"""
        base_gp = GaussianProcessClassifier(random_state=42)
        param_grid = EnhancedGPFeatures.get_hyperparameter_grid()
        
        if search_type == 'grid':
            # Grid search (kapsamlı ama yavaş)
            search = GridSearchCV(
                base_gp, 
                param_grid, 
                cv=cv, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        else:
            # Random search (hızlı)
            search = RandomizedSearchCV(
                base_gp,
                param_grid,
                n_iter=20,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        
        search.fit(X_train, y_train)
        
        return {
            'best_model': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
    
    @staticmethod
    def active_learning_selection(gp_model, X_unlabeled, n_samples=10):
        """Active learning için en belirsiz örnekleri seç"""
        try:
            # Tahmin olasılıkları
            y_pred_proba = gp_model.predict_proba(X_unlabeled)
            
            # Entropy ile uncertainty hesapla
            entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-15), axis=1)
            
            # En yüksek entropy'ye sahip örnekleri seç
            uncertain_indices = np.argsort(entropy)[-n_samples:]
            
            return {
                'selected_indices': uncertain_indices,
                'uncertainty_scores': entropy[uncertain_indices],
                'samples': X_unlabeled[uncertain_indices]
            }
        except Exception as e:
            print(f"⚠️ Active learning hatası: {e}")
            return {}
    
    @staticmethod
    def compute_feature_importance(gp_model, X_test, feature_names):
        """Feature importance via permutation"""
        if not hasattr(gp_model, 'predict'):
            return {}
        
        try:
            # Baseline accuracy
            baseline_acc = gp_model.score(X_test, gp_model.predict(X_test))
            
            importance_scores = {}
            
            for i, feature_name in enumerate(feature_names):
                # Feature'ı karıştır
                X_permuted = X_test.copy()
                np.random.shuffle(X_permuted[:, i])
                
                # Yeni accuracy
                permuted_acc = gp_model.score(X_permuted, gp_model.predict(X_permuted))
                
                # Importance = accuracy düşüşü
                importance_scores[feature_name] = baseline_acc - permuted_acc
            
            return importance_scores
        except Exception as e:
            print(f"⚠️ Feature importance hatası: {e}")
            return {}
    
    @staticmethod
    def calibration_analysis(gp_model, X_test, y_test):
        """Model calibration analizi"""
        try:
            y_pred_proba = gp_model.predict_proba(X_test)
            y_pred = gp_model.predict(X_test)
            
            # Confidence bins
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            confidences = np.max(y_pred_proba, axis=1)
            accuracies = (y_pred == y_test).astype(float)
            
            calibration_data = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = accuracies[in_bin].mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    
                    calibration_data.append({
                        'bin_lower': bin_lower,
                        'bin_upper': bin_upper,
                        'accuracy': accuracy_in_bin,
                        'confidence': avg_confidence_in_bin,
                        'proportion': prop_in_bin
                    })
            
            # Expected Calibration Error (ECE)
            ece = sum([
                abs(data['accuracy'] - data['confidence']) * data['proportion']
                for data in calibration_data
            ])
            
            return {
                'calibration_data': calibration_data,
                'expected_calibration_error': ece,
                'reliability_diagram': calibration_data
            }
        except Exception as e:
            print(f"⚠️ Calibration analizi hatası: {e}")
            return {}

# Test fonksiyonu
def test_enhanced_features():
    """Gelişmiş özellikleri test et"""
    print("🧪 GP Gelişmiş Özellik Testleri")
    print("=" * 50)
    
    # Sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3, 
                              n_informative=8, n_redundant=2, random_state=42)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # 1. Advanced kernels test
    print("\n🔬 Gelişmiş Kernels Test:")
    kernels = EnhancedGPFeatures.get_advanced_kernels()
    print(f"Toplam kernel sayısı: {len(kernels)}")
    
    # 2. Quick model test
    gp = GaussianProcessClassifier(
        kernel=kernels['football_optimized'], 
        random_state=42
    )
    gp.fit(X_train, y_train)
    
    # 3. Uncertainty metrics
    print("\n📊 Uncertainty Metrics:")
    uncertainty = EnhancedGPFeatures.compute_uncertainty_metrics(gp, X_test, y_test)
    if uncertainty:
        print(f"Ortalama Entropy: {uncertainty['mean_entropy']:.3f}")
        print(f"Ortalama Confidence: {uncertainty['mean_confidence']:.3f}")
        print(f"Log Loss: {uncertainty['log_loss']:.3f}")
    
    # 4. Calibration analysis
    print("\n🎯 Calibration Analysis:")
    calibration = EnhancedGPFeatures.calibration_analysis(gp, X_test, y_test)
    if calibration:
        print(f"Expected Calibration Error: {calibration['expected_calibration_error']:.3f}")
    
    print("\n✅ Test tamamlandı!")

if __name__ == "__main__":
    test_enhanced_features()
