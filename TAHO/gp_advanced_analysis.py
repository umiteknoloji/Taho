#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GP İçin Feature Importance ve Visualization Eklentileri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

class GPFeatureImportance:
    """GP için feature importance hesaplama"""
    
    @staticmethod
    def compute_permutation_importance(gp_model, X_test, y_test, feature_names, n_repeats=10):
        """Permutation importance hesapla"""
        try:
            print("🔍 Feature importance hesaplanıyor...")
            
            # Permutation importance
            perm_importance = permutation_importance(
                gp_model, X_test, y_test, 
                n_repeats=n_repeats, 
                random_state=42,
                scoring='accuracy'
            )
            
            # Sonuçları organize et
            importance_dict = {}
            for i, feature_name in enumerate(feature_names):
                importance_dict[feature_name] = {
                    'mean': perm_importance.importances_mean[i],
                    'std': perm_importance.importances_std[i],
                    'rank': np.argsort(perm_importance.importances_mean)[::-1].tolist().index(i) + 1
                }
            
            # En önemli feature'ları sırala
            sorted_features = sorted(importance_dict.items(), 
                                   key=lambda x: x[1]['mean'], reverse=True)
            
            print("✅ Feature importance hesaplandı")
            print("\n📊 En Önemli 5 Feature:")
            for i, (feature, stats) in enumerate(sorted_features[:5]):
                print(f"   {i+1}. {feature}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            return importance_dict, sorted_features
            
        except Exception as e:
            print(f"❌ Feature importance hatası: {e}")
            return {}, []
    
    @staticmethod
    def plot_feature_importance(importance_dict, save_path=None):
        """Feature importance grafiği çiz"""
        try:
            # Veriyi hazırla
            features = list(importance_dict.keys())
            means = [importance_dict[f]['mean'] for f in features]
            stds = [importance_dict[f]['std'] for f in features]
            
            # Sırala
            sorted_indices = np.argsort(means)[::-1]
            features_sorted = [features[i] for i in sorted_indices]
            means_sorted = [means[i] for i in sorted_indices]
            stds_sorted = [stds[i] for i in sorted_indices]
            
            # Plot
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(features_sorted)), means_sorted, 
                    xerr=stds_sorted, capsize=5, alpha=0.7)
            plt.yticks(range(len(features_sorted)), features_sorted)
            plt.xlabel('Permutation Importance')
            plt.title('GP Feature Importance (Permutation)')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Grafik kaydedildi: {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            print(f"❌ Grafik çizim hatası: {e}")
            return None

class GPVisualization:
    """GP modeli için görselleştirme araçları"""
    
    @staticmethod
    def plot_calibration_curve(calibration_data, save_path=None):
        """Calibration curve çiz"""
        try:
            if not calibration_data or 'calibration_data' not in calibration_data:
                print("⚠️ Calibration verisi bulunamadı")
                return None
            
            cal_data = calibration_data['calibration_data']
            
            # Veriyi hazırla
            confidences = [d['confidence'] for d in cal_data]
            accuracies = [d['accuracy'] for d in cal_data]
            
            # Plot
            plt.figure(figsize=(10, 8))
            
            # Perfect calibration line
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
            
            # Actual calibration
            plt.plot(confidences, accuracies, 'o-', linewidth=2, 
                    markersize=8, label='GP Model', color='red')
            
            # Fill area
            plt.fill_between(confidences, accuracies, alpha=0.3, color='red')
            
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Calibration Plot (ECE: {calibration_data["expected_calibration_error"]:.3f})')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Calibration grafiği kaydedildi: {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            print(f"❌ Calibration plot hatası: {e}")
            return None
    
    @staticmethod
    def plot_uncertainty_distribution(uncertainty_metrics, save_path=None):
        """Uncertainty dağılımını çiz"""
        try:
            if not uncertainty_metrics or 'uncertainty_scores' not in uncertainty_metrics:
                print("⚠️ Uncertainty verisi bulunamadı")
                return None
            
            entropy_scores = uncertainty_metrics['uncertainty_scores']
            confidence_scores = uncertainty_metrics['confidence_scores']
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Entropy distribution
            ax1.hist(entropy_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(np.mean(entropy_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(entropy_scores):.3f}')
            ax1.set_xlabel('Entropy (Uncertainty)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Uncertainty Distribution')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Confidence distribution
            ax2.hist(confidence_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(np.mean(confidence_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(confidence_scores):.3f}')
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Confidence Distribution')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Uncertainty grafiği kaydedildi: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Uncertainty plot hatası: {e}")
            return None
    
    @staticmethod
    def plot_kernel_comparison(kernel_performance, save_path=None):
        """Kernel performanslarını karşılaştır"""
        try:
            if not kernel_performance:
                print("⚠️ Kernel performance verisi bulunamadı")
                return None
            
            # Veriyi hazırla
            kernels = list(kernel_performance.keys())
            accuracies = [kernel_performance[k]['accuracy'] for k in kernels]
            cv_means = [kernel_performance[k]['cv_mean'] for k in kernels]
            cv_stds = [kernel_performance[k]['cv_std'] for k in kernels]
            training_times = [kernel_performance[k]['training_time'] for k in kernels]
            
            # Plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Accuracy comparison
            ax1.bar(range(len(kernels)), accuracies, alpha=0.7, color='blue')
            ax1.set_xticks(range(len(kernels)))
            ax1.set_xticklabels(kernels, rotation=45, ha='right')
            ax1.set_ylabel('Test Accuracy')
            ax1.set_title('Kernel Accuracy Comparison')
            ax1.grid(axis='y', alpha=0.3)
            
            # CV scores with error bars
            ax2.errorbar(range(len(kernels)), cv_means, yerr=cv_stds, 
                        fmt='o-', capsize=5, linewidth=2)
            ax2.set_xticks(range(len(kernels)))
            ax2.set_xticklabels(kernels, rotation=45, ha='right')
            ax2.set_ylabel('CV Score')
            ax2.set_title('Cross-Validation Scores')
            ax2.grid(alpha=0.3)
            
            # Training time
            ax3.bar(range(len(kernels)), training_times, alpha=0.7, color='red')
            ax3.set_xticks(range(len(kernels)))
            ax3.set_xticklabels(kernels, rotation=45, ha='right')
            ax3.set_ylabel('Training Time (s)')
            ax3.set_title('Training Time Comparison')
            ax3.grid(axis='y', alpha=0.3)
            
            # Accuracy vs Time scatter
            ax4.scatter(training_times, accuracies, s=100, alpha=0.7)
            for i, kernel in enumerate(kernels):
                ax4.annotate(kernel, (training_times[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax4.set_xlabel('Training Time (s)')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Accuracy vs Training Time')
            ax4.grid(alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Kernel comparison grafiği kaydedildi: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Kernel comparison plot hatası: {e}")
            return None

def test_advanced_analysis():
    """Gelişmiş analiz özelliklerini test et"""
    print("🔬 GP Gelişmiş Analiz Testleri")
    print("=" * 50)
    
    # Sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X, y = make_classification(n_samples=300, n_features=8, n_classes=3, 
                              n_informative=6, n_redundant=2, random_state=42)
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train GP model
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    gp = GaussianProcessClassifier(
        kernel=RBF(1.0) + WhiteKernel(1e-3),
        random_state=42
    )
    gp.fit(X_train_scaled, y_train)
    
    print("\n🎯 Model eğitimi tamamlandı")
    print(f"Test Accuracy: {gp.score(X_test_scaled, y_test):.3f}")
    
    # Feature importance
    print("\n📊 Feature Importance Analizi:")
    importance_dict, sorted_features = GPFeatureImportance.compute_permutation_importance(
        gp, X_test_scaled, y_test, feature_names, n_repeats=5
    )
    
    # Simulated uncertainty metrics for testing
    y_pred_proba = gp.predict_proba(X_test_scaled)
    entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-15), axis=1)
    confidence = np.max(y_pred_proba, axis=1)
    
    uncertainty_metrics = {
        'uncertainty_scores': entropy,
        'confidence_scores': confidence,
        'mean_entropy': np.mean(entropy),
        'mean_confidence': np.mean(confidence)
    }
    
    print(f"\n🔍 Uncertainty Analizi:")
    print(f"Ortalama Entropy: {uncertainty_metrics['mean_entropy']:.3f}")
    print(f"Ortalama Confidence: {uncertainty_metrics['mean_confidence']:.3f}")
    
    print("\n✅ Gelişmiş analiz testleri tamamlandı!")
    return {
        'model': gp,
        'importance': importance_dict,
        'uncertainty': uncertainty_metrics
    }

if __name__ == "__main__":
    test_advanced_analysis()
