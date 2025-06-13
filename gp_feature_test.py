#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GP Feature Importance - Sadece hesaplama, görselleştirme yok
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

def test_gp_feature_importance():
    """GP feature importance testini çalıştır"""
    print("🔬 GP Feature Importance Testi")
    print("=" * 50)
    
    # Sample data
    X, y = make_classification(n_samples=200, n_features=6, n_classes=3, 
                              n_informative=4, n_redundant=2, random_state=42)
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train GP model
    gp = GaussianProcessClassifier(
        kernel=RBF(1.0) + WhiteKernel(1e-3),
        random_state=42
    )
    gp.fit(X_train_scaled, y_train)
    
    print(f"\n🎯 Model eğitimi tamamlandı")
    print(f"Test Accuracy: {gp.score(X_test_scaled, y_test):.3f}")
    
    # Feature importance via permutation
    print("\n📊 Permutation Importance hesaplanıyor...")
    try:
        perm_importance = permutation_importance(
            gp, X_test_scaled, y_test, 
            n_repeats=5, 
            random_state=42,
            scoring='accuracy'
        )
        
        print("✅ Feature importance hesaplandı")
        print("\n📈 Feature Importance Sonuçları:")
        
        # Sonuçları sırala
        sorted_indices = np.argsort(perm_importance.importances_mean)[::-1]
        
        for i, idx in enumerate(sorted_indices):
            feature_name = feature_names[idx]
            importance_mean = perm_importance.importances_mean[idx]
            importance_std = perm_importance.importances_std[idx]
            print(f"   {i+1}. {feature_name}: {importance_mean:.4f} ± {importance_std:.4f}")
        
    except Exception as e:
        print(f"❌ Feature importance hatası: {e}")
    
    # Uncertainty analysis
    print("\n🔍 Uncertainty Analizi:")
    try:
        y_pred_proba = gp.predict_proba(X_test_scaled)
        
        # Entropy hesapla
        entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-15), axis=1)
        confidence = np.max(y_pred_proba, axis=1)
        
        print(f"Ortalama Entropy: {np.mean(entropy):.3f}")
        print(f"Ortalama Confidence: {np.mean(confidence):.3f}")
        print(f"Min Confidence: {np.min(confidence):.3f}")
        print(f"Max Confidence: {np.max(confidence):.3f}")
        
        # En belirsiz tahminler
        most_uncertain_indices = np.argsort(entropy)[-3:]
        print(f"\n🤔 En Belirsiz 3 Tahmin:")
        for i, idx in enumerate(most_uncertain_indices):
            print(f"   {i+1}. Sample {idx}: Entropy={entropy[idx]:.3f}, Confidence={confidence[idx]:.3f}")
        
    except Exception as e:
        print(f"❌ Uncertainty analizi hatası: {e}")
    
    print("\n✅ Test tamamlandı!")

if __name__ == "__main__":
    test_gp_feature_importance()
