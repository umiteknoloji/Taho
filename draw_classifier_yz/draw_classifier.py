import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import sys
import os

# Core modülünü import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_normalizer import DataNormalizer

class DrawClassifier(BaseEstimator):
    """
    Evrensel beraberlik tahmini sınıfı - farklı lig formatlarını destekler.
    """
    
    def __init__(self, threshold=0.3, random_state=42):
        self.clf = HistGradientBoostingClassifier(random_state=random_state)
        self.threshold = threshold
        self.normalizer = DataNormalizer()
        self.feature_columns = None
        self.is_trained = False

    def prepare_features(self, df):
        """
        Beraberlik tahmini için özellik matrisi hazırla.
        
        Args:
            df: Normalize edilmiş DataFrame
            
        Returns:
            np.array: Özellik matrisi
        """
        features = []
        
        # Temel istatistikler
        base_features = [
            'possession_home', 'possession_away',
            'total_shots_home', 'total_shots_away',
            'shots_on_target_home', 'shots_on_target_away',
            'corners_home', 'corners_away',
            'fouls_home', 'fouls_away',
            'offsides_home', 'offsides_away'
        ]
        
        for feature in base_features:
            if feature in df.columns:
                features.append(feature)
        
        # Türetilmiş özellikler
        derived_features = {}
        
        # Possession farkı
        if 'possession_home' in df.columns and 'possession_away' in df.columns:
            derived_features['possession_diff'] = df['possession_home'] - df['possession_away']
        
        # Şut oranları
        if 'total_shots_home' in df.columns and 'shots_on_target_home' in df.columns:
            derived_features['shot_accuracy_home'] = np.where(
                df['total_shots_home'] > 0,
                df['shots_on_target_home'] / df['total_shots_home'],
                0
            )
        
        if 'total_shots_away' in df.columns and 'shots_on_target_away' in df.columns:
            derived_features['shot_accuracy_away'] = np.where(
                df['total_shots_away'] > 0,
                df['shots_on_target_away'] / df['total_shots_away'],
                0
            )
        
        # Şut farkı
        if 'total_shots_home' in df.columns and 'total_shots_away' in df.columns:
            derived_features['total_shots_diff'] = df['total_shots_home'] - df['total_shots_away']
        
        # Korner farkı
        if 'corners_home' in df.columns and 'corners_away' in df.columns:
            derived_features['corners_diff'] = df['corners_home'] - df['corners_away']
        
        # DataFrame'e türetilmiş özellikleri ekle
        for feature_name, feature_values in derived_features.items():
            df[feature_name] = feature_values
            features.append(feature_name)
        
        self.feature_columns = features
        return df[features].fillna(0).values

    def create_draw_labels(self, df):
        """
        Beraberlik etiketleri oluştur.
        
        Args:
            df: Normalize edilmiş DataFrame
            
        Returns:
            np.array: Beraberlik etiketleri (1=beraberlik, 0=sonuç)
        """
        return (df['home_score'] == df['away_score']).astype(int)

    def train_from_files(self, file_paths, test_size=0.2):
        """
        Dosyalardan modeli eğit.
        
        Args:
            file_paths: Veri dosyalarının yolları
            test_size: Test oranı
            
        Returns:
            dict: Eğitim sonuçları
        """
        # Verileri normalize et ve birleştir
        df = self.normalizer.merge_datasets(file_paths)
        
        if df.empty:
            raise ValueError("Veri yüklenemedi")
        
        print(f"Toplam {len(df)} maç verisi yüklendi")
        
        # Özellikleri hazırla
        X = self.prepare_features(df)
        y = self.create_draw_labels(df)
        
        print(f"Özellik boyutu: {X.shape}")
        print(f"Beraberlik oranı: {y.mean():.3f}")
        
        # Modeli eğit
        self.clf.fit(X, y)
        self.is_trained = True
        
        # Cross-validation skoru
        cv_scores = cross_val_score(self.clf, X, y, cv=5, scoring='roc_auc')
        
        # Tahminler
        y_pred_proba = self.clf.predict_proba(X)[:, 1]
        y_pred = y_pred_proba >= self.threshold
        
        results = {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'train_auc': roc_auc_score(y, y_pred_proba),
            'classification_report': classification_report(y, y_pred),
            'feature_importance': dict(zip(self.feature_columns, self.clf.feature_importances_)),
            'total_matches': len(df),
            'draw_rate': y.mean()
        }
        
        return results

    def train(self, X, y_draw):
        """Eski API uyumluluğu için."""
        self.clf.fit(X, y_draw)
        self.is_trained = True

    def predict_proba(self, X):
        """Beraberlik olasılığını tahmin et."""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        return self.clf.predict_proba(X)

    def predict(self, X):
        """Beraberlik tahmini yap."""
        p = self.predict_proba(X)[:,1]
        return p >= self.threshold
    
    def predict_from_stats(self, home_stats, away_stats):
        """
        İstatistiklerden beraberlik tahmini yap.
        
        Args:
            home_stats: Ev sahibi istatistikleri dict
            away_stats: Deplasman istatistikleri dict
            
        Returns:
            dict: Tahmin sonuçları
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        
        # Tek satırlık DataFrame oluştur
        data = {
            'home_score': 0, 'away_score': 0,  # Dummy values
            'possession_home': home_stats.get('possession', 0.5),
            'possession_away': away_stats.get('possession', 0.5),
            'total_shots_home': home_stats.get('total_shots', 0),
            'total_shots_away': away_stats.get('total_shots', 0),
            'shots_on_target_home': home_stats.get('shots_on_target', 0),
            'shots_on_target_away': away_stats.get('shots_on_target', 0),
            'corners_home': home_stats.get('corners', 0),
            'corners_away': away_stats.get('corners', 0),
            'fouls_home': home_stats.get('fouls', 0),
            'fouls_away': away_stats.get('fouls', 0),
            'offsides_home': home_stats.get('offsides', 0),
            'offsides_away': away_stats.get('offsides', 0)
        }
        
        df = pd.DataFrame([data])
        X = self.prepare_features(df)
        
        proba = self.predict_proba(X)[0, 1]
        prediction = proba >= self.threshold
        
        return {
            'draw_probability': proba,
            'is_draw_predicted': prediction,
            'confidence': 'High' if abs(proba - 0.5) > 0.2 else 'Medium' if abs(proba - 0.5) > 0.1 else 'Low'
        }
