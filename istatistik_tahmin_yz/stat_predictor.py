"""
istatistik_tahmin_yz modülü:
Tahmini maç istatistikleri üretir - Evrensel lig desteği ile.
"""

import pandas as pd
import numpy as np
import sys
import os
import random
from sklearn.model_selection import train_test_split

# Core modülünü import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_normalizer import DataNormalizer

class StatPredictorModel:
    """
    Evrensel istatistik tahmin modeli - farklı lig formatlarını destekler.
    """
    
    def __init__(self, random_seed=42):
        self.model = None
        self.stat_names = None
        self.normalizer = DataNormalizer()
        self.feature_columns = None
        self.target_columns = None
        self.random_seed = random_seed
        self.is_trained = False
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
    
    def prepare_features(self, df):
        """
        İstatistik tahmini için özellik matrisi hazırla.
        
        Args:
            df: Normalize edilmiş DataFrame
            
        Returns:
            np.array: Özellik matrisi
        """
        features = []
        
        # Temel özellikler (hedef değişkenler hariç)
        base_features = [
            'possession_home', 'possession_away',
            'home_score', 'away_score'  # Skor bilgisi önemli feature
        ]
        
        # Lig spesifik özellikler
        if 'passes_home' in df.columns:
            base_features.extend(['passes_home', 'passes_away'])
        if 'pass_accuracy_home' in df.columns:
            base_features.extend(['pass_accuracy_home', 'pass_accuracy_away'])
        
        for feature in base_features:
            if feature in df.columns:
                features.append(feature)
        
        # Türetilmiş özellikler
        derived_features = {}
        
        # Maç sonucu kategorileri
        if 'home_score' in df.columns and 'away_score' in df.columns:
            derived_features['goal_difference'] = df['home_score'] - df['away_score']
            derived_features['total_goals'] = df['home_score'] + df['away_score']
            derived_features['is_draw'] = (df['home_score'] == df['away_score']).astype(int)
            derived_features['is_home_win'] = (df['home_score'] > df['away_score']).astype(int)
        
        # Possession dominance
        if 'possession_home' in df.columns and 'possession_away' in df.columns:
            derived_features['possession_dominance'] = df['possession_home'] - df['possession_away']
        
        # DataFrame'e türetilmiş özellikleri ekle
        for feature_name, feature_values in derived_features.items():
            df[feature_name] = feature_values
            features.append(feature_name)
        
        self.feature_columns = features
        return df[features].fillna(0).values
    
    def prepare_targets(self, df):
        """
        Hedef istatistikleri hazırla.
        
        Args:
            df: Normalize edilmiş DataFrame
            
        Returns:
            np.array: Hedef matris
        """
        target_stats = [
            'total_shots_home', 'total_shots_away',
            'shots_on_target_home', 'shots_on_target_away',
            'corners_home', 'corners_away',
            'fouls_home', 'fouls_away',
            'offsides_home', 'offsides_away'
        ]
        
        # Mevcut sütunları filtrele
        available_targets = [col for col in target_stats if col in df.columns]
        self.target_columns = available_targets
        self.stat_names = available_targets
        
        return df[available_targets].fillna(0).values
    
    def train_from_files(self, file_paths, test_size=0.2, n_classes=10):
        """
        Dosyalardan modeli eğit. Tüm istatistikler için çoklu sınıflandırıcı kullanılır.
        Args:
            file_paths: Veri dosyalarının yolları
            test_size: Test oranı
            n_classes: Her istatistik için sınıf sayısı (örn. 0-9 arası değerler için 10)
        Returns:
            dict: Eğitim sonuçları
        """
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        df = self.normalizer.merge_datasets(file_paths)
        if df.empty:
            raise ValueError("Veri yüklenemedi")
        print(f"Toplam {len(df)} maç verisi yüklendi")
        X = self.prepare_features(df)
        y = self.prepare_targets(df)
        print(f"Özellik boyutu: {X.shape}")
        print(f"Hedef boyutu: {y.shape}")
        print(f"Tahmin edilecek istatistikler: {self.stat_names}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed
        )
        from sklearn.ensemble import RandomForestClassifier
        self.models = []
        stat_accuracies = {}
        for i, stat_name in enumerate(self.stat_names):
            y_train_class = np.clip(y_train[:, i], 0, n_classes-1).astype(int)
            y_test_class = np.clip(y_test[:, i], 0, n_classes-1).astype(int)
            clf = RandomForestClassifier(n_estimators=100, random_state=self.random_seed, n_jobs=1)
            clf.fit(X_train, y_train_class)
            self.models.append(clf)
            y_pred = clf.predict(X_test)
            acc = np.mean(y_pred == y_test_class)
            stat_accuracies[f'{stat_name}_accuracy'] = acc
        results = {
            'stat_accuracies': stat_accuracies,
            'total_matches': len(df),
            'predicted_stats': self.stat_names,
            'overall_accuracy': np.mean(list(stat_accuracies.values()))
        }
        self.is_trained = True
        return results

    def train(self, X, y, n_classes=10):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        try:
            self.stat_names = list(y.columns)
        except:
            self.stat_names = [f"stat_{i}" for i in range(np.array(y).shape[1])]
        from sklearn.ensemble import RandomForestClassifier
        self.models = []
        for i in range(np.array(y).shape[1]):
            y_class = np.clip(np.array(y)[:, i], 0, n_classes-1).astype(int)
            clf = RandomForestClassifier(n_estimators=100, random_state=self.random_seed, n_jobs=1)
            clf.fit(X, y_class)
            self.models.append(clf)
        self.is_trained = True

    def predict(self, X):
        if not hasattr(self, 'models') or not self.models:
            raise RuntimeError("Model henüz eğitilmedi")
        preds = []
        for clf in self.models:
            preds.append(clf.predict(X))
        return np.stack(preds, axis=1)
    
    def predict_from_match_context(self, home_team, away_team, expected_score_home, expected_score_away, possession_home=0.5):
        """
        Maç bağlamından istatistik tahmini yap.
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takımı
            expected_score_home: Beklenen ev sahibi golü
            expected_score_away: Beklenen deplasman golü
            possession_home: Ev sahibi ball possession
            
        Returns:
            dict: Tahmin edilen istatistikler
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        
        possession_away = 1.0 - possession_home
        
        # Tek satırlık DataFrame oluştur
        data = {
            'possession_home': possession_home,
            'possession_away': possession_away,
            'home_score': expected_score_home,
            'away_score': expected_score_away,
            'goal_difference': expected_score_home - expected_score_away,
            'total_goals': expected_score_home + expected_score_away,
            'is_draw': 1 if expected_score_home == expected_score_away else 0,
            'is_home_win': 1 if expected_score_home > expected_score_away else 0,
            'possession_dominance': possession_home - possession_away
        }
        
        df = pd.DataFrame([data])
        X = self.prepare_features(df)
        
        pred = self.model.predict(X).reshape(-1)
        pred = np.maximum(pred, 0)  # Negatif değerleri sıfıra çek
        
        result = {n: float(v) for n, v in zip(self.stat_names, pred)}
        
        # Ek analizler
        if 'total_shots_home' in result and 'total_shots_away' in result:
            result['shot_dominance'] = result['total_shots_home'] - result['total_shots_away']
        
        if 'corners_home' in result and 'corners_away' in result:
            result['corner_dominance'] = result['corners_home'] - result['corners_away']
        
        return result
