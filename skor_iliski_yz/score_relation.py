"""
skor_iliski_yz modülü:
Maç istatistikleri ile skor ilişkisini öğrenen model - Evrensel lig desteği ile.
Örnek kullanım:
    model = ScoreRelationModel()
    model.train_from_files(['data/ALM_stat.json', 'data/BEL_stat.json'])
    print(model.get_feature_importance())
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
import os

# Core modülünü import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_normalizer import DataNormalizer

class ScoreRelationModel:
    """
    Evrensel skor-istatistik ilişki modeli - farklı lig formatlarını destekler.
    """
    
    def __init__(self, target="goal_diff", model_type="xgboost", random_state=42):
        self.target = target
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.normalizer = DataNormalizer()
        self.random_state = random_state
        self.is_trained = False
        
        # Hedef değişken seçenekleri
        self.target_options = {
            'goal_diff': 'Gol farkı (ev sahibi - deplasman)',
            'total_goals': 'Toplam gol sayısı',
            'home_goals': 'Ev sahibi gol sayısı',
            'away_goals': 'Deplasman gol sayısı',
            'home_win': 'Ev sahibi galibiyeti (0/1)',
            'draw': 'Beraberlik (0/1)'
        }
    
    def prepare_features(self, df):
        """
        Skor ilişkisi analizi için özellik matrisi hazırla.
        
        Args:
            df: Normalize edilmiş DataFrame
            
        Returns:
            np.array: Özellik matrisi
        """
        features = []
        
        # İstatistik özellikleri
        stat_features = [
            'possession_home', 'possession_away',
            'total_shots_home', 'total_shots_away',
            'shots_on_target_home', 'shots_on_target_away',
            'corners_home', 'corners_away',
            'fouls_home', 'fouls_away',
            'offsides_home', 'offsides_away'
        ]
        
        for feature in stat_features:
            if feature in df.columns:
                features.append(feature)
        
        # Türetilmiş özellikler
        derived_features = {}
        
        # Dominance features
        if 'possession_home' in df.columns and 'possession_away' in df.columns:
            derived_features['possession_dominance'] = df['possession_home'] - df['possession_away']
        
        if 'total_shots_home' in df.columns and 'total_shots_away' in df.columns:
            derived_features['shot_dominance'] = df['total_shots_home'] - df['total_shots_away']
            derived_features['shot_ratio'] = np.where(
                df['total_shots_away'] > 0,
                df['total_shots_home'] / df['total_shots_away'],
                1.0
            )
        
        if 'corners_home' in df.columns and 'corners_away' in df.columns:
            derived_features['corner_dominance'] = df['corners_home'] - df['corners_away']
        
        # Efficiency features
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
        
        # Attack intensity
        if 'total_shots_home' in df.columns and 'corners_home' in df.columns:
            derived_features['attack_intensity_home'] = df['total_shots_home'] + df['corners_home'] * 0.3
        
        if 'total_shots_away' in df.columns and 'corners_away' in df.columns:
            derived_features['attack_intensity_away'] = df['total_shots_away'] + df['corners_away'] * 0.3
        
        # Lig spesifik özellikler
        if 'passes_home' in df.columns and 'passes_away' in df.columns:
            derived_features['pass_dominance'] = df['passes_home'] - df['passes_away']
        
        if 'pass_accuracy_home' in df.columns and 'pass_accuracy_away' in df.columns:
            derived_features['pass_accuracy_diff'] = df['pass_accuracy_home'] - df['pass_accuracy_away']
        
        # DataFrame'e türetilmiş özellikleri ekle
        for feature_name, feature_values in derived_features.items():
            df[feature_name] = feature_values
            features.append(feature_name)
        
        self.feature_names = features
        return df[features].fillna(0).values
    
    def prepare_target(self, df):
        """
        Hedef değişkeni hazırla.
        
        Args:
            df: Normalize edilmiş DataFrame
            
        Returns:
            np.array: Hedef değişken
        """
        if self.target == 'goal_diff':
            return (df['home_score'] - df['away_score']).values
        elif self.target == 'total_goals':
            return (df['home_score'] + df['away_score']).values
        elif self.target == 'home_goals':
            return df['home_score'].values
        elif self.target == 'away_goals':
            return df['away_score'].values
        elif self.target == 'home_win':
            return (df['home_score'] > df['away_score']).astype(int).values
        elif self.target == 'draw':
            return (df['home_score'] == df['away_score']).astype(int).values
        else:
            raise ValueError(f"Geçersiz hedef: {self.target}")
    
    def train_from_files(self, file_paths, test_size=0.2):
        """
        Dosyalardan modeli eğit. Sadece sınıflandırıcı kullanılır.
        
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
        print(f"Hedef değişken: {self.target_options.get(self.target, self.target)}")
        
        # Özellikleri ve hedefi hazırla
        X = self.prepare_features(df)
        y = self.prepare_target(df)
        
        print(f"Özellik boyutu: {X.shape}")
        print(f"Hedef istatistikleri: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Model seçimi ve eğitimi - Sadece sınıflandırıcılar
        if self.target in ['home_win', 'draw']:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.ensemble import GradientBoostingClassifier
            
            if self.model_type == "xgboost":
                from xgboost import XGBClassifier
                self.model = XGBClassifier(objective="binary:logistic", random_state=self.random_state)
            elif self.model_type == "lightgbm":
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier(random_state=self.random_state, verbose=-1)
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            # Gol sayısı tahmini için sınıflandırma (0-6 gol)
            y_train_class = np.clip(y_train, 0, 6).astype(int)
            y_test_class = np.clip(y_test, 0, 6).astype(int)
            
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.ensemble import GradientBoostingClassifier
            
            if self.model_type == "xgboost":
                from xgboost import XGBClassifier
                self.model = XGBClassifier(objective="multi:softmax", num_class=7, random_state=self.random_state)
            elif self.model_type == "lightgbm":
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier(random_state=self.random_state, verbose=-1)
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            
            y_train, y_test = y_train_class, y_test_class
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Test tahminleri ve metrik hesaplamaları
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': acc,
            'confusion_matrix': cm.tolist(),
            'total_matches': len(df),
            'target': self.target
        }
        
        return results
    
    def train(self, X, y):
        """Eski API uyumluluğu için."""
        try:
            self.feature_names = list(X.columns)
        except:
            self.feature_names = [f"feat_{i}" for i in range(np.array(X).shape[1])]
        
        X_arr = np.array(X)
        y_arr = np.array(y)
        
        # Hedef türüne göre model seçimi
        if self.target in ['home_win', 'draw']:
            # İkili sınıflandırma
            from sklearn.ensemble import RandomForestClassifier
            
            if self.model_type == "xgboost":
                from xgboost import XGBClassifier
                self.model = XGBClassifier(objective="binary:logistic", random_state=self.random_state)
            elif self.model_type == "lightgbm":
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier(random_state=self.random_state, verbose=-1)
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            # Gol sayısı tahmini için sınıflandırma (0-6 gol)
            y_arr = np.clip(y_arr, 0, 6).astype(int)
            
            from sklearn.ensemble import RandomForestClassifier
            
            if self.model_type == "xgboost":
                from xgboost import XGBClassifier
                self.model = XGBClassifier(objective="multi:softmax", num_class=7, random_state=self.random_state)
            elif self.model_type == "lightgbm":
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier(random_state=self.random_state, verbose=-1)
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        self.model.fit(X_arr, y_arr)
        self.is_trained = True
    
    def get_feature_importance(self):
        """Özellik önem derecelerini döndür."""
        if self.model is None:
            raise RuntimeError("Model henüz eğitilmedi")
        
        try:
            # Çoğu sınıflandırma modeli feature_importances_ özelliğine sahip
            imps = self.model.feature_importances_
        except:
            # Bazı modellerde farklı özellik isimler olabilir
            try:
                imps = np.abs(self.model.coef_)
                if len(imps.shape) > 1:
                    imps = np.mean(np.abs(imps), axis=0)
            except:
                # Eğer hiçbir özellik önem derecesi yoksa uniform dağılım
                n_features = len(self.feature_names) if self.feature_names else 1
                imps = np.ones(n_features) / n_features
        
        importance_dict = {n: float(v) for n, v in zip(self.feature_names, imps)}
        return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    def predict(self, X):
        """Tahmin yap."""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        
        return self.model.predict(X)
    
    def analyze_stat_correlations(self, file_paths):
        """
        İstatistikler arası korelasyon analizi yap.
        
        Args:
            file_paths: Veri dosyalarının yolları
            
        Returns:
            dict: Korelasyon analizi sonuçları
        """
        df = self.normalizer.merge_datasets(file_paths)
        
        if df.empty:
            raise ValueError("Veri yüklenemedi")
        
        # İstatistik sütunları
        stat_columns = [col for col in df.columns if any(x in col for x in 
                       ['shots', 'corners', 'fouls', 'possession', 'offsides'])]
        
        # Skor sütunları
        score_columns = ['home_score', 'away_score']
        
        # Korelasyon matrisi
        corr_matrix = df[stat_columns + score_columns].corr()
        
        # En yüksek korelasyonlar
        score_correlations = {}
        for score_col in score_columns:
            correlations = corr_matrix[score_col].abs().sort_values(ascending=False)
            # Kendisi hariç
            correlations = correlations.drop(score_col)
            score_correlations[score_col] = correlations.head(10).to_dict()
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'top_correlations': score_correlations,
            'total_matches': len(df)
        }
