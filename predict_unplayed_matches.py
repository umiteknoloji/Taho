#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Oynanmamış Maçları Tespit ve Maç Sonucu Tahmin Sistemi
Bu script veri setindeki oynanmamış maçları otomatik tespit eder ve sadece maç sonucu (1X2) tahminleri yapar.
"""

import os
import warnings
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
import random
import logging

from core.data_normalizer import DataNormalizer

# Logging ayarları
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class UnplayedMatchPredictor:
    def __init__(self, random_seed=42):
        """Oynanmamış maç sonucu tahmin sistemini başlat - sadece 1X2 tahminleri"""
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        self.normalizer = DataNormalizer()
        self.all_data = None
        self.played_matches = None
        self.unplayed_matches = None
        self.models = {}
        
        print("🎯 Maç Sonucu Tahmin Sistemi (Sadece 1X2)")
        print("=" * 50)

    def parse_percentage(self, value):
        """Yüzde değerini float'a çevir"""
        try:
            if isinstance(value, str):
                if '%' in value:
                    return min(1.0, float(value.replace('%', '')) / 100)
                elif '/' in value:
                    # "5/15" gibi değerlerin sadece ilk kısmını al
                    parts = value.split('/')
                    return float(parts[0]) / 10.0 if float(parts[0]) > 1 else float(parts[0])
                else:
                    return min(1.0, float(value)) if float(value) > 1 else float(value)
            return float(value) if value is not None else 0.5
        except:
            return 0.5

    def parse_numeric(self, value):
        """Sayı değerini float'a çevir"""
        try:
            if isinstance(value, str):
                if '/' in value:
                    parts = value.split('/')
                    return float(parts[0])
                else:
                    return float(value)
            return float(value) if value is not None else 0.0
        except:
            return 0.0

    def load_and_analyze_data(self, file_path):
        """Veri dosyasını yükle ve oynanmamış maçları tespit et"""
        try:
            print(f"📊 Veri yükleniyor: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = f.read()
            
            # Veriyi normalize et
            self.all_data = self.normalizer.normalize_dataset(file_path)
            
            if self.all_data is None or len(self.all_data) == 0:
                print("❌ Veri normalize edilemedi")
                return False
            
            print(f"✅ Toplam {len(self.all_data)} maç yüklendi")
            
            # Oynanmış ve oynanmamış maçları ayır
            self._separate_matches()
            
            return True
            
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            return False

    def _separate_matches(self):
        """Oynanmış ve oynanmamış maçları ayır"""
        try:
            df = pd.DataFrame(self.all_data)
            
            # Oynanmış maçlar: hem home_score hem away_score var
            played_mask = (
                df['home_score'].notna() & 
                df['away_score'].notna() & 
                (df['home_score'] != '') & 
                (df['away_score'] != '')
            )
            
            self.played_matches = df[played_mask].copy()
            self.unplayed_matches = df[~played_mask].copy()
            
            print(f"🏁 Oynanmış maçlar: {len(self.played_matches)}")
            print(f"⏳ Oynanmamış maçlar: {len(self.unplayed_matches)}")
            
        except Exception as e:
            print(f"❌ Maç ayrıştırma hatası: {e}")

    def train_models(self, training_data):
        """Sadece maç sonucu tahmin modelini eğit"""
        try:
            print("🤖 Maç sonucu modeli eğitiliyor...")
            
            # Veriyi hazırla
            features_list = []
            match_results = []
            
            for _, match in training_data.iterrows():
                try:
                    # Temel özellikler
                    features = self._extract_features(match)
                    
                    if features is not None:
                        features_list.append(features)
                        
                        # Maç sonucunu hesapla (1X2)
                        home_score = int(match['home_score'])
                        away_score = int(match['away_score'])
                        
                        if home_score > away_score:
                            match_results.append("1")  # Ev sahibi galibiyeti
                        elif away_score > home_score:
                            match_results.append("2")  # Deplasman galibiyeti
                        else:
                            match_results.append("X")  # Beraberlik
                        
                except Exception as e:
                    continue
            
            if len(features_list) == 0:
                print("❌ Eğitim için yeterli veri yok")
                return False
            
            X = np.array(features_list)
            y = np.array(match_results)
            
            print(f"📊 Eğitim verisi: {len(X)} maç")
            print(f"📊 Sonuç dağılımı: 1={np.sum(y=='1')}, X={np.sum(y=='X')}, 2={np.sum(y=='2')}")
            
            # Direkt maç sonucu tahmini modeli (LightGBM)
            try:
                from lightgbm import LGBMClassifier
                self.models['result'] = LGBMClassifier(
                    objective='multiclass',
                    num_class=3,
                    max_depth=6,
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=self.random_seed,
                    verbose=-1
                )
                print("🚀 LightGBM modeli kullanılıyor")
            except ImportError:
                from sklearn.ensemble import RandomForestClassifier
                self.models['result'] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_seed,
                    class_weight='balanced'
                )
                print("🌲 Random Forest modeli kullanılıyor")
            
            # Modeli eğit
            self.models['result'].fit(X, y)
            
            print("✅ Maç sonucu modeli eğitildi")
            return True
            
        except Exception as e:
            print(f"❌ Model eğitimi hatası: {e}")
            return False

    def _extract_features(self, match):
        """Maç verilerinden özellik vektörü çıkar"""
        try:
            features = []
            
            # Temel skorlar
            features.extend([
                self.parse_numeric(match.get('home_goals_avg', 0)),
                self.parse_numeric(match.get('away_goals_avg', 0)),
                self.parse_numeric(match.get('home_goals_conceded_avg', 0)),
                self.parse_numeric(match.get('away_goals_conceded_avg', 0)),
            ])
            
            # Şut istatistikleri
            features.extend([
                self.parse_numeric(match.get('home_shots_avg', 0)),
                self.parse_numeric(match.get('away_shots_avg', 0)),
                self.parse_numeric(match.get('home_shots_on_target_avg', 0)),
                self.parse_numeric(match.get('away_shots_on_target_avg', 0)),
            ])
            
            # Top sahipliği
            features.extend([
                self.parse_percentage(match.get('home_possession_avg', 50)),
                self.parse_percentage(match.get('away_possession_avg', 50)),
            ])
            
            # Diğer istatistikler
            features.extend([
                self.parse_numeric(match.get('home_corners_avg', 0)),
                self.parse_numeric(match.get('away_corners_avg', 0)),
                self.parse_numeric(match.get('home_fouls_avg', 0)),
                self.parse_numeric(match.get('away_fouls_avg', 0)),
                self.parse_numeric(match.get('home_yellow_cards_avg', 0)),
                self.parse_numeric(match.get('away_yellow_cards_avg', 0)),
                self.parse_numeric(match.get('home_offsides_avg', 0)),
                self.parse_numeric(match.get('away_offsides_avg', 0)),
            ])
            
            return features
            
        except Exception as e:
            return None

    def predict_match(self, home_team, away_team, match_data):
        """Tek maç için sadece maç sonucu tahmin yap"""
        try:
            # Özellik vektörü hazırla
            features = self._extract_features(match_data)
            
            if features is None:
                return {
                    'predicted_result': 'Tahmin Edilemedi',
                    'result_meaning': 'Veri yetersiz',
                    'home_win_prob': 0.0,
                    'draw_prob': 0.0,
                    'away_win_prob': 0.0,
                    'confidence': 0.0
                }
            
            # Maç sonucu tahmini
            if 'result' in self.models:
                # Tahmin yap
                result_pred = self.models['result'].predict([features])[0]
                result_proba = self.models['result'].predict_proba([features])[0]
                
                # Olasılıkları parse et (sınıf sırası: "1", "2", "X" olabilir)
                classes = self.models['result'].classes_
                proba_dict = dict(zip(classes, result_proba))
                
                home_win_prob = proba_dict.get("1", 0.0)
                draw_prob = proba_dict.get("X", 0.0)
                away_win_prob = proba_dict.get("2", 0.0)
                
                # En yüksek olasılığı güven olarak kullan (0-1 arasında sınırla)
                confidence = min(max(max(result_proba), 0.0), 1.0)
                
            else:
                # Model yoksa varsayılan değerler
                result_pred = "X"
                home_win_prob = draw_prob = away_win_prob = 0.33
                confidence = 0.33
            
            # Sonuç açıklaması
            result_meanings = {
                "1": "Ev Sahibi Galibiyeti",
                "X": "Beraberlik", 
                "2": "Deplasman Galibiyeti"
            }
            
            return {
                'predicted_result': result_pred,
                'result_meaning': result_meanings.get(result_pred, 'Bilinmeyen'),
                'home_win_prob': float(home_win_prob * 100),  # 0-100 arasında yüzde
                'draw_prob': float(draw_prob * 100),          # 0-100 arasında yüzde
                'away_win_prob': float(away_win_prob * 100),  # 0-100 arasında yüzde
                'confidence': float(confidence * 100)         # 0-100 arasında yüzde
            }
            
        except Exception as e:
            print(f"❌ Tahmin hatası: {e}")
            return {
                'predicted_result': 'Hata',
                'result_meaning': 'Tahmin hatası',
                'home_win_prob': 0.0,
                'draw_prob': 0.0,
                'away_win_prob': 0.0,
                'confidence': 0.0
            }

def main():
    """Ana fonksiyon"""
    predictor = UnplayedMatchPredictor()
    
    # Test verisi
    test_file = 'data/ALM_stat.json'
    
    if predictor.load_and_analyze_data(test_file):
        if len(predictor.played_matches) > 0:
            # Modeli eğit
            predictor.train_models(predictor.played_matches)
            
            # Oynanmamış maçları tahmin et
            if len(predictor.unplayed_matches) > 0:
                print("\n🎯 Oynanmamış Maç Sonucu Tahminleri:")
                print("=" * 50)
                
                sample_matches = predictor.unplayed_matches.head(5)
                for idx, match in sample_matches.iterrows():
                    home_team = match.get('home', 'N/A')
                    away_team = match.get('away', 'N/A')
                    
                    prediction = predictor.predict_match(home_team, away_team, match)
                    
                    print(f"\n🏟️ {home_team} vs {away_team}")
                    print(f"🎯 Tahmin: {prediction['predicted_result']} - {prediction['result_meaning']}")
                    print(f"🏠 Ev Sahibi Galibiyeti: {prediction['home_win_prob']*100:.1f}%")
                    print(f"⚖️ Beraberlik: {prediction['draw_prob']*100:.1f}%") 
                    print(f"✈️ Deplasman Galibiyeti: {prediction['away_win_prob']*100:.1f}%")
                    print(f"💪 Güven: {prediction['confidence']*100:.1f}%")
            else:
                print("ℹ️ Oynanmamış maç bulunamadı")
        else:
            print("❌ Eğitim için yeterli oynanmış maç yok")
    else:
        print("❌ Veri yükleme başarısız")

if __name__ == "__main__":
    main()