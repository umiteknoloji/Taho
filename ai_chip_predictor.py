#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Chip Gaussian Process Futbol Tahmin Sistemi
M2 ve M3 Neural Engine'leri kullanarak Gaussian Process Classification ile tahmin
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
from gp_ai_chip_manager import AIChipFootballPredictor

# Logging ayarları
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class AIChipUnplayedMatchPredictor:
    def __init__(self, random_seed=42):
        """AI Chip Gaussian Process tahmin sistemi"""
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        self.normalizer = DataNormalizer()
        self.ai_predictor = AIChipFootballPredictor(random_seed)
        self.all_data = None
        self.played_matches = None
        self.unplayed_matches = None
        
        print("🧠 AI Chip Gaussian Process Futbol Tahmin Sistemi")
        print("⚡ M2 ve M3 Neural Engine'leri kullanılıyor")
        print("=" * 60)

    def load_and_analyze_data(self, file_path):
        """Veri yükle ve normalize et"""
        try:
            print(f"📊 Veri yükleniyor: {file_path}")
            
            # Normalize edilmiş veriyi al
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

    def train_models(self):
        """AI Chip'lerde Gaussian Process model eğitimi"""
        try:
            if len(self.played_matches) == 0:
                print("❌ Eğitim için oynanmış maç bulunamadı")
                return False
            
            print("🧠 AI Chip'lerde Gaussian Process eğitimi başlıyor...")
            
            # Training data hazırla
            training_data = self._prepare_training_data()
            
            if training_data is None:
                print("❌ Training data hazırlanamadı")
                return False
            
            # AI Chip'lerde eğit
            self.ai_predictor.train_models(training_data)
            
            print("✅ Gaussian Process model eğitimi tamamlandı")
            return True
            
        except Exception as e:
            print(f"❌ Model eğitimi hatası: {e}")
            return False
    
    def _prepare_training_data(self):
        """Training data hazırla"""
        try:
            training_data = []
            
            for _, match in self.played_matches.iterrows():
                try:
                    # Maç sonucunu hesapla
                    home_score = int(match['home_score'])
                    away_score = int(match['away_score'])
                    
                    if home_score > away_score:
                        result = '1'  # Ev sahibi galibiyeti
                    elif away_score > home_score:
                        result = '2'  # Deplasman galibiyeti
                    else:
                        result = 'X'  # Beraberlik
                    
                    # AI Chip için optimize edilmiş features
                    home_stats = self._extract_team_stats(match, 'home')
                    away_stats = self._extract_team_stats(match, 'away')
                    
                    # H2H stats
                    h2h_stats = self._extract_h2h_stats(match)
                    
                    match_data = {
                        'result': result,
                        'home_team_stats': home_stats,
                        'away_team_stats': away_stats,
                        'h2h_home_wins': h2h_stats.get('home_wins', 0),
                        'h2h_away_wins': h2h_stats.get('away_wins', 0),
                        'h2h_total': h2h_stats.get('total', 1)
                    }
                    
                    training_data.append(match_data)
                    
                except Exception as e:
                    continue
            
            if len(training_data) == 0:
                return None
            
            df = pd.DataFrame(training_data)
            print(f"📊 Training data: {len(df)} maç")
            print(f"📊 Sonuç dağılımı: 1={np.sum(df['result']=='1')}, X={np.sum(df['result']=='X')}, 2={np.sum(df['result']=='2')}")
            
            return df
            
        except Exception as e:
            print(f"❌ Training data hazırlama hatası: {e}")
            return None
    
    def _extract_team_stats(self, match, side):
        """Takım istatistiklerini çıkar"""
        try:
            stats = {
                'wins': self._parse_numeric(match.get(f'{side}_wins', 0)),
                'matches': max(1, self._parse_numeric(match.get(f'{side}_matches', 1))),
                'goals_for': self._parse_numeric(match.get(f'{side}_goals_avg', 0)),
                'goals_against': self._parse_numeric(match.get(f'{side}_goals_conceded_avg', 0)),
                'recent_form': self._parse_percentage(match.get(f'{side}_form', 0.5))
            }
            return stats
        except:
            return {'wins': 0, 'matches': 1, 'goals_for': 0, 'goals_against': 0, 'recent_form': 0.5}
    
    def _extract_h2h_stats(self, match):
        """Head-to-head istatistikleri"""
        try:
            return {
                'home_wins': self._parse_numeric(match.get('h2h_home_wins', 0)),
                'away_wins': self._parse_numeric(match.get('h2h_away_wins', 0)),
                'total': max(1, self._parse_numeric(match.get('h2h_total', 1)))
            }
        except:
            return {'home_wins': 0, 'away_wins': 0, 'total': 1}
    
    def _parse_numeric(self, value):
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
    
    def _parse_percentage(self, value):
        """Yüzde değerini float'a çevir"""
        try:
            if isinstance(value, str):
                if '%' in value:
                    return min(1.0, float(value.replace('%', '')) / 100)
                elif '/' in value:
                    parts = value.split('/')
                    return float(parts[0]) / 10.0 if float(parts[0]) > 1 else float(parts[0])
                else:
                    return min(1.0, float(value)) if float(value) > 1 else float(value)
            return float(value) if value is not None else 0.5
        except:
            return 0.5
    
    def predict_unplayed_matches(self):
        """Oynanmamış maçları AI Chip'lerde tahmin et"""
        try:
            if len(self.unplayed_matches) == 0:
                print("ℹ️ Oynanmamış maç bulunamadı")
                return []
            
            print(f"🎯 {len(self.unplayed_matches)} oynanmamış maç tahmin ediliyor...")
            
            # Batch prediction için maç listesi hazırla
            match_list = []
            for _, match in self.unplayed_matches.iterrows():
                home_team = match.get('home', 'N/A')
                away_team = match.get('away', 'N/A')
                week = match.get('week', None)
                match_list.append((home_team, away_team, week))
            
            # AI Chip'lerde distributed batch prediction
            predictions, device_usage = self.ai_predictor.predict_matches_batch(match_list)
            
            print(f"✅ Tahminler tamamlandı:")
            print(f"   M2 Neural Engine: {device_usage.get('M2_Neural', 0)} tahmin")
            print(f"   M3 Neural Engine: {device_usage.get('M3_Neural', 0)} tahmin")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Batch tahmin hatası: {e}")
            return []
    
    def predict_single_match(self, home_team, away_team, week=None):
        """Tek maç tahmini"""
        try:
            prediction = self.ai_predictor.predict_match(home_team, away_team, week)
            return prediction
        except Exception as e:
            print(f"❌ Tek maç tahmin hatası: {e}")
            return self._fallback_prediction(home_team, away_team)
    
    def _fallback_prediction(self, home_team, away_team):
        """Fallback prediction"""
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_result': '1',
            'result_meaning': 'Ev Sahibi Galibiyeti',
            'home_win_prob': 45.0,
            'draw_prob': 30.0,
            'away_win_prob': 25.0,
            'confidence': 45.0,
            'method': 'Fallback'
        }
    
    def get_neural_engine_status(self):
        """Neural Engine durumlarını al"""
        return self.ai_predictor.get_neural_engine_stats()
    
    def display_predictions(self, predictions):
        """Tahminleri güzel bir şekilde göster"""
        if not predictions:
            print("❌ Tahmin bulunamadı")
            return
        
        print(f"\n🎯 AI CHIP GAUSSIAN PROCESS TAHMİNLERİ")
        print("=" * 70)
        
        for i, pred in enumerate(predictions[:10], 1):  # İlk 10 tahmini göster
            print(f"\n{i:2d}. 🏟️ {pred['home_team']} vs {pred['away_team']}")
            print(f"     🎯 Tahmin: {pred['predicted_result']} - {pred['result_meaning']}")
            print(f"     🏠 Ev Sahibi: {pred['home_win_prob']:.1f}%")
            print(f"     ⚖️ Beraberlik: {pred['draw_prob']:.1f}%")
            print(f"     ✈️ Deplasman: {pred['away_win_prob']:.1f}%")
            print(f"     💪 Güven: {pred['confidence']:.1f}%")
            print(f"     🧠 Method: {pred.get('method', 'GP_Neural_Engine')}")
        
        if len(predictions) > 10:
            print(f"\n... ve {len(predictions) - 10} maç daha")
    
    def run_full_analysis(self, file_path):
        """Tam analiz çalıştır"""
        try:
            # 1. Veri yükle
            if not self.load_and_analyze_data(file_path):
                return False
            
            # 2. Model eğit
            if not self.train_models():
                return False
            
            # 3. Neural Engine durumu
            stats = self.get_neural_engine_status()
            print(f"\n🧠 Neural Engine Durumu:")
            print(f"   M2: {stats['M2_Neural_Engine']['status']} (Load: {stats['M2_Neural_Engine']['neural_load']:.1f}%)")
            print(f"   M3: {stats['M3_Neural_Engine']['status']} (Load: {stats['M3_Neural_Engine']['neural_load']:.1f}%)")
            
            # 4. Tahminleri yap
            predictions = self.predict_unplayed_matches()
            
            # 5. Sonuçları göster
            self.display_predictions(predictions)
            
            return True
            
        except Exception as e:
            print(f"❌ Tam analiz hatası: {e}")
            return False

def main():
    """Ana fonksiyon"""
    predictor = AIChipUnplayedMatchPredictor()
    
    # Test verisi
    test_file = 'data/ALM_stat.json'
    
    print("🚀 AI Chip Gaussian Process Tahmin Sistemi Test Ediliyor...")
    
    if os.path.exists(test_file):
        predictor.run_full_analysis(test_file)
    else:
        print(f"❌ Test dosyası bulunamadı: {test_file}")
        print("📂 Mevcut data dosyalarını kontrol edin")

if __name__ == "__main__":
    main()
