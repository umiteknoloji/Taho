#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parametrik Backtest Sistemi
Herhangi bir hafta için backtest çalıştırır
"""

import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import argparse
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import random

# Proje modüllerini import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.data_normalizer import DataNormalizer
    from elo_rating import ELORatingSystem
    from football_gp_classifier import FootballGPClassifier
except ImportError as e:
    print(f"⚠️ Modül import hatası: {e}")
    # Fallback - basit modeller kullan
    print("🔄 Basit modeller kullanılacak...")

class ParametricBacktestSystem:
    """Parametrik Backtest Sistemi - Herhangi bir hafta için test"""
    
    def __init__(self, use_enhanced_features: bool = True, use_ensemble: bool = True, use_gp: bool = True, random_seed: int = 42):
        self.use_enhanced_features = use_enhanced_features
        self.use_ensemble = use_ensemble
        self.use_gp = use_gp
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # 1X2 Maç sonucu sınıflandırıcısı 
        self.result_classifier = None
        self.gp_classifier = None  # GP Classifier
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # FootballGPClassifier
        if self.use_gp:
            try:
                self.football_gp = FootballGPClassifier()
                print("✅ GP Classifier yüklendi")
            except:
                print("⚠️ GP Classifier yüklenemedi, standart modeller kullanılacak")
                self.use_gp = False
        
        # ELO Rating sistemi
        try:
            from elo_rating import ELORatingSystem
            self.elo_system = ELORatingSystem()
            self.use_elo = True
        except:
            print("⚠️ ELO rating sistemi yüklenemedi")
            self.elo_system = None
            self.use_elo = False
        
        # Modül yükleme
        # İleri seviye modül zorunlu, fallback yok
        # self.feature_extractor = EnhancedFeatureExtractor()  # Kaldırıldı
        # self.evaluator = AdvancedEvaluator()  # Kaldırıldı
        
        self.results_history = {}
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Veriyi yükle ve DataFrame'e çevir"""
        print(f"📊 Veri yükleniyor: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ {len(data)} maç yüklendi")
        
        # DataFrame'e çevir
        df_data = []
        for i, match in enumerate(data):
            try:
                # Temel maç bilgileri
                home_team = match.get('home', '')
                away_team = match.get('away', '')
                
                # Skor bilgileri
                score = match.get('score', {})
                full_time = score.get('fullTime', {}) if score else {}
                
                home_score = full_time.get('home', 0) if full_time else 0
                away_score = full_time.get('away', 0) if full_time else 0
                
                # Hafta bilgisi
                week = match.get('week', 0)
                if not week:
                    # Hafta bilgisi yoksa index'e göre hesapla (Alman ligi için 9 maç/hafta)
                    week = (i // 9) + 1
                else:
                    # Hafta bilgisini int'e çevir
                    week = int(week)
                
                # Tarih bilgisi (ELO determinizmi için)
                date = match.get('date', '')
                
                df_data.append({
                    'home_team': home_team,
                    'away_team': away_team, 
                    'home_score': int(home_score) if home_score else 0,
                    'away_score': int(away_score) if away_score else 0,
                    'week': week,
                    'date': date,
                    'match_index': i
                })
                
            except Exception as e:
                print(f"⚠️ Maç {i} işlenirken hata: {e}")
                continue
        
        df = pd.DataFrame(df_data)
        # Deterministiklik için veri sırasını sabitle
        df = df.sort_values(['week', 'home_team', 'away_team']).reset_index(drop=True)
        # Sıralama sonrası match_index'i yeniden oluştur (deterministik sıra için)
        df['match_index'] = df.index
        print(f"✅ {len(df)} maç DataFrame'e dönüştürüldü")
        print(f"📈 Hafta aralığı: {df['week'].min()} - {df['week'].max()}")
        
        return df
    
    def get_available_weeks(self, df: pd.DataFrame) -> List[int]:
        """Mevcut hafta numaralarını döndür"""
        return sorted(df['week'].unique().tolist())
    
    def extract_basic_features(self, df: pd.DataFrame, match_index: int) -> Dict[str, float]:
        """Basit özellik çıkarımı"""
        match = df.iloc[match_index]
        home_team = match['home_team']
        away_team = match['away_team']
        
        features = {}
        
        # Son 5 maçın performansı
        home_recent = self._get_recent_performance(df, home_team, match_index, True, 5)
        away_recent = self._get_recent_performance(df, away_team, match_index, False, 5)
        
        features.update({
            'home_goals_avg': home_recent['goals_scored'],
            'home_conceded_avg': home_recent['goals_conceded'],
            'home_wins_rate': home_recent['wins_rate'],
            'away_goals_avg': away_recent['goals_scored'],
            'away_conceded_avg': away_recent['goals_conceded'], 
            'away_wins_rate': away_recent['wins_rate'],
        })
        
        # Form özellikler
        home_form = self._get_form_score(df, home_team, match_index, True)
        away_form = self._get_form_score(df, away_team, match_index, False)
        
        features.update({
            'home_form': home_form,
            'away_form': away_form,
            'form_diff': home_form - away_form
        })
        
        # Gelişmiş istatistiksel özellikler
        features.update({
            'home_attack_strength': home_recent['goals_scored'] / max(away_recent['goals_conceded'], 0.5),
            'away_attack_strength': away_recent['goals_scored'] / max(home_recent['goals_conceded'], 0.5),
            'home_defense_strength': 2.0 - home_recent['goals_conceded'],
            'away_defense_strength': 2.0 - away_recent['goals_conceded'],
            'goal_avg_diff': home_recent['goals_scored'] - away_recent['goals_scored'],
            'conceded_diff': away_recent['goals_conceded'] - home_recent['goals_conceded'],
            'home_consistency': self._get_consistency_score(df, home_team, match_index),
            'away_consistency': self._get_consistency_score(df, away_team, match_index),
            'home_advantage': 1.0,  # Ev sahibi avantajı
            'momentum_diff': self._get_momentum_diff(df, home_team, away_team, match_index)
        })
        
        # ELO Rating özellikleri
        if self.use_elo and self.elo_system:
            try:
                elo_features = self.elo_system.get_team_strength_features(home_team, away_team)
                features.update(elo_features)
            except:
                # ELO sistemi hata verirse default değerler
                features.update({
                    'elo_home_rating': 1500,
                    'elo_away_rating': 1500,
                    'elo_rating_diff': 0,
                    'elo_home_win_prob': 0.5,
                    'elo_draw_prob': 0.3,
                    'elo_away_win_prob': 0.2,
                    'elo_home_strength': 1.0,
                    'elo_away_strength': 1.0,
                    'elo_strength_ratio': 1.0
                })
        else:
            # ELO sistemi yoksa default değerler
            features.update({
                'elo_home_rating': 1500,
                'elo_away_rating': 1500,
                'elo_rating_diff': 0,
                'elo_home_win_prob': 0.5,
                'elo_draw_prob': 0.3,
                'elo_away_win_prob': 0.2,
                'elo_home_strength': 1.0,
                'elo_away_strength': 1.0,
                'elo_strength_ratio': 1.0
            })
        
        return features
    
    def _get_recent_performance(self, df: pd.DataFrame, team: str, current_index: int, 
                              is_home: bool, window: int = 5) -> Dict[str, float]:
        """Son maçlardaki performans"""
        goals_scored = []
        goals_conceded = []
        results = []
        
        count = 0
        for i in range(current_index - 1, -1, -1):
            if count >= window:
                break
                
            match = df.iloc[i]
            if match['home_team'] == team:
                goals_scored.append(match['home_score'])
                goals_conceded.append(match['away_score'])
                if match['home_score'] > match['away_score']:
                    results.append(1)  # Win
                elif match['home_score'] == match['away_score']:
                    results.append(0.5)  # Draw
                else:
                    results.append(0)  # Loss
                count += 1
            elif match['away_team'] == team:
                goals_scored.append(match['away_score'])
                goals_conceded.append(match['home_score'])
                if match['away_score'] > match['home_score']:
                    results.append(1)  # Win
                elif match['away_score'] == match['home_score']:
                    results.append(0.5)  # Draw
                else:
                    results.append(0)  # Loss
                count += 1
        
        return {
            'goals_scored': sum(goals_scored) / len(goals_scored) if goals_scored else 1.0,
            'goals_conceded': sum(goals_conceded) / len(goals_conceded) if goals_conceded else 1.0,
            'wins_rate': sum(results) / len(results) if results else 0.5
        }
    
    def _get_form_score(self, df: pd.DataFrame, team: str, current_index: int, 
                       is_home: bool) -> float:
        """Takım formu (ağırlıklı) - Son 10 maç, en eski maçtan en yeni maça ağırlık artırır"""
        form_scores = []
        # Son 10 maç için ağırlıklar: en eski maçtan (0.05) en yeni maça (0.20) doğru artış
        weights = [0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.15, 0.17, 0.20]
        
        count = 0
        for i in range(current_index - 1, -1, -1):
            if count >= 10:  # Son 10 maç
                break
                
            match = df.iloc[i]
            if match['home_team'] == team:
                if match['home_score'] > match['away_score']:
                    form_scores.append(3)  # Win
                elif match['home_score'] == match['away_score']:
                    form_scores.append(1)  # Draw
                else:
                    form_scores.append(0)  # Loss
                count += 1
            elif match['away_team'] == team:
                if match['away_score'] > match['home_score']:
                    form_scores.append(3)  # Win
                elif match['away_score'] == match['home_score']:
                    form_scores.append(1)  # Draw
                else:
                    form_scores.append(0)  # Loss
                count += 1
        
        if not form_scores:
            return 1.5  # Neutral
        
        # form_scores[0] = en son maç, form_scores[-1] = en eski maç
        # Ağırlıkları tersine çevirmemiz gerek çünkü en yeni maç daha ağırlıklı olmalı
        form_scores.reverse()  # En eski maçtan en yeni maça doğru sırala
        
        # Ağırlıklı ortalama
        weighted_form = 0
        total_weight = 0
        for i, score in enumerate(form_scores):
            weight = weights[i] if i < len(weights) else weights[-1]  # Son ağırlığı kullan
            weighted_form += score * weight
            total_weight += weight
        
        return weighted_form / total_weight if total_weight > 0 else 1.5
    
    def train_classification_models(self, df: pd.DataFrame, train_weeks: int):
        """Sınıflandırma modellerini eğit"""
        print("🤖 Sınıflandırma modelleri eğitiliyor...")
        print(f"[LOG] Seed: {self.random_seed}")
        print(f"[LOG] Eğitim veri hash: {hash(df.to_string())}")
        
        # Eğitim verilerini hazırla
        train_data = df[df['week'] <= train_weeks]
        
        features_list = []
        home_scores = []
        away_scores = []
        results = []  # 1X2 için
        
        for _, match in train_data.iterrows():
            try:
                features = self.extract_basic_features(df, match['match_index'])
                feature_vector = list(features.values())
                
                features_list.append(feature_vector)
                home_scores.append(min(match['home_score'], 5))  # Max 5 gol sınırı
                away_scores.append(min(match['away_score'], 5))  # Max 5 gol sınırı
                
                # Sonuç sınıflandırması (0: Deplasman, 1: Beraberlik, 2: Ev sahibi)
                if match['home_score'] > match['away_score']:
                    results.append(2)  # Ev sahibi kazandı
                elif match['home_score'] == match['away_score']:
                    results.append(1)  # Beraberlik
                else:
                    results.append(0)  # Deplasman kazandı
                    
            except Exception as e:
                print(f"⚠️ Özellik çıkarımı hatası: {e}")
                continue
        
        if len(features_list) < 10:
            print("⚠️ Yetersiz eğitim verisi")
            return False
        
        # Features'ları normalize et
        X = np.array(features_list)
        X_scaled = self.scaler.fit_transform(X)
        
        # GP Classifier ile başla
        if self.use_gp:
            try:
                # GP Classifier - Kernel kombinasyonu
                kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                self.gp_classifier = GaussianProcessClassifier(
                    kernel=kernel,
                    random_state=self.random_seed,
                    n_restarts_optimizer=2,
                    max_iter_predict=100
                )
                
                # Eğitim verisi boyutunu kontrol et (GP için)
                if len(X_scaled) > 500:
                    # Çok büyük veri seti için örnekleme
                    from sklearn.utils import resample
                    X_sampled, y_sampled = resample(X_scaled, results, 
                                                   n_samples=500, 
                                                   random_state=self.random_seed,
                                                   stratify=results)
                    self.gp_classifier.fit(X_sampled, y_sampled)
                    print("🧠 GP Classifier eğitildi (örneklenmiş veri ile)")
                else:
                    self.gp_classifier.fit(X_scaled, results)
                    print("🧠 GP Classifier eğitildi")
                
                # FootballGPClassifier'ı da eğit
                if hasattr(self, 'football_gp') and self.football_gp:
                    try:
                        # Veriyi FootballGPClassifier formatına çevir
                        gp_data = []
                        for i, (features, result) in enumerate(zip(features_list, results)):
                            # Features sözlüğü formatında hazırla
                            feature_dict = {}
                            feature_names = [
                                'home_goals_avg', 'home_conceded_avg', 'home_wins_rate',
                                'away_goals_avg', 'away_conceded_avg', 'away_wins_rate',
                                'home_form', 'away_form', 'form_diff'
                            ]
                            for j, name in enumerate(feature_names[:len(features)]):
                                feature_dict[name] = features[j]
                            
                            gp_data.append({
                                'features': feature_dict,
                                'result': result
                            })
                        
                        self.football_gp.train(gp_data)
                        print("⚽ FootballGPClassifier eğitildi")
                    except Exception as e:
                        print(f"⚠️ FootballGPClassifier eğitim hatası: {e}")
                
                # Ana classifier olarak GP'yi ayarla
                self.result_classifier = self.gp_classifier
                
            except Exception as e:
                print(f"⚠️ GP Classifier eğitim hatası: {e}")
                self.use_gp = False
                # Fallback'e geç
                
        # GP başarısız olduysa veya kullanılmıyorsa standart modeller
        if not self.use_gp or self.result_classifier is None:
            try:
                from lightgbm import LGBMClassifier
                self.result_classifier = LGBMClassifier(
                    objective='multiclass',
                    num_class=3,
                    max_depth=8,
                    n_estimators=150,
                    learning_rate=0.1,
                    random_state=self.random_seed,
                    verbose=-1
                )
                print("🚀 LightGBM kullanılıyor (1X2 için)")
            except ImportError:
                self.result_classifier = GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=10,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=self.random_seed
                )
                print("🌲 GradientBoosting kullanılıyor (1X2 için)")
            
            self.result_classifier.fit(X_scaled, results)
        
        print(f"[LOG] Model parametreleri: result_classifier={self.result_classifier.get_params()}")
        self.is_trained = True
        print("✅ Sınıflandırma modelleri eğitildi")
        
        # Eğitim başarısını test et
        # Sadece 1X2 sonuç doğruluğunu kontrol et
        result_pred = self.result_classifier.predict(X_scaled)
        result_acc = accuracy_score(results, result_pred)
        
        print(f"📊 Eğitim doğruluk oranı:")
        print(f"   1X2 Sonuç: {result_acc:.3f}")
        
        return True
    
    def classification_predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """1X2 sınıflandırma tabanlı tahmin"""
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        if not self.is_trained:
            print("⚠️ Modeller henüz eğitilmedi, fallback tahmin kullanılıyor")
            return {
                'predicted_result': 'X',
                'result_meaning': 'Beraberlik (Fallback)',
                'home_win_prob': 33.0,  # Yüzde formatında
                'draw_prob': 34.0,      # Yüzde formatında
                'away_win_prob': 33.0,  # Yüzde formatında
                'confidence': 34.0      # Yüzde formatında
            }
        
        try:
            # Feature vector'ü hazırla
            feature_vector = np.array([list(features.values())]).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # GP Classifier kullanılıyorsa öncelik ver
            if self.use_gp and self.gp_classifier is not None:
                try:
                    # GP ile tahmin yap
                    result_pred = self.gp_classifier.predict(feature_vector_scaled)[0]
                    result_proba = self.gp_classifier.predict_proba(feature_vector_scaled)[0]
                    
                    # FootballGPClassifier ile de tahmin yap ve sonuçları birleştir
                    if hasattr(self, 'football_gp') and self.football_gp:
                        try:
                            # Features'ı sözlük formatına çevir
                            feature_names = [
                                'home_goals_avg', 'home_conceded_avg', 'home_wins_rate',
                                'away_goals_avg', 'away_conceded_avg', 'away_wins_rate',
                                'home_form', 'away_form', 'form_diff'
                            ]
                            feature_dict = {}
                            feature_values = list(features.values())
                            for i, name in enumerate(feature_names[:len(feature_values)]):
                                feature_dict[name] = feature_values[i]
                            
                            gp_prediction = self.football_gp.predict(feature_dict)
                            
                            # İki GP tahmini arasında ensemble yap
                            if gp_prediction and 'probabilities' in gp_prediction:
                                gp_proba = gp_prediction['probabilities']
                                # Ağırlıklı ortalama (sklearn GP: 0.7, FootballGP: 0.3)
                                ensemble_proba = 0.7 * result_proba + 0.3 * np.array([
                                    gp_proba.get('away_win', 0.33),
                                    gp_proba.get('draw', 0.33), 
                                    gp_proba.get('home_win', 0.33)
                                ])
                                result_proba = ensemble_proba
                                
                                print("🔮 GP Ensemble tahmini kullanıldı")
                        except Exception as e:
                            print(f"⚠️ FootballGP ensemble hatası: {e}")
                    
                    print("🧠 GP Classifier kullanıldı")
                    
                except Exception as e:
                    print(f"⚠️ GP tahmin hatası: {e}")
                    # Standart classifier'a fallback
                    result_pred = self.result_classifier.predict(feature_vector_scaled)[0]
                    result_proba = self.result_classifier.predict_proba(feature_vector_scaled)[0]
                    print("🔄 Standart classifier'a geçildi")
            else:
                # Standart 1X2 sonuç tahminleri
                result_pred = self.result_classifier.predict(feature_vector_scaled)[0]
                result_proba = self.result_classifier.predict_proba(feature_vector_scaled)[0]
            
            # Olasılıkları parse et
            classes = self.result_classifier.classes_  # [0, 1, 2]
            proba_dict = dict(zip(classes, result_proba))
            
            # Sınıf indeksini 1X2 formatına çevir
            class_to_result = {0: "2", 1: "X", 2: "1"}  # 0:Deplasman, 1:Beraberlik, 2:Ev sahibi
            predicted_result_1x2 = class_to_result.get(result_pred, "X")
            
            away_win_prob = proba_dict.get(0, 0.0)  # Sınıf 0 = Deplasman
            draw_prob = proba_dict.get(1, 0.0)      # Sınıf 1 = Beraberlik  
            home_win_prob = proba_dict.get(2, 0.0)  # Sınıf 2 = Ev sahibi
            
            # En yüksek olasılığı güven olarak kullan (0-1 arasında sınırla)
            confidence = min(max(max(result_proba), 0.0), 1.0)
            
            # Sonuç açıklaması
            result_meanings = {
                "1": "Ev Sahibi Galibiyeti",
                "X": "Beraberlik", 
                "2": "Deplasman Galibiyeti"
            }
            
            # GP kullanımı durumunda açıklamaya ekle
            result_meaning = result_meanings.get(predicted_result_1x2, 'Bilinmeyen')
            if self.use_gp and self.gp_classifier is not None:
                result_meaning += " (GP)"
            
            return {
                'predicted_result': predicted_result_1x2,
                'result_meaning': result_meaning,
                'home_win_prob': float(home_win_prob * 100),  # 0-100 arasında yüzde
                'draw_prob': float(draw_prob * 100),          # 0-100 arasında yüzde
                'away_win_prob': float(away_win_prob * 100),  # 0-100 arasında yüzde
                'confidence': float(confidence * 100)         # 0-100 arasında yüzde
            }
            
        except Exception as e:
            print(f"⚠️ 1X2 tahmini hatası: {e}")
            return {
                'predicted_result': 'X',
                'result_meaning': 'Beraberlik (Hata)',
                'home_win_prob': 33.0,  # Yüzde formatında
                'draw_prob': 34.0,      # Yüzde formatında
                'away_win_prob': 33.0,  # Yüzde formatında
                'confidence': 34.0      # Yüzde formatında
            }
    
    def run_backtest(self, data_path: str, test_week: int, 
                    train_weeks: Optional[int] = None) -> Dict[str, Any]:
        """Belirtilen hafta için backtest çalıştır"""
        print(f"🎯 {test_week}. HAFTA BACKTESTİ")
        print("=" * 50)
        
        # Veriyi yükle
        df = self.load_data(data_path)
        
        # Hafta kontrolü
        available_weeks = self.get_available_weeks(df)
        if test_week not in available_weeks:
            raise ValueError(f"Hafta {test_week} bulunamadı. Mevcut haftalar: {available_weeks}")
        
        # Eğitim ve test setlerini ayır
        if train_weeks is None:
            train_weeks = test_week - 1
        
        train_data = df[df['week'] <= train_weeks]
        test_data = df[df['week'] == test_week]
        
        print(f"📈 Eğitim verileri: {len(train_data)} maç ({train_weeks} hafta)")
        print(f"🎯 Test verileri: {len(test_data)} maç ({test_week}. hafta)")
        
        if len(test_data) == 0:
            raise ValueError(f"{test_week}. hafta için test verisi bulunamadı!")
        
        # Oynanmış ve oynanmamış maçları ayır
        played_test_data = test_data[(test_data['home_score'] > 0) | (test_data['away_score'] > 0)]
        unplayed_test_data = test_data[(test_data['home_score'] == 0) & (test_data['away_score'] == 0)]
        
        # ELO rating sistemini eğit (tüm geçmiş maçlarla)
        if self.use_elo and self.elo_system:
            print("🔢 ELO rating sistemi eğitiliyor...")
            # Tarih sırasına göre sırala (determinizm için)
            train_data_sorted = train_data.sort_values(['week', 'date', 'home_team', 'away_team']).reset_index(drop=True)
            train_matches = []
            for _, match in train_data_sorted.iterrows():
                train_matches.append({
                    'home': match['home_team'],
                    'away': match['away_team'],
                    'date': match['date'],
                    'week': match['week'],
                    'score': {
                        'fullTime': {
                            'home': match['home_score'],
                            'away': match['away_score']
                        }
                    }
                })
            if train_matches:
                self.elo_system.train_from_matches(train_matches)
        
        # Sınıflandırma modellerini eğit
        if len(train_data) >= 10:  # Minimum eğitim verisi kontrolü
            self.train_classification_models(df, train_weeks)
        else:
            print("⚠️ Yetersiz eğitim verisi, temel tahmin kullanılacak")
        
        # Eğer sadece oynanmamış maçlar varsa, onları tahmin et
        if len(played_test_data) == 0 and len(unplayed_test_data) > 0:
            print(f"🔮 Sadece oynanmamış maçlar bulundu: {len(unplayed_test_data)} maç")
            predictions = []
            match_names = []
            
            for _, test_match in unplayed_test_data.iterrows():
                try:
                    features = self.extract_basic_features(df, test_match['match_index'])
                    # 1X2 tahmin yap
                    prediction = self.classification_predict(features)
                    predictions.append((prediction['predicted_result'], prediction))
                    match_names.append(f"{test_match['home_team']} vs {test_match['away_team']}")
                    
                    # Sonucu göster
                    print(f"🔮 Tahmin: {test_match['home_team']} vs {test_match['away_team']} | {prediction['predicted_result']} - {prediction['result_meaning']}")
                    print(f"   Güven: {prediction['confidence']:.1f}%")
                    
                except Exception as e:
                    print(f"❌ Tahmin hatası: {e}")
                    predictions.append((1, 1))
                    match_names.append(f"{test_match['home_team']} vs {test_match['away_team']}")
                    
            print(f"\n🎯 {test_week}. haftada oynanmamış {len(predictions)} maç için tahmin üretildi.")
            return {
                'test_week': test_week,
                'unplayed_predictions': list(zip(match_names, predictions)),
                'timestamp': datetime.now().isoformat()
            }
        
        # Normal backtest - oynanmış maçlar için
        predictions = []
        actuals = []
        match_names = []
        
        for _, test_match in played_test_data.iterrows():
            try:
                # Test maçının özelliklerini çıkar
                features = self.extract_basic_features(df, test_match['match_index'])
                
                # 1X2 tahmin yap
                prediction = self.classification_predict(features)
                
                predictions.append((test_match['home_team'] + " vs " + test_match['away_team'], prediction, 
                                  (test_match['home_score'], test_match['away_score'])))
                match_names.append(f"{test_match['home_team']} vs {test_match['away_team']}")
                
                # Gerçek maç sonucu (1X2)
                if test_match['home_score'] > test_match['away_score']:
                    actual_result = "1"  # Ev sahibi kazandı
                elif test_match['home_score'] == test_match['away_score']:
                    actual_result = "X"  # Beraberlik
                else:
                    actual_result = "2"  # Deplasman kazandı
                
                # Tahmin doğruluğu
                pred_result = prediction['predicted_result']
                is_correct = pred_result == actual_result
                status = "✅" if is_correct else "❌"
                
                print(f"{status} {test_match['home_team']} vs {test_match['away_team']}")
                print(f"   Tahmin: {pred_result} - {prediction['result_meaning']} (Güven: {prediction['confidence']*100:.1f}%)")
                print(f"   Gerçek: {actual_result} | Skor: {test_match['home_score']}-{test_match['away_score']}")
                
            except Exception as e:
                print(f"❌ Tahmin hatası: {e}")
                # Fallback 1X2 tahmin
                fallback_pred = {
                    'predicted_result': 'X',
                    'result_meaning': 'Beraberlik (Fallback)',
                    'home_win_prob': 33.0,  # Yüzde formatında
                    'draw_prob': 34.0,      # Yüzde formatında
                    'away_win_prob': 33.0,  # Yüzde formatında
                    'confidence': 34.0      # Yüzde formatında
                }
                predictions.append((test_match['home_team'] + " vs " + test_match['away_team'], fallback_pred, 
                                  (test_match['home_score'], test_match['away_score'])))
                match_names.append(f"{test_match['home_team']} vs {test_match['away_team']}")
        
        # Değerlendirme - 1X2 format için
        metrics = self.evaluate_1x2_predictions(predictions)
        
        # Sonuçları kaydet
        result = {
            'test_week': test_week,
            'train_weeks': train_weeks,
            'metrics': metrics,
            'predictions': predictions,  # 1X2 format: (match_name, prediction_dict, actual_scores)
            'summary': self.generate_summary(metrics, predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        self.results_history[test_week] = result
        
        # Özet yazdır
        print(f"\n📊 SONUÇLAR - {test_week}. HAFTA")
        print("=" * 30)
        print(f"Toplam maç: {len(predictions)}")
        print(f"1X2 Doğru tahmin: {metrics['correct_results']}")
        print(f"1X2 Başarı oranı: {metrics['result_accuracy']:.1%}")
        print(f"Ortalama güven: {metrics['avg_confidence']:.1%}")
        
        return result
    
    def evaluate_1x2_predictions(self, predictions: List[Tuple[str, Dict, Tuple[int, int]]]) -> Dict[str, float]:
        """1X2 tahminlerini değerlendir"""
        if not predictions:
            return {}
        
        correct_results = 0
        total_confidence = 0
        total = len(predictions)
        
        for match_name, prediction, (actual_home, actual_away) in predictions:
            try:
                # Gerçek sonucu hesapla
                if actual_home > actual_away:
                    actual_result = "1"  # Ev sahibi kazandı
                elif actual_home == actual_away:
                    actual_result = "X"  # Beraberlik
                else:
                    actual_result = "2"  # Deplasman kazandı
                
                # Tahmin doğruluğu
                pred_result = prediction.get('predicted_result', 'X')
                if pred_result == actual_result:
                    correct_results += 1
                
                # Güven skoru
                confidence = prediction.get('confidence', 0.33)
                total_confidence += confidence
                
            except Exception as e:
                print(f"⚠️ Değerlendirme hatası: {e}")
                continue
        
        return {
            'correct_results': correct_results,
            'result_accuracy': correct_results / total if total > 0 else 0,
            'avg_confidence': total_confidence / total if total > 0 else 0,
            'total_matches': total
        }
        
        exact_matches = 0
        close_predictions = 0
        correct_results = 0
        total_goals_diff = 0
        total = len(predictions)
        
        for idx, ((pred_h, pred_a), (actual_h, actual_a)) in enumerate(zip(predictions, actuals)):
            # Tam skor
            if pred_h == actual_h and pred_a == actual_a:
                exact_matches += 1
            # Yakın tahmin (her skor için max 1 fark)
            if abs(pred_h - actual_h) <= 1 and abs(pred_a - actual_a) <= 1:
                close_predictions += 1
            # Sonuç doğruluğu (1X2)
            pred_result = 'X' if pred_h == pred_a else ('1' if pred_h > pred_a else '2')
            actual_result = 'X' if actual_h == actual_a else ('1' if actual_h > actual_a else '2')
            if pred_result == actual_result:
                correct_results += 1
            # Toplam gol farkı
    
    def generate_summary(self, metrics: Dict[str, float], predictions: List, actuals: List = None) -> str:
        """Özet rapor oluştur - 1X2 format için"""
        summary = []
        summary.append(f"📊 1X2 Backtest Özeti")
        summary.append(f"Toplam Maç: {metrics['total_matches']}")
        summary.append(f"1X2 Doğru Tahmin: {metrics['correct_results']}/{metrics['total_matches']} ({metrics['result_accuracy']:.1%})")
        summary.append(f"Ortalama Güven: {metrics['avg_confidence']:.1%}")
        summary.append(f"Tahmin Sistemi: Sadece 1X2 Maç Sonucu")
        return "\n".join(summary)
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Tüm sonuçların özeti"""
        if not self.results_history:
            return {"message": "Henüz backtest çalıştırılmadı"}
        
        summary = {
            "tested_weeks": list(self.results_history.keys()),
            "best_week": max(self.results_history.keys(), 
                           key=lambda w: self.results_history[w]['metrics']['accuracy']),
            "average_accuracy": sum([r['metrics']['accuracy'] for r in self.results_history.values()]) / len(self.results_history),
            "total_tests": len(self.results_history)
        }
        
        return summary

    def _get_consistency_score(self, df: pd.DataFrame, team: str, current_index: int) -> float:
        """Takımın istikrar skorunu hesapla (son 5 maçtaki gol sapması)"""
        goals = []
        count = 0
        
        for i in range(current_index - 1, -1, -1):
            if count >= 5:
                break
                
            match = df.iloc[i]
            if match['home_team'] == team:
                goals.append(match['home_score'])
                count += 1
            elif match['away_team'] == team:
                goals.append(match['away_score'])
                count += 1
        
        if len(goals) < 2:
            return 0.5  # Neutral
        
        # Standart sapmanın tersi (düşük sapma = yüksek istikrar)
        std_dev = np.std(goals)
        return max(0, 1.0 - (std_dev / 3.0))  # 0-1 arası normalize
    
    def _get_momentum_diff(self, df: pd.DataFrame, home_team: str, away_team: str, current_index: int) -> float:
        """Son 3 maçın momentum farkını hesapla"""
        home_momentum = self._get_team_momentum(df, home_team, current_index)
        away_momentum = self._get_team_momentum(df, away_team, current_index)
        return home_momentum - away_momentum
    
    def _get_team_momentum(self, df: pd.DataFrame, team: str, current_index: int) -> float:
        """Takımın momentumunu hesapla (son 3 maç ağırlıklı)"""
        points = []
        weights = [0.6, 0.4]  # Son 2 maç: en yeniye daha fazla ağırlık
        count = 0
        
        for i in range(current_index - 1, -1, -1):
            if count >= 2:
                break
                
            match = df.iloc[i]
            if match['home_team'] == team:
                if match['home_score'] > match['away_score']:
                    points.append(3)
                elif match['home_score'] == match['away_score']:
                    points.append(1)
                else:
                    points.append(0)
                count += 1
            elif match['away_team'] == team:
                if match['away_score'] > match['home_score']:
                    points.append(3)
                elif match['away_score'] == match['home_score']:
                    points.append(1)
                else:
                    points.append(0)
                count += 1
        
        if not points:
            return 1.0  # Neutral
        
        # Ağırlıklı momentum
        momentum = 0
        for i, point in enumerate(points):
            weight = weights[i] if i < len(weights) else 0.1
            momentum += point * weight
        
        return momentum / 3.0  # 0-1 arası normalize

def main():
    """Ana fonksiyon - CLI kullanımı"""
    parser = argparse.ArgumentParser(description='Parametrik Backtest Sistemi')
    parser.add_argument('--week', type=int, required=True, help='Test edilecek hafta numarası')
    parser.add_argument('--data', type=str, default='data/ALM_stat.json', help='Veri dosyası yolu')
    parser.add_argument('--train-weeks', type=int, help='Eğitim hafta sayısı (varsayılan: test_week - 1)')
    parser.add_argument('--output', type=str, help='Sonuç dosyası yolu')
    parser.add_argument('--seed', type=int, default=42, help='Rastgelelik için seed (varsayılan: 42)')
    
    args = parser.parse_args()
    
    # Backtest sistemi oluştur
    system = ParametricBacktestSystem(random_seed=args.seed)
    
    try:
        # Backtest çalıştır
        result = system.run_backtest(args.data, args.week, args.train_weeks)
        
        # Sonucu kaydet
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✅ Sonuçlar kaydedildi: {args.output}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parametrik Backtest Sistemi")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik için seed (varsayılan: 42)")
    args, unknown = parser.parse_known_args()
    seed = args.seed
    if len(sys.argv) == 1:
        # Etkileşimli mod
        print("🎯 PARAMETRIK BACKTEST SİSTEMİ")
        print("=" * 50)
        system = ParametricBacktestSystem(random_seed=seed)
        try:
            # Veri dosyasını kontrol et
            data_path = "data/ALM_stat.json"
            if not os.path.exists(data_path):
                print(f"❌ Veri dosyası bulunamadı: {data_path}")
                exit(1)
            # Mevcut haftaları göster
            df = system.load_data(data_path)
            weeks = system.get_available_weeks(df)
            print(f"📅 Mevcut haftalar: {weeks}")
            # Hafta seç
            week = int(input("\n🎯 Test edilecek hafta numarasını girin: "))
            # Backtest çalıştır
            result = system.run_backtest(data_path, week)
            print("\n🎉 Backtest tamamlandı!")
        except KeyboardInterrupt:
            print("\n👋 İptal edildi.")
        except Exception as e:
            print(f"\n❌ Hata: {e}")
    else:
        # CLI mod
        exit(main())
