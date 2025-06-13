#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M3 Neural Engine Futbol GP Classification Sistemi
Gaussian Process ile futbol maç sonucu (1X2) tahminleri
"""

import os
import warnings
os.environ["PYTHONHASHSEED"] = "42"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
import time
import psutil
from datetime import datetime

from core.data_normalizer import DataNormalizer

class FootballGPClassifier:
    """Futbol verileri için GP Classification sistemi"""
    
    def __init__(self):
        self.normalizer = DataNormalizer()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.gp_model = None
        self.feature_names = []
        self.kernel_performance = {}
        
        # Neural Engine monitoring
        self.neural_engine_usage = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'training_time': 0,
            'prediction_time': 0
        }
        
    def train(self, gp_data):
        """
        Basit eğitim metodu - parametric_backtest için
        gp_data: List[Dict] formatında [{'features': {...}, 'result': int}]
        """
        try:
            if not gp_data:
                print("⚠️ FootballGP: Eğitim verisi boş")
                return False
                
            # Basit bir GP modeli oluştur
            features_list = []
            results_list = []
            
            for item in gp_data:
                features = item.get('features', {})
                result = item.get('result', 1)
                
                # Features'ı vector'e çevir
                feature_vector = [
                    features.get('home_goals_avg', 1.0),
                    features.get('home_conceded_avg', 1.0),
                    features.get('home_wins_rate', 0.5),
                    features.get('away_goals_avg', 1.0),
                    features.get('away_conceded_avg', 1.0),
                    features.get('away_wins_rate', 0.5),
                    features.get('home_form', 1.5),
                    features.get('away_form', 1.5),
                    features.get('form_diff', 0.0)
                ]
                features_list.append(feature_vector)
                results_list.append(result)
            
            if len(features_list) < 5:
                print("⚠️ FootballGP: Yetersiz eğitim verisi")
                return False
                
            # Label encoding
            unique_results = list(set(results_list))
            self.label_encoder.fit(unique_results)
            encoded_results = self.label_encoder.transform(results_list)
            
            # Feature scaling
            X = np.array(features_list)
            X_scaled = self.scaler.fit_transform(X)
            
            # GP Classifier
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            self.gp_model = GaussianProcessClassifier(
                kernel=kernel,
                random_state=42,
                n_restarts_optimizer=1,
                max_iter_predict=50
            )
            
            self.gp_model.fit(X_scaled, encoded_results)
            self.feature_names = [f'feature_{i}' for i in range(len(feature_vector))]
            
            print(f"✅ FootballGP eğitildi: {len(gp_data)} sample")
            return True
            
        except Exception as e:
            print(f"❌ FootballGP eğitim hatası: {e}")
            return False
    
    def predict(self, feature_dict):
        """
        Basit tahmin metodu - parametric_backtest için
        feature_dict: Dict formatında features
        """
        try:
            if self.gp_model is None:
                return None
                
            # Feature vector'ü oluştur
            feature_vector = [
                feature_dict.get('home_goals_avg', 1.0),
                feature_dict.get('home_conceded_avg', 1.0),
                feature_dict.get('home_wins_rate', 0.5),
                feature_dict.get('away_goals_avg', 1.0),
                feature_dict.get('away_conceded_avg', 1.0),
                feature_dict.get('away_wins_rate', 0.5),
                feature_dict.get('home_form', 1.5),
                feature_dict.get('away_form', 1.5),
                feature_dict.get('form_diff', 0.0)
            ]
            
            X = np.array([feature_vector])
            X_scaled = self.scaler.transform(X)
            
            # Tahmin
            prediction = self.gp_model.predict(X_scaled)[0]
            probabilities = self.gp_model.predict_proba(X_scaled)[0]
            
            # Label decode
            result = self.label_encoder.inverse_transform([prediction])[0]
            
            # Olasılıkları eşleştir
            classes = self.label_encoder.classes_
            prob_dict = {}
            for cls, prob in zip(classes, probabilities):
                if cls == 0:
                    prob_dict['away_win'] = float(prob)
                elif cls == 1:
                    prob_dict['draw'] = float(prob)
                elif cls == 2:
                    prob_dict['home_win'] = float(prob)
            
            return {
                'predicted_result': result,
                'probabilities': prob_dict,
                'confidence': float(max(probabilities))
            }
            
        except Exception as e:
            print(f"❌ FootballGP tahmin hatası: {e}")
            return None
        
    def load_football_data(self, data_file='data/ALM_stat.json'):
        """Futbol verilerini yükle ve hazırla"""
        print(f"🏈 Futbol verileri yükleniyor: {data_file}")
        
        try:
            # Veriyi normalize et
            df = self.normalizer.normalize_dataset(data_file)
            
            if df.empty:
                print("❌ Veri yüklenemedi")
                return None
                
            # Sadece tamamlanmış maçları al
            df = df[df['home_score'].notna() & df['away_score'].notna()].copy()
            
            if len(df) == 0:
                print("❌ Tamamlanmış maç bulunamadı")
                return None
                
            print(f"✅ {len(df)} tamamlanmış maç yüklendi")
            
            # Maç sonucu etiketlerini oluştur (1X2)
            df['match_result'] = df.apply(self._get_match_result, axis=1)
            
            # Feature'ları çıkar
            features = self._extract_features(df)
            
            print(f"📊 {features.shape[1]} özellik çıkarıldı")
            print(f"🎯 Sınıf dağılımı:")
            for result, count in df['match_result'].value_counts().items():
                print(f"   {result}: {count} maç ({count/len(df)*100:.1f}%)")
                
            return df, features
            
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            return None
    
    def _get_match_result(self, row):
        """Maç sonucunu hesapla (1X2)"""
        home_score = row['home_score']
        away_score = row['away_score']
        
        if home_score > away_score:
            return '1'  # Ev sahibi galibiyeti
        elif home_score == away_score:
            return 'X'  # Beraberlik
        else:
            return '2'  # Deplasman galibiyeti
    
    def _extract_features(self, df):
        """Maç verilerinden özellikler çıkar"""
        feature_columns = [
            # Şut istatistikleri
            'shots_home', 'shots_away',
            'shots_on_target_home', 'shots_on_target_away',
            
            # Top sahipliği
            'possession_home', 'possession_away',
            
            # Paslar
            'passes_home', 'passes_away',
            'pass_accuracy_home', 'pass_accuracy_away',
            
            # Savunma
            'corners_home', 'corners_away',
            'fouls_home', 'fouls_away',
            'yellow_cards_home', 'yellow_cards_away',
            'red_cards_home', 'red_cards_away',
            'offsides_home', 'offsides_away',
            
            # Ortalar
            'crosses_successful_home', 'crosses_successful_away',
            'crosses_total_home', 'crosses_total_away'
        ]
        
        # Mevcut kolonları filtrele
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_names = available_features
        
        print(f"🔍 Kullanılabilir özellikler: {len(available_features)}")
        
        # Missing değerleri doldur
        features = df[available_features].fillna(0)
        
        return features.values
    
    def get_optimal_kernels(self):
        """Neural Engine için optimize edilmiş kernel'lar"""
        return {
            'rbf': RBF(length_scale=1.0),
            'matern32': Matern(nu=1.5, length_scale=1.0),
            'matern52': Matern(nu=2.5, length_scale=1.0),
            'combined': RBF(1.0) + WhiteKernel(1e-3),
            'neural_optimized': Matern(nu=2.5, length_scale=0.5) + WhiteKernel(1e-2)
        }
    
    def train_gp_models(self, features, labels, test_size=0.2):
        """Farklı kernel'larla GP modelleri eğit"""
        print("\n🧠 GP Classification modelleri eğitiliyor...")
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Label encoding
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        kernels = self.get_optimal_kernels()
        results = {}
        
        for kernel_name, kernel in kernels.items():
            print(f"\n⚡ {kernel_name.upper()} kernel eğitiliyor...")
            
            start_time = time.time()
            
            # Neural Engine monitoring başlat
            cpu_before = psutil.cpu_percent()
            memory_before = psutil.virtual_memory().percent
            
            try:
                # GP Classifier oluştur
                gp_classifier = GaussianProcessClassifier(
                    kernel=kernel,
                    n_restarts_optimizer=3,  # Neural Engine için optimize
                    max_iter_predict=100,
                    multi_class='one_vs_rest',
                    random_state=42
                )
                
                # Eğitim
                gp_classifier.fit(X_train_scaled, y_train_encoded)
                
                # Test tahminleri
                y_pred = gp_classifier.predict(X_test_scaled)
                y_pred_proba = gp_classifier.predict_proba(X_test_scaled)
                
                training_time = time.time() - start_time
                
                # Neural Engine monitoring bitir
                cpu_after = psutil.cpu_percent()
                memory_after = psutil.virtual_memory().percent
                
                # Performans hesapla
                accuracy = accuracy_score(y_test_encoded, y_pred)
                
                # Cross validation
                cv_scores = cross_val_score(
                    gp_classifier, X_train_scaled, y_train_encoded, 
                    cv=3, scoring='accuracy'
                )
                
                results[kernel_name] = {
                    'model': gp_classifier,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time,
                    'neural_cpu_usage': cpu_after - cpu_before,
                    'neural_memory_usage': memory_after - memory_before,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"✅ Accuracy: {accuracy:.3f}")
                print(f"⏱️ Training time: {training_time:.2f}s")
                print(f"🧠 Neural Engine CPU: +{cpu_after-cpu_before:.1f}%")
                
            except Exception as e:
                print(f"❌ {kernel_name} kernel hatası: {e}")
                continue
        
        # En iyi modeli seç
        if results:
            best_kernel = max(results.keys(), key=lambda k: results[k]['accuracy'])
            self.gp_model = results[best_kernel]['model']
            self.kernel_performance = results
            
            print(f"\n🏆 En iyi kernel: {best_kernel.upper()}")
            print(f"🎯 En iyi accuracy: {results[best_kernel]['accuracy']:.3f}")
            
            # Detaylı classification report
            y_pred_best = results[best_kernel]['predictions']
            y_test_labels = self.label_encoder.inverse_transform(y_test_encoded)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred_best)
            
            print(f"\n📊 Classification Report ({best_kernel}):")
            print(classification_report(y_test_labels, y_pred_labels))
            
            return X_test_scaled, y_test_encoded, results
        
        return None, None, {}
    
    def predict_match_result(self, home_stats, away_stats):
        """Tek maç için sonuç tahmin et"""
        if self.gp_model is None:
            return None
        
        try:
            # Feature vektörü oluştur
            features = []
            
            # Şut istatistikleri
            features.extend([
                home_stats.get('shots', 10),
                away_stats.get('shots', 10),
                home_stats.get('shots_on_target', 5),
                away_stats.get('shots_on_target', 5)
            ])
            
            # Top sahipliği
            features.extend([
                home_stats.get('possession', 50),
                away_stats.get('possession', 50)
            ])
            
            # Diğer istatistikler (mevcut feature'lara göre)
            for feature_name in self.feature_names[6:]:  # İlk 6'sı zaten eklendi
                home_key = feature_name.replace('_home', '').replace('_away', '')
                if '_home' in feature_name:
                    features.append(home_stats.get(home_key, 0))
                else:
                    features.append(away_stats.get(home_key, 0))
            
            # Eksik feature'ları sıfırla doldur
            while len(features) < len(self.feature_names):
                features.append(0)
            
            # Sadece gerekli sayıda feature al
            features = features[:len(self.feature_names)]
            
            # Normalize et
            features_scaled = self.scaler.transform([features])
            
            # Tahmin
            prediction = self.gp_model.predict(features_scaled)[0]
            probabilities = self.gp_model.predict_proba(features_scaled)[0]
            
            # Label decode
            result = self.label_encoder.inverse_transform([prediction])[0]
            
            # Olasılıkları eşleştir
            classes = self.label_encoder.classes_
            prob_dict = dict(zip(classes, probabilities))
            
            return {
                'predicted_result': result,
                'result_meaning': {'1': 'Ev Sahibi Galibiyeti', 'X': 'Beraberlik', '2': 'Deplasman Galibiyeti'}[result],
                'home_win_prob': prob_dict.get('1', 0) * 100,
                'draw_prob': prob_dict.get('X', 0) * 100,
                'away_win_prob': prob_dict.get('2', 0) * 100,
                'confidence': max(probabilities) * 100
            }
            
        except Exception as e:
            print(f"❌ Tahmin hatası: {e}")
            return None
    
    def batch_predict(self, matches_data):
        """Toplu tahmin yap"""
        if self.gp_model is None:
            return []
        
        predictions = []
        
        for match in matches_data:
            home_stats = match.get('home_stats', {})
            away_stats = match.get('away_stats', {})
            
            pred = self.predict_match_result(home_stats, away_stats)
            if pred:
                predictions.append({
                    'match_name': f"{match.get('home', 'Home')} vs {match.get('away', 'Away')}",
                    'prediction': pred
                })
        
        return predictions
    
    def get_model_summary(self):
        """Model özet bilgileri"""
        if not self.kernel_performance:
            return {}
        
        summary = {
            'total_kernels_tested': len(self.kernel_performance),
            'best_kernel': max(self.kernel_performance.keys(), 
                             key=lambda k: self.kernel_performance[k]['accuracy']),
            'neural_engine_usage': self.neural_engine_usage,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_ready': self.gp_model is not None
        }
        
        # Kernel performansları
        summary['kernel_results'] = {}
        for kernel, results in self.kernel_performance.items():
            summary['kernel_results'][kernel] = {
                'accuracy': results['accuracy'],
                'cv_mean': results['cv_mean'],
                'cv_std': results['cv_std'],
                'training_time': results['training_time']
            }
        
        return summary

def main():
    """GP Classification demo"""
    print("🚀 M3 Neural Engine Futbol GP Classification Sistemi")
    print("=" * 60)
    
    # GP Classifier oluştur
    gp_classifier = FootballGPClassifier()
    
    # Alman Ligi verilerini yükle
    data_result = gp_classifier.load_football_data('data/ALM_stat.json')
    
    if data_result is None:
        print("❌ Veri yüklenemedi, çıkılıyor...")
        return
    
    df, features = data_result
    labels = df['match_result'].values
    
    # GP modellerini eğit
    X_test, y_test, results = gp_classifier.train_gp_models(features, labels)
    
    if not results:
        print("❌ Model eğitimi başarısız")
        return
    
    # Model özeti
    summary = gp_classifier.get_model_summary()
    print(f"\n📊 Model Özeti:")
    print(f"🏆 En iyi kernel: {summary['best_kernel']}")
    print(f"🎯 Accuracy: {summary['kernel_results'][summary['best_kernel']]['accuracy']:.3f}")
    print(f"⚡ Özellik sayısı: {summary['feature_count']}")
    
    # Örnek tahmin
    print(f"\n🔮 Örnek Tahmin:")
    example_home = {'shots': 15, 'shots_on_target': 7, 'possession': 60}
    example_away = {'shots': 8, 'shots_on_target': 3, 'possession': 40}
    
    prediction = gp_classifier.predict_match_result(example_home, example_away)
    if prediction:
        print(f"🏠 Ev Sahibi Stats: {example_home}")
        print(f"✈️ Deplasman Stats: {example_away}")
        print(f"🎯 Tahmin: {prediction['predicted_result']} - {prediction['result_meaning']}")
        print(f"📊 Olasılıklar:")
        print(f"   1 (Ev): {prediction['home_win_prob']:.1f}%")
        print(f"   X (Ber): {prediction['draw_prob']:.1f}%")
        print(f"   2 (Dep): {prediction['away_win_prob']:.1f}%")
        print(f"💪 Güven: {prediction['confidence']:.1f}%")
    
    print(f"\n🎉 GP Classification sistemi hazır!")

if __name__ == "__main__":
    main()
