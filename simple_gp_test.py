#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basit GP Tahmin Testi
Gerçek futbol verisi ile GP performansını test edelim
"""

import json
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def parse_match_data(data):
    """Maç verisini parse et"""
    matches = []
    
    for match in data:
        try:
            # Skorları çıkar
            score_data = match.get('score', {})
            full_time = score_data.get('fullTime', {})
            home_score = int(full_time.get('home', 0))
            away_score = int(full_time.get('away', 0))
            
            # Sonuç hesapla
            if home_score > away_score:
                result = '1'
            elif home_score == away_score:
                result = 'X'
            else:
                result = '2'
            
            # İstatistikleri parse et
            stats = match.get('stats', {})
            
            parsed_match = {
                'home_team': match.get('home', ''),
                'away_team': match.get('away', ''),
                'home_score': home_score,
                'away_score': away_score,
                'total_goals': home_score + away_score,
                'result': result,
                'week': int(match.get('week', 0))
            }
            
            # Basit istatistikleri ekle
            for stat_name, stat_data in stats.items():
                if isinstance(stat_data, dict):
                    home_val = parse_stat_value(stat_data.get('home', '0'))
                    away_val = parse_stat_value(stat_data.get('away', '0'))
                    
                    stat_key = stat_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                    parsed_match[f'home_{stat_key}'] = home_val
                    parsed_match[f'away_{stat_key}'] = away_val
            
            matches.append(parsed_match)
            
        except Exception as e:
            print(f"Parse error: {e}")
            continue
    
    return matches

def parse_stat_value(value):
    """İstatistik değerini sayıya çevir"""
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        value = value.replace('%', '')
        
        if '/' in value:
            parts = value.split('/')
            if len(parts) == 2 and parts[1] != '0':
                return float(parts[0]) / float(parts[1]) * 100
            return float(parts[0]) if parts[0].isdigit() else 0
        
        try:
            return float(value)
        except:
            return 0
    
    return 0

def create_simple_features(df):
    """Basit özellikler oluştur"""
    print("🔧 Basit özellik mühendisliği...")
    
    # Sort by week
    df = df.sort_values('week').reset_index(drop=True)
    
    # Son 3 maçın ortalaması
    for team_col in ['home_team', 'away_team']:
        score_col = 'home_score' if team_col == 'home_team' else 'away_score'
        
        df[f'{team_col}_avg_goals_3'] = df.groupby(team_col)[score_col].rolling(3).mean().reset_index(drop=True)
        df[f'{team_col}_avg_conceded_3'] = df.groupby(team_col)['total_goals'].rolling(3).mean().reset_index(drop=True)
    
    # Form (points from last 3 games)
    df['home_form'] = 0
    df['away_form'] = 0
    
    for idx, row in df.iterrows():
        # Home team form
        home_matches = df[(df['home_team'] == row['home_team']) & (df.index < idx)].tail(3)
        home_points = sum(3 if r == '1' else (1 if r == 'X' else 0) for r in home_matches['result'])
        df.at[idx, 'home_form'] = home_points
        
        # Away team form
        away_matches = df[(df['away_team'] == row['away_team']) & (df.index < idx)].tail(3)
        away_points = sum(3 if r == '2' else (1 if r == 'X' else 0) for r in away_matches['result'])
        df.at[idx, 'away_form'] = away_points
    
    # Feature differences
    df['form_difference'] = df['home_form'] - df['away_form']
    df['avg_goals_difference'] = df.get('home_team_avg_goals_3', 0) - df.get('away_team_avg_goals_3', 0)
    
    return df

def test_gp_performance():
    """GP performansını test et"""
    print("🚀 GP PERFORMANS TESTİ")
    print("="*40)
    
    try:
        # Veriyi yükle
        with open('data/TR_stat.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 TR Ligi: {len(data)} maç yüklendi")
        
        # Parse et
        matches = parse_match_data(data)
        df = pd.DataFrame(matches)
        
        print(f"🎯 Parse edildi: {len(df)} maç")
        print(f"📈 Sonuç dağılımı: {df['result'].value_counts().to_dict()}")
        
        # Özellik mühendisliği
        enhanced_df = create_simple_features(df)
        
        # Feature selection
        feature_columns = [col for col in enhanced_df.columns 
                          if col not in ['home_team', 'away_team', 'result'] and 
                          not col.startswith('home_') and not col.startswith('away_')]
        
        # Manuel özellik seçimi
        selected_features = [
            'total_goals', 'week', 'home_form', 'away_form', 
            'form_difference'
        ]
        
        # Sayısal özellikler ekle
        numeric_features = [col for col in enhanced_df.columns 
                           if enhanced_df[col].dtype in ['int64', 'float64'] and 
                           col not in ['home_score', 'away_score', 'result']]
        
        # İstatistik özelliklerini ekle
        stat_features = [col for col in enhanced_df.columns if 'home_' in col or 'away_' in col]
        selected_features.extend(stat_features[:10])  # İlk 10 istatistik
        
        # Available features'ları kullan
        available_features = [f for f in selected_features if f in enhanced_df.columns]
        
        print(f"🔧 Kullanılan özellikler ({len(available_features)}):")
        for f in available_features:
            print(f"  • {f}")
        
        if not available_features:
            print("❌ Hiç özellik bulunamadı!")
            return
        
        # X ve y hazırla
        X = enhanced_df[available_features].fillna(0)
        y = enhanced_df['result']
        
        # Veri kontrolü
        print(f"\\n📊 Veri boyutu: {X.shape}")
        print(f"🎯 Hedef dağılım: {y.value_counts().to_dict()}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"\\n🔄 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Veriyi normalize et
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Label encode
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # GP Kernelleri test et
        kernels = {
            'RBF': ConstantKernel(1.0) * RBF(length_scale=1.0),
            'Matern_2.5': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
            'RBF_Noise': ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3),
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        print(f"\\n🎯 KERNEL TEST SONUÇLARI:")
        
        for name, kernel in kernels.items():
            try:
                gp = GaussianProcessClassifier(
                    kernel=kernel,
                    random_state=42,
                    n_restarts_optimizer=1,
                    max_iter_predict=100
                )
                
                # Cross validation
                cv_scores = cross_val_score(gp, X_train_scaled, y_train_encoded, cv=3, scoring='accuracy')
                avg_score = cv_scores.mean()
                
                print(f"  {name:15}: %{avg_score*100:.1f} ± {cv_scores.std()*100:.1f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = gp
                    best_name = name
                    
            except Exception as e:
                print(f"  {name:15}: Hata - {str(e)[:30]}")
        
        if best_model is None:
            print("❌ Hiçbir model çalışmadı!")
            return
        
        # En iyi modeli eğit
        print(f"\\n🏆 En iyi model: {best_name} (%{best_score*100:.1f})")
        best_model.fit(X_train_scaled, y_train_encoded)
        
        # Test predictions
        y_pred_encoded = best_model.predict(X_test_scaled)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Probabilities
        y_proba = best_model.predict_proba(X_test_scaled)
        
        # Test accuracy
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"\\n📈 TEST SONUÇLARI:")
        print(f"  Doğruluk: %{test_accuracy*100:.1f}")
        
        # Confidence analysis
        confidences = []
        for prob in y_proba:
            max_prob = np.max(prob)
            entropy = -np.sum(prob * np.log(prob + 1e-8))
            normalized_entropy = entropy / np.log(len(prob))
            confidence = max_prob * (1 - normalized_entropy * 0.5)
            confidences.append(confidence * 100)
        
        # High confidence predictions
        high_conf_threshold = 70
        high_conf_indices = [i for i, conf in enumerate(confidences) if conf >= high_conf_threshold]
        
        if high_conf_indices:
            high_conf_pred = [y_pred[i] for i in high_conf_indices]
            high_conf_actual = [y_test.iloc[i] for i in high_conf_indices]
            high_conf_acc = accuracy_score(high_conf_actual, high_conf_pred)
            
            print(f"  Yüksek Güvenli (%{high_conf_threshold}+): %{high_conf_acc*100:.1f}")
            print(f"  Yüksek Güvenli Oran: {len(high_conf_indices)}/{len(y_pred)} (%{len(high_conf_indices)/len(y_pred)*100:.1f})")
        
        print(f"  Ortalama Güven: %{np.mean(confidences):.1f}")
        
        # Detailed report
        print(f"\\n📋 DETAYLI RAPOR:")
        print(classification_report(y_test, y_pred))
        
        # Sonuç dağılımı
        print(f"\\n🎯 SONUÇ DAĞILIMI (Test):")
        test_result_counts = pd.Series(y_test).value_counts()
        pred_result_counts = pd.Series(y_pred).value_counts()
        
        for result in ['1', 'X', '2']:
            actual_count = test_result_counts.get(result, 0)
            pred_count = pred_result_counts.get(result, 0)
            print(f"  {result}: Gerçek {actual_count}, Tahmin {pred_count}")
        
        return {
            'test_accuracy': test_accuracy,
            'cv_score': best_score,
            'high_conf_accuracy': high_conf_acc if high_conf_indices else 0,
            'high_conf_ratio': len(high_conf_indices)/len(y_pred) if high_conf_indices else 0,
            'avg_confidence': np.mean(confidences)
        }
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_gp_performance()
    
    if result:
        print(f"\\n✅ TEST TAMAMLANDI!")
        print(f"📊 Özet:")
        print(f"  • CV Doğruluk: %{result['cv_score']*100:.1f}")
        print(f"  • Test Doğruluk: %{result['test_accuracy']*100:.1f}")
        print(f"  • Yüksek Güvenli Doğruluk: %{result['high_conf_accuracy']*100:.1f}")
        print(f"  • Ortalama Güven: %{result['avg_confidence']:.1f}")
        
        if result['test_accuracy'] < 0.4:
            print("\\n🔴 DURUM: Kötü - Sistem iyileştirilmeli")
        elif result['test_accuracy'] < 0.5:
            print("\\n🟡 DURUM: Zayıf - Özellikler geliştirilmeli")
        elif result['test_accuracy'] < 0.55:
            print("\\n🟢 DURUM: Orta - Kabul edilebilir")
        else:
            print("\\n🟢 DURUM: İyi - Başarılı model")
