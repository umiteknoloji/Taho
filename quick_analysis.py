#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GP Tahmin Doğruluğu Analizi
"""

import json
import pandas as pd
import numpy as np
from collections import Counter

def analyze_data():
    print("🚀 FUTBOL VERİSİ ANALİZİ")
    print("="*40)
    
    try:
        # TR ligini yükle
        with open('data/TR_stat.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 TR Ligi: {len(data)} maç")
        
        # Sonuçları parse et
        results = []
        goals = []
        
        for match in data:
            try:
                score_data = match.get('score', {})
                full_time = score_data.get('fullTime', {})
                home_score = int(full_time.get('home', 0))
                away_score = int(full_time.get('away', 0))
                
                if home_score > away_score:
                    results.append('1')
                elif home_score == away_score:
                    results.append('X')
                else:
                    results.append('2')
                
                goals.append(home_score + away_score)
                
            except:
                continue
        
        print(f"✅ Parse edildi: {len(results)} maç")
        
        # Sonuç dağılımı
        result_dist = Counter(results)
        
        print(f"\n🎯 SONUÇ DAĞILIMI:")
        for result, count in result_dist.items():
            print(f"  {result}: {count} ({count/len(results)*100:.1f}%)")
        
        # Gol istatistikleri
        print(f"\n⚽ GOL İSTATİSTİKLERİ:")
        print(f"  Ortalama gol: {np.mean(goals):.1f}")
        print(f"  Medyan gol: {np.median(goals):.1f}")
        print(f"  Min-Max gol: {min(goals)}-{max(goals)}")
        
        # Naive baseline
        most_common_result = result_dist.most_common(1)[0][0]
        naive_accuracy = result_dist.most_common(1)[0][1] / len(results)
        
        print(f"\n🎲 NAIVE BASELINE:")
        print(f"  En çok çıkan: {most_common_result}")
        print(f"  Naive doğruluk: %{naive_accuracy*100:.1f}")
        
        # GP için öneriler
        print(f"\n💡 GP İÇİN ÖNERİLER:")
        
        if naive_accuracy > 0.5:
            print("  🔴 UYARI: Veri dengesiz!")
            print("  📈 Çözüm: Class balancing uygula")
            print("  🎯 Hedef: Balanced accuracy kullan")
        else:
            print("  🟢 İyi: Veri dengeli")
            print("  📈 Strateji: Feature engineering odaklı")
            print("  🎯 Hedef: Standard accuracy")
        
        # Tahmin hedefleri
        print(f"\n🎯 TAHMİN HEDEFLERI:")
        min_target = naive_accuracy + 0.05
        good_target = naive_accuracy + 0.10
        excellent_target = naive_accuracy + 0.15
        
        print(f"  Minimum: %{min_target*100:.1f}")
        print(f"  İyi: %{good_target*100:.1f}")  
        print(f"  Mükemmel: %{excellent_target*100:.1f}")
        
        # GP iyileştirme stratejileri
        print(f"\n🚀 GP İYİLEŞTİRME STRATEJİLERİ:")
        print("  1. 📊 VERİ İYİLEŞTİRMESİ:")
        print("     • Daha fazla istatistik ekle")
        print("     • Son 5 maç formu")
        print("     • Head-to-head geçmiş")
        print("     • Ev sahibi avantajı")
        
        print("  2. 🔧 GP OPTİMİZASYONU:")
        print("     • Kernel selection (RBF vs Matern)")
        print("     • Hyperparameter tuning")
        print("     • Feature scaling")
        print("     • Confidence thresholding")
        
        print("  3. 🎯 SONUÇ STRATEJİSİ:")
        print("     • Sadece yüksek güvenli tahminler")
        print("     • Draw tahminlerini filtrele")
        print("     • Liga özel modeller")
        print("     • Ensemble yaklaşımı")
        
        return {
            'total_matches': len(results),
            'result_distribution': dict(result_dist),
            'naive_accuracy': naive_accuracy,
            'avg_goals': np.mean(goals),
            'targets': {
                'minimum': min_target,
                'good': good_target, 
                'excellent': excellent_target
            }
        }
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return None

if __name__ == "__main__":
    result = analyze_data()
    
    if result:
        print(f"\n✅ ANALİZ TAMAMLANDI!")
        print(f"📊 {result['total_matches']} maç analiz edildi")
        print(f"🎯 Naive baseline: %{result['naive_accuracy']*100:.1f}")
        print(f"⚽ Ortalama gol: {result['avg_goals']:.1f}")
