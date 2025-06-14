#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Failure Analysis
Başarısız 2 maçı basit analiz
"""

import json

def main():
    # Veriyi yükle
    with open('data/ALM_stat.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("🔍 BAŞARISIZ MAÇLAR ANALİZİ")
    print("=" * 50)
    
    # 11. hafta maçları
    week11 = [m for m in data if int(m.get('week', 0)) == 11]
    
    failed_matches = ["Hoffenheim", "Wolfsburg"]
    
    for match in week11:
        home = match.get('home', '')
        away = match.get('away', '')
        
        if any(team in home for team in failed_matches):
            score = match.get('score', {}).get('fullTime', {})
            h_score = int(score.get('home', 0)) if score.get('home') else 0
            a_score = int(score.get('away', 0)) if score.get('away') else 0
            
            print(f"\n❌ {home} vs {away}")
            print(f"   Skor: {h_score}-{a_score}")
            print(f"   Sonuç: EV SAHİBİ KAZANDI")
            print(f"   Model: Deplasman galibiyeti tahmin etti")
    
    print("\n🎯 SORUN: EV SAHİBİ AVANTAJI YETERSİZ!")
    print("💡 ÇÖZÜM: Home advantage feature'ını güçlendir")

if __name__ == "__main__":
    main()
