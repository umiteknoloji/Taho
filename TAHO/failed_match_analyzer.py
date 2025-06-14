#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Failed Match Deep Analyzer
Başarısız tahmin edilen 2 maçı derinlemesine analiz eder
"""

import json
import numpy as np
from collections import defaultdict

class FailedMatchAnalyzer:
    def __init__(self):
        self.data = []
        self.failed_matches = [
            "Hoffenheim vs RB Leipzig",
            "Wolfsburg vs Union Berlin"
        ]
    
    def load_data(self, file_path):
        """Veriyi yükle"""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"📊 {len(self.data)} maç yüklendi")
    
    def analyze_failed_teams(self):
        """Başarısız tahmin edilen takımları analiz et"""
        print("🔍 BAŞARISIZ MAÇLARIN DETAYLI ANALİZİ")
        print("=" * 60)
        
        failed_teams = ["Hoffenheim", "RB Leipzig", "Wolfsburg", "Union Berlin"]
        
        for team in failed_teams:
            print(f"\n📈 {team} DETAYLI ANALİZ:")
            self.analyze_team_details(team)
    
    def analyze_team_details(self, team_name):
        """Takım detaylarını analiz et"""
        team_matches = []
        
        # Takımın 11. haftaya kadar olan maçları
        for match in self.data:
            week = int(match.get('week', 0))
            if week >= 11:
                continue
                
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            if team_name in home_team or team_name in away_team:
                team_matches.append(match)
        
        if not team_matches:
            print(f"   ❌ {team_name} için maç bulunamadı")
            return
        
        print(f"   📊 Toplam maç sayısı: {len(team_matches)}")
        
        # Home/Away performance
        home_matches = [m for m in team_matches if team_name in m.get('home', '')]
        away_matches = [m for m in team_matches if team_name in m.get('away', '')]
        
        print(f"   🏠 Ev sahibi maçları: {len(home_matches)}")
        print(f"   ✈️ Deplasman maçları: {len(away_matches)}")
        
        # Son 5 maç analizi
        recent_matches = sorted(team_matches, key=lambda x: int(x.get('week', 0)))[-5:]
        print(f"   🔥 Son 5 maç:")
        
        for match in recent_matches:
            week = match.get('week')
            home = match.get('home', '')
            away = match.get('away', '')
            score = match.get('score', {}).get('fullTime', {})
            home_score = score.get('home', 0)
            away_score = score.get('away', 0)
            
            is_home = team_name in home
            result = self.get_result(home_score, away_score, is_home)
            
            print(f"      Hafta {week}: {home} {home_score}-{away_score} {away} → {result}")
        
        # Ev sahibi performansı
        if home_matches:
            home_wins = 0
            home_draws = 0
            home_losses = 0
            home_goals_for = 0
            home_goals_against = 0
            
            for match in home_matches:
                score = match.get('score', {}).get('fullTime', {})
                home_score = int(score.get('home', 0))
                away_score = int(score.get('away', 0))
                
                home_goals_for += home_score
                home_goals_against += away_score
                
                if home_score > away_score:
                    home_wins += 1
                elif home_score == away_score:
                    home_draws += 1
                else:
                    home_losses += 1
            
            print(f"   🏠 Ev sahibi performansı:")
            print(f"      Galibiyet: {home_wins}, Beraberlik: {home_draws}, Mağlubiyet: {home_losses}")
            print(f"      Gol ortalaması: {home_goals_for/len(home_matches):.1f}")
            print(f"      Yediği gol: {home_goals_against/len(home_matches):.1f}")
        
        # Deplasman performansı
        if away_matches:
            away_wins = 0
            away_draws = 0
            away_losses = 0
            away_goals_for = 0
            away_goals_against = 0
            
            for match in away_matches:
                score = match.get('score', {}).get('fullTime', {})
                home_score = int(score.get('home', 0))
                away_score = int(score.get('away', 0))
                
                away_goals_for += away_score
                away_goals_against += home_score
                
                if away_score > home_score:
                    away_wins += 1
                elif away_score == home_score:
                    away_draws += 1
                else:
                    away_losses += 1
            
            print(f"   ✈️ Deplasman performansı:")
            print(f"      Galibiyet: {away_wins}, Beraberlik: {away_draws}, Mağlubiyet: {away_losses}")
            print(f"      Gol ortalaması: {away_goals_for/len(away_matches):.1f}")
            print(f"      Yediği gol: {away_goals_against/len(away_matches):.1f}")
    
    def get_result(self, home_score, away_score, is_home):
        """Maç sonucunu al"""
        if home_score > away_score:
            return "GALİBİYET" if is_home else "MAĞLUBIYET"
        elif home_score == away_score:
            return "BERABERLİK"
        else:
            return "MAĞLUBIYET" if is_home else "GALİBİYET"
    
    def analyze_week11_matches(self):
        """11. hafta maçlarını analiz et"""
        print(f"\n🎯 11. HAFTA BAŞARISIZ MAÇLAR ANALİZİ")
        print("=" * 60)
        
        week11_matches = [m for m in self.data if int(m.get('week', 0)) == 11]
        
        for match in week11_matches:
            home = match.get('home', '')
            away = match.get('away', '')
            match_name = f"{home} vs {away}"
            
            if match_name in self.failed_matches:
                print(f"\n❌ {match_name}")
                score = match.get('score', {}).get('fullTime', {})
                home_score = score.get('home', 0)
                away_score = score.get('away', 0)
                print(f"   📊 Skor: {home_score}-{away_score}")
                
                # Her takımın o ana kadarki formu
                self.analyze_match_context(home, away, 11)
    
    def analyze_match_context(self, home_team, away_team, target_week):
        """Maç kontekstini analiz et"""
        print(f"   🔍 Maç Konteksti:")
        
        # Her iki takımın da o ana kadarki son 3 maçı
        for team in [home_team, away_team]:
            recent_matches = []
            for match in self.data:
                week = int(match.get('week', 0))
                if week >= target_week:
                    continue
                    
                if team in match.get('home', '') or team in match.get('away', ''):
                    recent_matches.append((week, match))
            
            # Son 3 maç
            recent_matches = sorted(recent_matches, key=lambda x: x[0])[-3:]
            
            print(f"      {team} son 3 maç:")
            total_points = 0
            for week, match in recent_matches:
                home = match.get('home', '')
                away = match.get('away', '')
                score = match.get('score', {}).get('fullTime', {})
                home_score = int(score.get('home', 0))
                away_score = int(score.get('away', 0))
                
                is_home = team in home
                
                if home_score > away_score:
                    points = 3 if is_home else 0
                elif home_score == away_score:
                    points = 1
                else:
                    points = 0 if is_home else 3
                
                total_points += points
                result_str = "GAL" if points == 3 else "BER" if points == 1 else "MAĞ"
                venue = "Ev" if is_home else "Dep"
                
                print(f"         Hafta {week}: {home} {home_score}-{away_score} {away} → {result_str} ({venue})")
            
            print(f"      → Son 3 maçta {total_points} puan (Ort: {total_points/3:.1f})")
    
    def find_patterns(self):
        """Başarısızlık pattern'lerini bul"""
        print(f"\n🔍 BAŞARISIZLIK PATTERN ANALİZİ")
        print("=" * 60)
        
        print("📊 Gözlemlenen Patternler:")
        print("1. Her iki başarısız maçta da ev sahibi kazandı")
        print("2. Modelimiz deplasman galibiyeti tahmin etti")
        print("3. Her iki maçta da ev sahibi takım daha düşük sıralama")
        print()
        
        print("🎯 Potansiyel Sebepler:")
        print("• Ev sahibi avantajı yeterince güçlü modellenmiyor")
        print("• Düşük sıralı takımların motivasyon faktörü eksik")
        print("• Takım sıralamasına aşırı güvenme")
        print("• Ev sahibi takımın özel durumları (sakatlık, transfer vb.) eksik")
        print()
        
        print("💡 Çözüm Önerileri:")
        print("• Ev sahibi avantajı ağırlığını artır")
        print("• Takım motivasyon faktörlerini ekle")
        print("• Sıralama bazlı tahminleri yumuşat")
        print("• Sezon içi form değişikliklerini daha iyi yakala")

def main():
    analyzer = FailedMatchAnalyzer()
    analyzer.load_data('data/ALM_stat.json')
    analyzer.analyze_failed_teams()
    analyzer.analyze_week11_matches()
    analyzer.find_patterns()

if __name__ == "__main__":
    main()
