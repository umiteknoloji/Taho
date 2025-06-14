#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derinlemesine Başarısızlık Analizi
Neden venue-specific model %100 başarılı olamıyor?
"""

import json
import numpy as np
from collections import defaultdict

class DeepFailureAnalyzer:
    def __init__(self):
        self.data = []
        self.home_stats = defaultdict(dict)
        self.away_stats = defaultdict(dict)
        
    def load_data(self, league_file):
        """Data yükle"""
        with open(league_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"📊 {len(self.data)} maç yüklendi")
        
    def calculate_venue_stats(self, matches):
        """Venue-specific istatistikleri hesapla"""
        
        # Reset stats
        self.home_stats = defaultdict(lambda: {
            'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0, 'points': 0,
            'recent_matches': []
        })
        
        self.away_stats = defaultdict(lambda: {
            'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0, 'points': 0,
            'recent_matches': []
        })
        
        for match in matches:
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            week = int(match.get('week', 0))
            
            # Home team stats
            self.home_stats[home_team]['matches'] += 1
            self.home_stats[home_team]['goals_for'] += home_score
            self.home_stats[home_team]['goals_against'] += away_score
            
            if home_score > away_score:
                self.home_stats[home_team]['wins'] += 1
                self.home_stats[home_team]['points'] += 3
                result = 'W'
            elif home_score == away_score:
                self.home_stats[home_team]['draws'] += 1
                self.home_stats[home_team]['points'] += 1
                result = 'D'
            else:
                self.home_stats[home_team]['losses'] += 1
                result = 'L'
            
            self.home_stats[home_team]['recent_matches'].append({
                'week': week, 'result': result, 'score': f"{home_score}-{away_score}",
                'opponent': away_team
            })
            
            # Away team stats
            self.away_stats[away_team]['matches'] += 1
            self.away_stats[away_team]['goals_for'] += away_score
            self.away_stats[away_team]['goals_against'] += home_score
            
            if away_score > home_score:
                self.away_stats[away_team]['wins'] += 1
                self.away_stats[away_team]['points'] += 3
                result = 'W'
            elif away_score == home_score:
                self.away_stats[away_team]['draws'] += 1
                self.away_stats[away_team]['points'] += 1
                result = 'D'
            else:
                self.away_stats[away_team]['losses'] += 1
                result = 'L'
            
            self.away_stats[away_team]['recent_matches'].append({
                'week': week, 'result': result, 'score': f"{away_score}-{home_score}",
                'opponent': home_team
            })
        
        # Sort recent matches by week (latest 5)
        for team in self.home_stats:
            self.home_stats[team]['recent_matches'] = sorted(
                self.home_stats[team]['recent_matches'], 
                key=lambda x: x['week']
            )[-5:]
            
        for team in self.away_stats:
            self.away_stats[team]['recent_matches'] = sorted(
                self.away_stats[team]['recent_matches'], 
                key=lambda x: x['week']
            )[-5:]
    
    def analyze_failed_predictions(self, target_week=11):
        """Başarısız tahminleri detaylandır"""
        
        train_data = [match for match in self.data if int(match.get('week', 0)) < target_week]
        test_data = [match for match in self.data if int(match.get('week', 0)) == target_week]
        
        print(f"\n🎯 Hafta {target_week} Analizi")
        print(f"📈 Eğitim: {len(train_data)} maç")
        print(f"🔍 Test: {len(test_data)} maç")
        
        # Calculate stats from training data
        self.calculate_venue_stats(train_data)
        
        print(f"\n{'='*80}")
        print(f"🔥 BAŞARISIZ TAHMİNLERİN DERİN ANALİZİ")
        print(f"{'='*80}")
        
        failed_matches = [
            ("Hoffenheim", "RB Leipzig", "4-3", "Expected: Draw, Actual: Home Win"),
            ("Wolfsburg", "Union Berlin", "1-0", "Expected: Draw, Actual: Home Win")
        ]
        
        for home_team, away_team, actual_score, error_desc in failed_matches:
            print(f"\n🚨 BAŞARISIZ TAHMİN: {home_team} vs {away_team}")
            print(f"📊 Gerçek Skor: {actual_score}")
            print(f"❌ Hata: {error_desc}")
            
            self._analyze_single_match(home_team, away_team, target_week)
            
    def _analyze_single_match(self, home_team, away_team, week):
        """Tek maç detaylı analiz"""
        
        print(f"\n🏠 {home_team} (EVİNDE):")
        home_data = self.home_stats[home_team]
        if home_data['matches'] > 0:
            home_win_rate = (home_data['wins'] / home_data['matches']) * 100
            home_avg_goals = home_data['goals_for'] / home_data['matches']
            home_avg_conceded = home_data['goals_against'] / home_data['matches']
            
            print(f"   📈 Ev maçları: {home_data['matches']}")
            print(f"   🏆 Galibiyet oranı: {home_win_rate:.1f}%")
            print(f"   ⚽ Ortalama attığı gol: {home_avg_goals:.2f}")
            print(f"   🥅 Ortalama yediği gol: {home_avg_conceded:.2f}")
            print(f"   📊 Puan ortalaması: {home_data['points'] / home_data['matches']:.2f}")
            
            print(f"   📅 Son 5 ev maçı:")
            for match in home_data['recent_matches']:
                print(f"      Hafta {match['week']}: vs {match['opponent']} {match['score']} ({match['result']})")
        else:
            print(f"   ❌ {home_team} için ev maçı verisi yok!")
            
        print(f"\n✈️ {away_team} (DEPLASMASINDA):")
        away_data = self.away_stats[away_team]
        if away_data['matches'] > 0:
            away_win_rate = (away_data['wins'] / away_data['matches']) * 100
            away_avg_goals = away_data['goals_for'] / away_data['matches']
            away_avg_conceded = away_data['goals_against'] / away_data['matches']
            
            print(f"   📈 Deplasman maçları: {away_data['matches']}")
            print(f"   🏆 Galibiyet oranı: {away_win_rate:.1f}%")
            print(f"   ⚽ Ortalama attığı gol: {away_avg_goals:.2f}")
            print(f"   🥅 Ortalama yediği gol: {away_avg_conceded:.2f}")
            print(f"   📊 Puan ortalaması: {away_data['points'] / away_data['matches']:.2f}")
            
            print(f"   📅 Son 5 deplasman maçı:")
            for match in away_data['recent_matches']:
                print(f"      Hafta {match['week']}: vs {match['opponent']} {match['score']} ({match['result']})")
        else:
            print(f"   ❌ {away_team} için deplasman maçı verisi yok!")
            
        # Calculate prediction logic
        print(f"\n🤖 TAHMİN LOJİĞİ:")
        if home_data['matches'] > 0 and away_data['matches'] > 0:
            home_strength = home_data['points'] / home_data['matches']
            away_strength = away_data['points'] / away_data['matches']
            home_advantage = 0.4  # Ev avantajı
            
            adjusted_home = home_strength + home_advantage
            strength_diff = adjusted_home - away_strength
            
            print(f"   🏠 Ev gücü: {home_strength:.2f} + {home_advantage} (ev avantajı) = {adjusted_home:.2f}")
            print(f"   ✈️ Deplasman gücü: {away_strength:.2f}")
            print(f"   ⚖️ Güç farkı: {strength_diff:.2f}")
            
            if strength_diff > 0.4:
                prediction = "1 (Ev galibiyeti)"
            elif strength_diff < -0.4:
                prediction = "2 (Deplasman galibiyeti)"
            else:
                prediction = "X (Beraberlik)"
                
            print(f"   🎯 Tahmin: {prediction}")
            
            # Analyze why prediction failed
            print(f"\n🔍 BAŞARISIZLIK SEBEPLERİ:")
            if prediction == "X (Beraberlik)":
                print(f"   • Model beraberlik öngördü çünkü güç farkı çok az ({strength_diff:.2f})")
                print(f"   • Ancak ev avantajı yeterince güçlü değerlendirilememiş olabilir")
                
                if home_data['matches'] < 5:
                    print(f"   • {home_team} için az ev maçı verisi ({home_data['matches']} maç)")
                    
                if away_data['matches'] < 5:
                    print(f"   • {away_team} için az deplasman maçı verisi ({away_data['matches']} maç)")
                    
        print(f"\n{'-'*60}")

def main():
    analyzer = DeepFailureAnalyzer()
    analyzer.load_data('data/ALM_stat.json')
    analyzer.analyze_failed_predictions(target_week=11)

if __name__ == "__main__":
    main()
