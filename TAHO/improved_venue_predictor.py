#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
İyileştirilmiş Venue Predictor
Ev avantajını artırarak daha doğru tahminler
"""

import json
import numpy as np
from collections import defaultdict

class ImprovedVenuePredictor:
    def __init__(self):
        self.home_stats = defaultdict(dict)
        self.away_stats = defaultdict(dict)
        
    def load_and_analyze(self, league_file, target_week):
        """Data yükle ve analiz et"""
        
        with open(league_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        train_data = [match for match in data if int(match.get('week', 0)) < target_week]
        test_data = [match for match in data if int(match.get('week', 0)) == target_week]
        
        print(f"📊 {league_file} yüklendi: {len(data)} maç")
        print(f"🎯 Eğitim: {len(train_data)} maç, Test: {len(test_data)} maç")
        
        # Calculate venue-specific stats
        self._calculate_venue_stats(train_data)
        
        return train_data, test_data
    
    def _calculate_venue_stats(self, matches):
        """İyileştirilmiş venue-specific istatistikleri hesapla"""
        
        for match in matches:
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            score = match.get('score', {}).get('fullTime', {})
            if not score:
                continue
                
            home_score = int(score.get('home', 0))
            away_score = int(score.get('away', 0))
            
            # Initialize if not exists
            if home_team not in self.home_stats:
                self.home_stats[home_team] = {
                    'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                    'goals_for': 0, 'goals_against': 0, 'points': 0,
                    'recent_form': []
                }
            
            if away_team not in self.away_stats:
                self.away_stats[away_team] = {
                    'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                    'goals_for': 0, 'goals_against': 0, 'points': 0,
                    'recent_form': []
                }
            
            # Home team stats
            self.home_stats[home_team]['matches'] += 1
            self.home_stats[home_team]['goals_for'] += home_score
            self.home_stats[home_team]['goals_against'] += away_score
            
            if home_score > away_score:
                self.home_stats[home_team]['wins'] += 1
                self.home_stats[home_team]['points'] += 3
                self.home_stats[home_team]['recent_form'].append(3)
            elif home_score == away_score:
                self.home_stats[home_team]['draws'] += 1
                self.home_stats[home_team]['points'] += 1
                self.home_stats[home_team]['recent_form'].append(1)
            else:
                self.home_stats[home_team]['losses'] += 1
                self.home_stats[home_team]['recent_form'].append(0)
            
            # Away team stats
            self.away_stats[away_team]['matches'] += 1
            self.away_stats[away_team]['goals_for'] += away_score
            self.away_stats[away_team]['goals_against'] += home_score
            
            if away_score > home_score:
                self.away_stats[away_team]['wins'] += 1
                self.away_stats[away_team]['points'] += 3
                self.away_stats[away_team]['recent_form'].append(3)
            elif away_score == home_score:
                self.away_stats[away_team]['draws'] += 1
                self.away_stats[away_team]['points'] += 1
                self.away_stats[away_team]['recent_form'].append(1)
            else:
                self.away_stats[away_team]['losses'] += 1
                self.away_stats[away_team]['recent_form'].append(0)
        
        # Keep only last 5 form results
        for team in self.home_stats:
            self.home_stats[team]['recent_form'] = self.home_stats[team]['recent_form'][-5:]
        for team in self.away_stats:
            self.away_stats[team]['recent_form'] = self.away_stats[team]['recent_form'][-5:]
    
    def predict_match(self, home_team, away_team, home_advantage=0.6):
        """İyileştirilmiş maç tahmini"""
        
        h_data = self.home_stats.get(home_team, {'matches': 0})
        a_data = self.away_stats.get(away_team, {'matches': 0})
        
        if h_data['matches'] == 0 or a_data['matches'] == 0:
            return "X", 33.0  # Veri yoksa beraberlik
        
        # Temel güç hesabı
        home_strength = h_data['points'] / h_data['matches']
        away_strength = a_data['points'] / a_data['matches']
        
        # Son form bonusu
        home_form_bonus = 0
        if h_data['recent_form']:
            recent_avg = sum(h_data['recent_form']) / len(h_data['recent_form'])
            home_form_bonus = (recent_avg - 1.5) * 0.2  # Form bonusu
        
        away_form_bonus = 0
        if a_data['recent_form']:
            recent_avg = sum(a_data['recent_form']) / len(a_data['recent_form'])
            away_form_bonus = (recent_avg - 1.5) * 0.2
        
        # Motivasyon faktörü - zayıf takımlar bazen sürpriz yapar
        home_motivation = 0
        if home_strength < 1.0:  # Düşük performanslı takım
            home_motivation = 0.3  # Motivasyon bonusu
        
        # Adjusted strengths
        adjusted_home = home_strength + home_advantage + home_form_bonus + home_motivation
        adjusted_away = away_strength + away_form_bonus
        
        strength_diff = adjusted_home - adjusted_away
        
        # İyileştirilmiş karar eşikleri
        if strength_diff > 0.3:  # 0.4'ten 0.3'e düşürdük
            prediction = "1"
            confidence = min(90.0, 50 + abs(strength_diff) * 30)
        elif strength_diff < -0.3:  # 0.4'ten 0.3'e düşürdük
            prediction = "2"
            confidence = min(90.0, 50 + abs(strength_diff) * 30)
        else:
            prediction = "X"
            confidence = 50 + (0.3 - abs(strength_diff)) * 20
        
        return prediction, confidence
    
    def test_predictions(self, test_data, home_advantage=0.6):
        """Test maçlarında tahmin yap"""
        
        correct = 0
        total = len(test_data)
        
        print(f"\n🎯 İYİLEŞTİRİLMİŞ VENUE-SPECIFIC MODEL (Ev Avantajı: {home_advantage})")
        print(f"{'='*70}")
        
        for i, match in enumerate(test_data, 1):
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            score = match.get('score', {}).get('fullTime', {})
            if not score:
                continue
                
            home_score = int(score.get('home', 0))
            away_score = int(score.get('away', 0))
            
            if home_score > away_score:
                actual = "1"
            elif home_score == away_score:
                actual = "X"
            else:
                actual = "2"
            
            prediction, confidence = self.predict_match(home_team, away_team, home_advantage)
            
            is_correct = prediction == actual
            if is_correct:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{i}. {status} {home_team} vs {away_team}")
            print(f"   Tahmin: {prediction} | Gerçek: {actual} | Güven: {confidence:.1f}%")
            print(f"   Skor: {home_score}-{away_score}")
            
            if not is_correct:
                # Başarısızlık analizi
                h_data = self.home_stats.get(home_team, {'matches': 0})
                a_data = self.away_stats.get(away_team, {'matches': 0})
                
                if h_data['matches'] > 0 and a_data['matches'] > 0:
                    home_strength = h_data['points'] / h_data['matches']
                    away_strength = a_data['points'] / a_data['matches']
                    strength_diff = (home_strength + home_advantage) - away_strength
                    
                    print(f"   🔍 Ev gücü: {home_strength:.2f}, Deplasman gücü: {away_strength:.2f}")
                    print(f"   ⚖️ Güç farkı: {strength_diff:.2f}")
            
            print()
        
        accuracy = (correct / total) * 100
        print(f"💡 SONUÇ: {correct}/{total} doğru tahmin")
        print(f"📊 Başarı oranı: {accuracy:.1f}%")
        
        return accuracy

def main():
    predictor = ImprovedVenuePredictor()
    
    # Test different home advantage values
    for home_adv in [0.4, 0.6, 0.8, 1.0]:
        print(f"\n{'='*80}")
        print(f"🏠 EV AVANTAJI: {home_adv}")
        print(f"{'='*80}")
        
        train_data, test_data = predictor.load_and_analyze('data/ALM_stat.json', target_week=11)
        accuracy = predictor.test_predictions(test_data, home_advantage=home_adv)
        
        print(f"\n🎯 Ev avantajı {home_adv} ile başarı oranı: {accuracy:.1f}%")

if __name__ == "__main__":
    main()
