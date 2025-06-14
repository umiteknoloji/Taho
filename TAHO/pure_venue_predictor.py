#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure Venue-Specific Prediction Model
Takımları SADECE venue-specific maçlarına göre değerlendiren model
"""

import json
import numpy as np
from collections import defaultdict

class PureVenuePredictor:
    def __init__(self):
        self.home_stats = defaultdict(dict)  # team -> home stats
        self.away_stats = defaultdict(dict)  # team -> away stats
        
    def load_and_analyze(self, league_file, target_week):
        """Data yükle ve venue-specific stats hesapla"""
        
        with open(league_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        train_data = [match for match in data if int(match.get('week', 0)) < target_week]
        test_data = [match for match in data if int(match.get('week', 0)) == target_week]
        
        print(f"📊 {league_file} yüklendi: {len(data)} maç")
        print(f"🎯 Eğitim: {len(train_data)} maç, Test: {len(test_data)} maç")
        
        # Calculate pure venue-specific stats
        self._calculate_venue_stats(train_data)
        
        return train_data, test_data
    
    def _calculate_venue_stats(self, matches):
        """Pure venue-specific istatistikleri hesapla"""
        
        for match in matches:
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            # Initialize if not exists
            if home_team not in self.home_stats:
                self.home_stats[home_team] = {
                    'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                    'goals_for': 0, 'goals_against': 0, 'points': 0
                }
            
            if away_team not in self.away_stats:
                self.away_stats[away_team] = {
                    'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                    'goals_for': 0, 'goals_against': 0, 'points': 0
                }
            
            # Update HOME team's HOME stats
            self.home_stats[home_team]['matches'] += 1
            self.home_stats[home_team]['goals_for'] += home_score
            self.home_stats[home_team]['goals_against'] += away_score
            
            # Update AWAY team's AWAY stats
            self.away_stats[away_team]['matches'] += 1
            self.away_stats[away_team]['goals_for'] += away_score
            self.away_stats[away_team]['goals_against'] += home_score
            
            # Result
            if home_score > away_score:
                # Home win
                self.home_stats[home_team]['wins'] += 1
                self.home_stats[home_team]['points'] += 3
                self.away_stats[away_team]['losses'] += 1
            elif home_score == away_score:
                # Draw
                self.home_stats[home_team]['draws'] += 1
                self.home_stats[home_team]['points'] += 1
                self.away_stats[away_team]['draws'] += 1
                self.away_stats[away_team]['points'] += 1
            else:
                # Away win
                self.away_stats[away_team]['wins'] += 1
                self.away_stats[away_team]['points'] += 3
                self.home_stats[home_team]['losses'] += 1
        
        print(f"📈 Venue-specific stats hesaplandı")
        
    def predict_match(self, home_team, away_team):
        """Tek maç tahmini - PURE VENUE-SPECIFIC"""
        
        # Get HOME team's HOME performance
        home_stats = self.home_stats.get(home_team, {})
        home_matches = home_stats.get('matches', 0)
        
        if home_matches == 0:
            home_strength = 1.5  # Default
        else:
            home_points_per_match = home_stats.get('points', 0) / home_matches
            home_strength = home_points_per_match
        
        # Get AWAY team's AWAY performance  
        away_stats = self.away_stats.get(away_team, {})
        away_matches = away_stats.get('matches', 0)
        
        if away_matches == 0:
            away_strength = 1.0  # Default (away teams generally weaker)
        else:
            away_points_per_match = away_stats.get('points', 0) / away_matches
            away_strength = away_points_per_match
        
        # Simple prediction logic
        strength_diff = home_strength - away_strength
        
        # Add home advantage (typically 0.3-0.5 points)
        home_advantage = 0.4
        adjusted_diff = strength_diff + home_advantage
        
        # Prediction thresholds
        if adjusted_diff > 0.8:
            prediction = '1'  # Home win
            confidence = min(0.9, 0.6 + adjusted_diff * 0.2)
        elif adjusted_diff < -0.5:
            prediction = '2'  # Away win  
            confidence = min(0.9, 0.6 + abs(adjusted_diff) * 0.2)
        else:
            prediction = 'X'  # Draw
            confidence = 0.6 - abs(adjusted_diff) * 0.3
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'home_strength': home_strength,
            'away_strength': away_strength,
            'strength_diff': strength_diff,
            'adjusted_diff': adjusted_diff
        }
    
    def test_predictions(self, test_data):
        """Test data'sında tahmin yap"""
        
        results = []
        
        for match in test_data:
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            # Prediction
            pred_result = self.predict_match(home_team, away_team)
            
            # Actual result
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0))
            away_score = int(score.get('away', 0))
            
            if home_score > away_score:
                actual = '1'
            elif home_score == away_score:
                actual = 'X'
            else:
                actual = '2'
            
            is_correct = pred_result['prediction'] == actual
            
            result = {
                'match': f"{home_team} vs {away_team}",
                'home_team': home_team,
                'away_team': away_team,
                'predicted': pred_result['prediction'],
                'actual': actual,
                'is_correct': is_correct,
                'confidence': pred_result['confidence'],
                'home_score': home_score,
                'away_score': away_score,
                'home_strength': pred_result['home_strength'],
                'away_strength': pred_result['away_strength'],
                'strength_diff': pred_result['strength_diff'],
                'adjusted_diff': pred_result['adjusted_diff']
            }
            
            results.append(result)
        
        return results
    
    def analyze_results(self, results):
        """Sonuç analizi"""
        
        correct = sum(1 for r in results if r['is_correct'])
        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"\n🎯 PURE VENUE-SPECIFIC MODEL SONUÇLARI:")
        print(f"Doğru tahmin: {correct}/{total}")
        print(f"Başarı oranı: {accuracy:.1f}%")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            status = "✅" if result['is_correct'] else "❌"
            print(f"{i}. {status} {result['match']}")
            print(f"   Tahmin: {result['predicted']} | Gerçek: {result['actual']} | Güven: {result['confidence']:.1%}")
            print(f"   Skor: {result['home_score']}-{result['away_score']}")
            print(f"   Ev Gücü: {result['home_strength']:.2f} | Deplasman Gücü: {result['away_strength']:.2f}")
            print(f"   Güç Farkı: {result['strength_diff']:.2f} | Ev Avantajlı: {result['adjusted_diff']:.2f}")
            
            if not result['is_correct']:
                # Failure analysis
                print(f"   🔍 BAŞARISIZLIK ANALİZİ:")
                if result['predicted'] == '1' and result['actual'] != '1':
                    print(f"      Ev sahibi beklenen kadar güçlü değildi")
                elif result['predicted'] == '2' and result['actual'] != '2':
                    print(f"      Deplasman takımı beklenen kadar güçlü değildi")
                elif result['predicted'] == 'X' and result['actual'] != 'X':
                    print(f"      Beraberlik beklentisi yanlıştı")
            print()
        
        return accuracy

def test_pure_venue_system():
    """Pure venue-specific system test"""
    
    predictor = PureVenuePredictor()
    train_data, test_data = predictor.load_and_analyze('data/ALM_stat.json', 11)
    
    results = predictor.test_predictions(test_data)
    accuracy = predictor.analyze_results(results)
    
    print(f"\n💡 PURE VENUE-SPECIFIC BAŞARI: {accuracy:.1f}%")
    
    if accuracy == 100:
        print("🏆 MÜKEMMEL! Pure venue-specific ile %100 başarı!")
    else:
        print("📈 Venue-specific değerlendirmede hala eksikler var...")

if __name__ == "__main__":
    test_pure_venue_system()
