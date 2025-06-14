#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Venue-Specific Performance Analyzer
Başarısız tahminlerin venue-specific performance analizini yapar
"""

import json
import pandas as pd
from no_h2h_gp import NoH2HGP

def analyze_failed_predictions():
    """Week 11'deki başarısız tahminleri venue-specific performance açısından analiz et"""
    
    # Load data and create model
    predictor = NoH2HGP()
    train_data, test_data = predictor.load_and_analyze_data('data/ALM_stat.json', 11)
    
    # Train model
    predictor.train_model(train_data)
    
    # Get predictions
    results = predictor.predict_matches(test_data)
    
    # Focus on failed predictions
    failed_matches = [r for r in results if not r['is_correct']]
    
    print("🔍 BAŞARISIZ TAHMİNLERİN VENUE-SPECIFIC ANALİZİ")
    print("=" * 60)
    
    for match in failed_matches:
        home_team = match['home_team']
        away_team = match['away_team']
        
        print(f"\n❌ {home_team} vs {away_team}")
        print(f"Tahmin: {match['predicted']} | Gerçek: {match['actual']}")
        print(f"Skor: {match['home_score']}-{match['away_score']}")
        
        # Get team stats
        home_stats = predictor.team_stats.get(home_team, {})
        away_stats = predictor.team_stats.get(away_team, {})
        
        # HOME TEAM ANALYSIS
        print(f"\n🏠 {home_team} (Ev Sahibi):")
        home_matches = home_stats.get('home_matches', 0)
        home_wins = home_stats.get('home_wins', 0)
        home_draws = home_stats.get('home_draws', 0)
        home_losses = home_stats.get('home_losses', 0)
        home_goals_for = home_stats.get('home_goals_for', 0)
        home_goals_against = home_stats.get('home_goals_against', 0)
        
        if home_matches > 0:
            home_win_rate = (home_wins / home_matches) * 100
            home_points_per_match = (home_wins * 3 + home_draws) / home_matches
            home_goals_per_match = home_goals_for / home_matches
            home_conceded_per_match = home_goals_against / home_matches
            
            print(f"  Evde oynanan maç: {home_matches}")
            print(f"  Evde G-B-M: {home_wins}-{home_draws}-{home_losses}")
            print(f"  Evde kazanma oranı: {home_win_rate:.1f}%")
            print(f"  Evde maç başı puan: {home_points_per_match:.2f}")
            print(f"  Evde maç başı gol: {home_goals_per_match:.2f}")
            print(f"  Evde maç başı yediği: {home_conceded_per_match:.2f}")
            print(f"  Evde form: {home_stats.get('weighted_home_form', 0):.3f}")
        else:
            print("  Evde maç verisi yok!")
        
        # AWAY TEAM ANALYSIS
        print(f"\n✈️ {away_team} (Deplasman):")
        away_matches = away_stats.get('away_matches', 0)
        away_wins = away_stats.get('away_wins', 0)
        away_draws = away_stats.get('away_draws', 0)
        away_losses = away_stats.get('away_losses', 0)
        away_goals_for = away_stats.get('away_goals_for', 0)
        away_goals_against = away_stats.get('away_goals_against', 0)
        
        if away_matches > 0:
            away_win_rate = (away_wins / away_matches) * 100
            away_points_per_match = (away_wins * 3 + away_draws) / away_matches
            away_goals_per_match = away_goals_for / away_matches
            away_conceded_per_match = away_goals_against / away_matches
            
            print(f"  Deplasmanda oynanan maç: {away_matches}")
            print(f"  Deplasmanda G-B-M: {away_wins}-{away_draws}-{away_losses}")
            print(f"  Deplasmanda kazanma oranı: {away_win_rate:.1f}%")
            print(f"  Deplasmanda maç başı puan: {away_points_per_match:.2f}")
            print(f"  Deplasmanda maç başı gol: {away_goals_per_match:.2f}")
            print(f"  Deplasmanda maç başı yediği: {away_conceded_per_match:.2f}")
            print(f"  Deplasman form: {away_stats.get('weighted_away_form', 0):.3f}")
        else:
            print("  Deplasman maç verisi yok!")
        
        # FEATURE ANALYSIS
        print(f"\n📊 Feature Değerleri:")
        features = predictor.create_pure_form_features({
            'home': home_team,
            'away': away_team
        })
        
        key_features = [
            'venue_strength_diff',
            'venue_form_diff', 
            'venue_win_rate_diff',
            'venue_draw_tendency',
            'master_venue_indicator'
        ]
        
        for feature in key_features:
            if feature in features:
                print(f"  {feature}: {features[feature]:.3f}")
        
        # PREDICTION EXPLANATION
        print(f"\n💭 Tahmin Açıklaması:")
        venue_strength_diff = features.get('venue_strength_diff', 0)
        
        if venue_strength_diff > 0.5:
            print(f"  Model ev sahibini güçlü gördü (+{venue_strength_diff:.2f})")
        elif venue_strength_diff < -0.5:
            print(f"  Model deplasman takımını güçlü gördü ({venue_strength_diff:.2f})")
        else:
            print(f"  Model dengeli bir maç öngördü ({venue_strength_diff:.2f})")
        
        print("-" * 60)

def compare_venue_vs_total_performance():
    """Venue-specific vs total performance karşılaştırması"""
    
    predictor = NoH2HGP()
    train_data, test_data = predictor.load_and_analyze_data('data/ALM_stat.json', 11)
    
    print("\n📈 VENUE-SPECIFIC vs TOTAL PERFORMANCE KARŞILAŞTIRMASI")
    print("=" * 70)
    
    # Calculate stats for each team
    teams_analysis = []
    
    for team, stats in predictor.team_stats.items():
        total_matches = stats.get('total_matches', 0)
        home_matches = stats.get('home_matches', 0)
        away_matches = stats.get('away_matches', 0)
        
        if total_matches >= 5:  # En az 5 maç oynamış takımlar
            # Total performance
            total_win_rate = stats.get('total_wins', 0) / total_matches * 100
            total_points_per_match = (stats.get('total_wins', 0) * 3 + stats.get('total_draws', 0)) / total_matches
            
            # Home performance
            home_win_rate = stats.get('home_wins', 0) / max(home_matches, 1) * 100
            home_points_per_match = (stats.get('home_wins', 0) * 3 + stats.get('home_draws', 0)) / max(home_matches, 1)
            
            # Away performance
            away_win_rate = stats.get('away_wins', 0) / max(away_matches, 1) * 100
            away_points_per_match = (stats.get('away_wins', 0) * 3 + stats.get('away_draws', 0)) / max(away_matches, 1)
            
            # Calculate differences
            home_advantage = home_points_per_match - total_points_per_match
            away_disadvantage = away_points_per_match - total_points_per_match
            venue_effect = home_points_per_match - away_points_per_match
            
            teams_analysis.append({
                'team': team,
                'total_ppm': total_points_per_match,
                'home_ppm': home_points_per_match,
                'away_ppm': away_points_per_match,
                'venue_effect': venue_effect,
                'home_advantage': home_advantage,
                'away_disadvantage': away_disadvantage
            })
    
    # Sort by venue effect
    teams_analysis.sort(key=lambda x: x['venue_effect'], reverse=True)
    
    print(f"{'Takım':<20} {'Total':<6} {'Home':<6} {'Away':<6} {'Venue':<7} {'H.Adv':<6} {'A.Dis':<6}")
    print("-" * 70)
    
    for team_data in teams_analysis:
        print(f"{team_data['team']:<20} "
              f"{team_data['total_ppm']:<6.2f} "
              f"{team_data['home_ppm']:<6.2f} "
              f"{team_data['away_ppm']:<6.2f} "
              f"{team_data['venue_effect']:<7.2f} "
              f"{team_data['home_advantage']:<6.2f} "
              f"{team_data['away_disadvantage']:<6.2f}")
    
    # Calculate average venue effect
    avg_venue_effect = sum(t['venue_effect'] for t in teams_analysis) / len(teams_analysis)
    print(f"\n📊 Ortalama venue effect: {avg_venue_effect:.2f} puan")
    
    # Find extreme cases
    max_venue_effect = max(teams_analysis, key=lambda x: x['venue_effect'])
    min_venue_effect = min(teams_analysis, key=lambda x: x['venue_effect'])
    
    print(f"🏠 En güçlü ev avantajı: {max_venue_effect['team']} (+{max_venue_effect['venue_effect']:.2f})")
    print(f"✈️ En zayıf ev avantajı: {min_venue_effect['team']} ({min_venue_effect['venue_effect']:.2f})")

if __name__ == "__main__":
    analyze_failed_predictions()
    compare_venue_vs_total_performance()
