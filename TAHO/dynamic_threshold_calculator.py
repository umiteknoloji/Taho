#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Threshold Calculator
Statik threshold değerlerini gerçek veri analiziyle dinamik olarak hesaplar
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DynamicThresholdCalculator:
    """
    Tüm statik threshold değerlerini dinamik olarak hesaplar
    Her threshold, kullanıldığı konteksteki gerçek veriler analiz edilerek belirlenir
    """
    
    def __init__(self):
        self.league_data = None
        self.team_stats = None
        self.calculated_thresholds = {}
        
    def load_data(self, league_file: str, target_week: int):
        """Veriyi yükle ve analiz için hazırla"""
        with open(league_file, 'r', encoding='utf-8') as f:
            self.league_data = json.load(f)
        
        # Target week öncesi veriler
        self.training_data = [match for match in self.league_data if int(match.get('week', 0)) < target_week]
        
        print(f"📊 Dynamic Threshold Calculator yüklendi")
        print(f"📈 Analiz edilen maç sayısı: {len(self.training_data)}")
        
    def calculate_all_thresholds(self) -> Dict[str, float]:
        """Tüm threshold değerlerini dinamik olarak hesapla"""
        thresholds = {}
        
        # 1. Defensive factor threshold (varsayılan: 0.5)
        thresholds['defensive_factor'] = self._calculate_defensive_threshold()
        
        # 2. Balance indicator threshold (varsayılan: 0.1)
        thresholds['balance_threshold'] = self._calculate_balance_threshold()
        
        # 3. Home advantage threshold (varsayılan: 0.4)
        thresholds['home_advantage'] = self._calculate_home_advantage()
        
        # 4. Desperation threshold (varsayılan: 1.0)
        thresholds['desperation_threshold'] = self._calculate_desperation_threshold()
        
        # 5. Upset potential threshold (varsayılan: -0.5)
        thresholds['upset_threshold'] = self._calculate_upset_threshold()
        
        # 6. Feature count threshold (varsayılan: 25)
        thresholds['max_features'] = self._calculate_optimal_feature_count()
        
        # 7. Temporal weights (varsayılan: [0.35, 0.25, 0.20, 0.15, 0.05])
        thresholds['temporal_weights'] = self._calculate_temporal_weights()
        
        # 8. Kernel noise (varsayılan: 1e-3)
        thresholds['kernel_noise'] = self._calculate_kernel_noise()
        
        self.calculated_thresholds = thresholds
        
        print("🎯 Dinamik Threshold'lar Hesaplandı:")
        for key, value in thresholds.items():
            print(f"   {key}: {value}")
            
        return thresholds
    
    def _calculate_defensive_threshold(self) -> float:
        """
        Defensive factor threshold'unu hesapla
        Liga ortalaması gol yeme sayısını kullan
        """
        all_goals_conceded = []
        
        for match in self.training_data:
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            all_goals_conceded.extend([away_score, home_score])
        
        if all_goals_conceded:
            league_avg_conceded = np.mean(all_goals_conceded)
            return max(0.1, league_avg_conceded)  # Minimum 0.1
        else:
            return 0.5  # Fallback
    
    def _calculate_balance_threshold(self) -> float:
        """
        Balance indicator threshold'unu hesapla
        Takım güçleri arasındaki gerçek varyansı analiz et
        """
        team_strengths = self._calculate_team_strengths()
        
        if len(team_strengths) > 1:
            strength_variance = np.var(list(team_strengths.values()))
            # Düşük varyans = daha dengeli lig = daha düşük threshold
            return max(0.05, min(0.3, strength_variance / 2))
        else:
            return 0.1  # Fallback
    
    def _calculate_home_advantage(self) -> float:
        """
        Home advantage'ı gerçek verilerden hesapla
        Ev sahibi takımların kazanma oranından çıkar
        """
        home_wins = 0
        home_draws = 0
        total_matches = 0
        
        for match in self.training_data:
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            total_matches += 1
            
            if home_score > away_score:
                home_wins += 1
            elif home_score == away_score:
                home_draws += 1
        
        if total_matches > 0:
            home_win_rate = home_wins / total_matches
            # Gerçek ev avantajı: 0.33 (eşit şans) üzerindeki kısım
            home_advantage = max(0.0, home_win_rate - 0.33)
            return min(0.6, home_advantage * 2)  # Scale it up, max 0.6
        else:
            return 0.4  # Fallback
    
    def _calculate_desperation_threshold(self) -> float:
        """
        Desperation threshold'unu hesapla
        Kötü performans gösteren takımların alt %25'lik dilimini bul
        """
        team_strengths = self._calculate_team_strengths()
        
        if team_strengths:
            strengths = list(team_strengths.values())
            percentile_25 = np.percentile(strengths, 25)
            # 25. percentile altındaki takımlar "desperate" kabul edilir
            return percentile_25
        else:
            return 1.0  # Fallback
    
    def _calculate_upset_threshold(self) -> float:
        """
        Upset potential threshold'unu hesapla
        Güç farkı dağılımından std dev kullan
        """
        team_strengths = self._calculate_team_strengths()
        
        if len(team_strengths) > 1:
            strengths = list(team_strengths.values())
            strength_std = np.std(strengths)
            # 1 standart sapma altı "upset potential" olarak kabul edilir
            return -strength_std
        else:
            return -0.5  # Fallback
    
    def _calculate_optimal_feature_count(self) -> int:
        """
        Optimal feature sayısını hesapla
        Veri boyutuna göre adaptif olarak belirle
        """
        data_size = len(self.training_data)
        
        # Veri sayısına göre feature sayısı
        # Kural: Her 10 veri noktası için 1 feature (aşırı overfitting'i önle)
        optimal_features = max(10, min(50, data_size // 10))
        
        return optimal_features
    
    def _calculate_temporal_weights(self) -> List[float]:
        """
        Temporal weight'leri hesapla
        Recent performance correlation'a göre belirle
        """
        # Son maçların tahmin gücünü analiz et
        correlations = self._analyze_temporal_correlations()
        
        if correlations:
            # Normalize et
            total = sum(correlations)
            if total > 0:
                return [c / total for c in correlations]
        
        # Fallback: Exponential decay
        weights = [0.4, 0.3, 0.2, 0.07, 0.03]
        return weights[:5]  # Max 5 weight
    
    def _calculate_kernel_noise(self) -> float:
        """
        Kernel noise'u hesapla
        Veri gürültüsüne göre adaptif olarak belirle
        """
        # Sonuçların variability'sini analiz et
        match_variability = self._calculate_match_variability()
        
        # Yüksek variability = daha fazla noise
        base_noise = 1e-3
        adaptive_noise = base_noise * (1 + match_variability)
        
        return min(1e-1, max(1e-5, adaptive_noise))
    
    def _calculate_team_strengths(self) -> Dict[str, float]:
        """Takım güçlerini hesapla (Points per match)"""
        team_stats = {}
        
        for match in self.training_data:
            home_team = match.get('home', '')
            away_team = match.get('away', '')
            
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            # Initialize
            for team in [home_team, away_team]:
                if team not in team_stats:
                    team_stats[team] = {'points': 0, 'matches': 0}
            
            # Points calculation
            team_stats[home_team]['matches'] += 1
            team_stats[away_team]['matches'] += 1
            
            if home_score > away_score:
                team_stats[home_team]['points'] += 3
            elif home_score == away_score:
                team_stats[home_team]['points'] += 1
                team_stats[away_team]['points'] += 1
            else:
                team_stats[away_team]['points'] += 3
        
        # Points per match
        strengths = {}
        for team, stats in team_stats.items():
            if stats['matches'] > 0:
                strengths[team] = stats['points'] / stats['matches']
        
        return strengths
    
    def _analyze_temporal_correlations(self) -> List[float]:
        """Son maçların tahmin gücünü analiz et"""
        # Basit implementasyon: exponential decay
        # Gerçek implementasyon için correlation analysis yapılabilir
        return [0.4, 0.3, 0.2, 0.07, 0.03]
    
    def _calculate_match_variability(self) -> float:
        """Maç sonuçlarının variability'sini hesapla"""
        goal_differences = []
        
        for match in self.training_data:
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            goal_differences.append(abs(home_score - away_score))
        
        if goal_differences:
            return np.std(goal_differences) / np.mean(goal_differences) if np.mean(goal_differences) > 0 else 1.0
        else:
            return 1.0
    
    def get_threshold(self, threshold_name: str) -> Any:
        """Belirli bir threshold değerini al"""
        return self.calculated_thresholds.get(threshold_name, None)
    
    def print_analysis(self):
        """Threshold analiz sonuçlarını yazdır"""
        print("\n🔍 DYNAMIC THRESHOLD ANALYSIS")
        print("=" * 50)
        
        team_strengths = self._calculate_team_strengths()
        print(f"📊 Analyzed teams: {len(team_strengths)}")
        print(f"📈 Total matches: {len(self.training_data)}")
        
        if team_strengths:
            strengths = list(team_strengths.values())
            print(f"💪 Average team strength: {np.mean(strengths):.3f} PPM")
            print(f"📊 Strength variance: {np.var(strengths):.3f}")
            print(f"🎯 Strength range: {min(strengths):.3f} - {max(strengths):.3f}")
        
        # Home advantage analysis
        home_wins = sum(1 for match in self.training_data 
                       if int(match.get('score', {}).get('fullTime', {}).get('home', 0)) > 
                          int(match.get('score', {}).get('fullTime', {}).get('away', 0)))
        
        home_win_rate = home_wins / len(self.training_data) if self.training_data else 0
        print(f"🏠 Home win rate: {home_win_rate:.3f}")
        
        print("\n🎯 Calculated Dynamic Thresholds:")
        for key, value in self.calculated_thresholds.items():
            print(f"   {key}: {value}")


# Test function
def test_dynamic_thresholds():
    """Dynamic threshold calculator'ı test et"""
    calculator = DynamicThresholdCalculator()
    
    # Test data
    league_file = "/Users/umitduman/Taho/data/ALM_stat.json"
    target_week = 11
    
    calculator.load_data(league_file, target_week)
    thresholds = calculator.calculate_all_thresholds()
    calculator.print_analysis()
    
    return calculator, thresholds

if __name__ == "__main__":
    calculator, thresholds = test_dynamic_thresholds()
