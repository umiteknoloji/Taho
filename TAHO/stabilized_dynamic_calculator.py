#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stabilized Dynamic Threshold Calculator
Performans tutarsızlığını önlemek için optimize edilmiş threshold hesaplamaları
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class StabilizedDynamicThresholdCalculator:
    """
    Optimize edilmiş, daha istikrarlı dynamic threshold calculator
    Performans tutarsızlığını minimize eder
    """
    
    def __init__(self):
        self.league_data = None
        self.team_stats = None
        self.calculated_thresholds = {}
        self.training_data = None
        
    def load_data(self, league_file: str, target_week: int):
        """Veriyi yükle ve analiz için hazırla"""
        with open(league_file, 'r', encoding='utf-8') as f:
            self.league_data = json.load(f)
        
        # Target week öncesi veriler
        self.training_data = [match for match in self.league_data if int(match.get('week', 0)) < target_week]
        
        print(f"📊 Stabilized Dynamic Threshold Calculator yüklendi")
        print(f"📈 Analiz edilen maç sayısı: {len(self.training_data)}")
        
    def calculate_all_thresholds(self) -> Dict[str, float]:
        """Tüm threshold değerlerini STABILIZED şekilde hesapla"""
        thresholds = {}
        
        # 1. STABILIZED Defensive factor (median + outlier filtering)
        thresholds['defensive_factor'] = self._calculate_stabilized_defensive_threshold()
        
        # 2. STABILIZED Balance indicator (robust variance)
        thresholds['balance_threshold'] = self._calculate_stabilized_balance_threshold()
        
        # 3. STABILIZED Home advantage (weighted + confidence interval)
        thresholds['home_advantage'] = self._calculate_stabilized_home_advantage()
        
        # 4. STABILIZED Desperation threshold (percentile smoothing)
        thresholds['desperation_threshold'] = self._calculate_stabilized_desperation_threshold()
        
        # 5. STABILIZED Upset potential (robust standard deviation)
        thresholds['upset_threshold'] = self._calculate_stabilized_upset_threshold()
        
        # 6. CONSERVATIVE Feature count (overfitting prevention)
        thresholds['max_features'] = self._calculate_conservative_feature_count()
        
        # 7. SMOOTH Temporal weights (gradual decay)
        thresholds['temporal_weights'] = self._calculate_smooth_temporal_weights()
        
        # 8. ADAPTIVE Kernel noise (data quality based)
        thresholds['kernel_noise'] = self._calculate_adaptive_kernel_noise()
        
        self.calculated_thresholds = thresholds
        
        print("🎯 Stabilized Dynamic Threshold'lar Hesaplandı:")
        for key, value in thresholds.items():
            if isinstance(value, list):
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value:.4f}")
            
        return thresholds
    
    def _calculate_stabilized_defensive_threshold(self) -> float:
        """
        STABILIZED defensive factor - median + IQR filtering kullan
        Outlier'lara karşı daha robust
        """
        all_goals_conceded = []
        
        for match in self.training_data:
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            all_goals_conceded.extend([away_score, home_score])
        
        if all_goals_conceded:
            # Median kullan mean yerine (outlier'lara daha az duyarlı)
            median_conceded = np.median(all_goals_conceded)
            
            # IQR ile outlier'ları filtrele
            q75, q25 = np.percentile(all_goals_conceded, [75, 25])
            iqr = q75 - q25
            
            # Daha konservatif threshold
            stabilized_factor = max(0.5, min(2.5, median_conceded + 0.25 * iqr))
            return stabilized_factor
        else:
            return 1.0  # Safe fallback
    
    def _calculate_stabilized_balance_threshold(self) -> float:
        """
        STABILIZED balance threshold - robust variance estimation
        """
        team_strengths = self._calculate_team_strengths()
        
        if len(team_strengths) > 3:  # Minimum 4 takım gerekli
            strengths = list(team_strengths.values())
            
            # Robust variance (MAD - Median Absolute Deviation)
            median_strength = np.median(strengths)
            mad = np.median([abs(s - median_strength) for s in strengths])
            
            # Convert MAD to robust variance estimate
            robust_variance = (mad * 1.4826) ** 2  # 1.4826 is scaling factor
            
            # More conservative balance threshold
            return max(0.05, min(0.25, robust_variance / 3))
        else:
            return 0.1  # Safe fallback
    
    def _calculate_stabilized_home_advantage(self) -> float:
        """
        STABILIZED home advantage - confidence interval based
        Küçük sample size'larda daha konservatif
        """
        home_wins = draws = total_matches = 0
        
        for match in self.training_data:
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            total_matches += 1
            
            if home_score > away_score:
                home_wins += 1
            elif home_score == away_score:
                draws += 1
        
        if total_matches >= 20:  # Minimum sample size için check
            home_win_rate = home_wins / total_matches
            
            # Confidence interval hesabı (Wilson score interval)
            n = total_matches
            p = home_win_rate
            z = 1.96  # 95% confidence
            
            # Wilson score interval lower bound (more conservative)
            wilson_lower = (p + z*z/(2*n) - z * np.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)
            
            # Home advantage = actual rate - expected (0.33) with confidence adjustment
            base_advantage = max(0.0, wilson_lower - 0.33)
            
            # Scale but cap at reasonable levels
            stabilized_advantage = max(0.05, min(0.35, base_advantage * 1.5))
            
            return stabilized_advantage
        else:
            # Small sample size - use conservative default
            return 0.15
    
    def _calculate_stabilized_desperation_threshold(self) -> float:
        """
        STABILIZED desperation threshold - smoothed percentiles
        """
        team_strengths = self._calculate_team_strengths()
        
        if team_strengths:
            strengths = list(team_strengths.values())
            
            # Use 30th percentile instead of 25th (less extreme)
            percentile_30 = np.percentile(strengths, 30)
            
            # Smooth with overall average
            overall_avg = np.mean(strengths)
            smoothed_threshold = 0.7 * percentile_30 + 0.3 * overall_avg
            
            return max(0.8, min(1.5, smoothed_threshold))
        else:
            return 1.0
    
    def _calculate_stabilized_upset_threshold(self) -> float:
        """
        STABILIZED upset threshold - robust standard deviation
        """
        team_strengths = self._calculate_team_strengths()
        
        if len(team_strengths) > 2:
            strengths = list(team_strengths.values())
            
            # Robust standard deviation (using MAD)
            median_strength = np.median(strengths)
            mad = np.median([abs(s - median_strength) for s in strengths])
            robust_std = mad * 1.4826
            
            # More conservative upset threshold
            return max(-1.0, min(-0.3, -0.8 * robust_std))
        else:
            return -0.5
    
    def _calculate_conservative_feature_count(self) -> int:
        """
        CONSERVATIVE feature count - overfitting prevention
        """
        data_size = len(self.training_data)
        
        # Much more conservative feature selection
        if data_size < 40:
            return 5  # Very few features for small datasets
        elif data_size < 60:
            return 7
        elif data_size < 80:
            return 9
        elif data_size < 100:
            return 11
        elif data_size < 120:
            return 13
        else:
            return min(15, data_size // 8)  # Max 15 features, 1 per 8 samples
    
    def _calculate_smooth_temporal_weights(self) -> List[float]:
        """
        SMOOTH temporal weights - gradual decay, less aggressive
        """
        # More gradual decay, less emphasis on most recent
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        
        # Normalize to sum to 1
        total = sum(weights)
        return [w / total for w in weights]
    
    def _calculate_adaptive_kernel_noise(self) -> float:
        """
        ADAPTIVE kernel noise - based on data quality and size
        """
        data_size = len(self.training_data)
        
        # Calculate match outcome variability
        match_variability = self._calculate_match_variability()
        
        # Base noise adjusted for data size and quality
        if data_size < 50:
            base_noise = 5e-3  # Higher noise for small datasets
        elif data_size < 100:
            base_noise = 3e-3
        else:
            base_noise = 2e-3
        
        # Adjust for match variability
        adaptive_noise = base_noise * (1 + 0.5 * match_variability)
        
        return min(1e-2, max(1e-4, adaptive_noise))
    
    def _calculate_team_strengths(self) -> Dict[str, float]:
        """Takım güçlerini hesapla (Points per match) - Robust version"""
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
        
        # Points per match with minimum match requirement
        strengths = {}
        for team, stats in team_stats.items():
            if stats['matches'] >= 3:  # Minimum 3 maç gerekli
                strengths[team] = stats['points'] / stats['matches']
        
        return strengths
    
    def _calculate_match_variability(self) -> float:
        """Maç sonuçlarının variability'sini hesapla - Robust version"""
        goal_differences = []
        
        for match in self.training_data:
            score = match.get('score', {}).get('fullTime', {})
            home_score = int(score.get('home', 0)) if score.get('home') else 0
            away_score = int(score.get('away', 0)) if score.get('away') else 0
            
            goal_differences.append(abs(home_score - away_score))
        
        if goal_differences:
            # Use coefficient of variation (robust measure)
            mean_diff = np.mean(goal_differences)
            std_diff = np.std(goal_differences)
            
            if mean_diff > 0:
                cv = std_diff / mean_diff
                return min(2.0, cv)  # Cap at 2.0
            else:
                return 1.0
        else:
            return 1.0
    
    def get_threshold(self, threshold_name: str) -> Any:
        """Belirli bir threshold değerini al"""
        return self.calculated_thresholds.get(threshold_name, None)
    
    def print_stability_analysis(self):
        """Stability analiz sonuçlarını yazdır"""
        print("\n🔍 STABILITY ANALYSIS")
        print("=" * 50)
        
        team_strengths = self._calculate_team_strengths()
        print(f"📊 Valid teams: {len(team_strengths)}")
        print(f"📈 Total matches: {len(self.training_data)}")
        
        if team_strengths:
            strengths = list(team_strengths.values())
            median_strength = np.median(strengths)
            mad = np.median([abs(s - median_strength) for s in strengths])
            
            print(f"💪 Median team strength: {median_strength:.3f} PPM")
            print(f"📊 Robust variability (MAD): {mad:.3f}")
            print(f"🎯 Strength range: {min(strengths):.3f} - {max(strengths):.3f}")
        
        # Data quality metrics
        match_var = self._calculate_match_variability()
        print(f"⚡ Match variability: {match_var:.3f}")
        
        data_quality = "HIGH" if len(self.training_data) >= 80 and match_var < 1.5 else \
                      "MEDIUM" if len(self.training_data) >= 50 and match_var < 2.0 else "LOW"
        print(f"📋 Data quality: {data_quality}")

# Test function
def test_stabilized_thresholds():
    """Stabilized threshold calculator'ı test et"""
    calculator = StabilizedDynamicThresholdCalculator()
    
    # Test data
    league_file = "/Users/umitduman/Taho/data/ALM_stat.json"
    
    print("🎯 STABILIZED DYNAMIC THRESHOLDS TEST")
    print("=" * 60)
    
    results = {}
    
    for week in [10, 11, 12]:
        print(f"\n📅 Week {week}:")
        
        calculator.load_data(league_file, week)
        thresholds = calculator.calculate_all_thresholds()
        calculator.print_stability_analysis()
        
        results[week] = thresholds.copy()
    
    # Compare stability across weeks
    print(f"\n📊 STABILITY COMPARISON ACROSS WEEKS:")
    print("-" * 60)
    
    threshold_keys = set()
    for week_thresholds in results.values():
        threshold_keys.update(week_thresholds.keys())
    
    for key in sorted(threshold_keys):
        if key != 'temporal_weights':  # Skip list type
            values = []
            for week in [10, 11, 12]:
                if key in results[week]:
                    values.append(results[week][key])
            
            if values and len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val > 0 else 0
                
                stability = "STABLE" if cv < 0.1 else "MODERATE" if cv < 0.2 else "UNSTABLE"
                print(f"   {key:20s}: {mean_val:8.4f} ±{std_val:.4f} [{stability}]")
    
    return results

if __name__ == "__main__":
    results = test_stabilized_thresholds()
