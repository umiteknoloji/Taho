#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-League GP Analysis System
Farklı liglerde performans karşılaştırması
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_gp_system import AdvancedGPPredictor
import warnings
warnings.filterwarnings('ignore')

class MultiLeagueGPAnalyzer:
    def __init__(self):
        self.league_files = {
            'TR': {'file': 'data/TR_stat.json', 'name': '🇹🇷 Türkiye Süper Lig'},
            'ENG': {'file': 'data/ENG_stat.json', 'name': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League'},
            'ESP': {'file': 'data/ESP_stat.json', 'name': '🇪🇸 La Liga'},
            'ALM': {'file': 'data/ALM_stat.json', 'name': '🇩🇪 Bundesliga'},
            'FRA': {'file': 'data/FRA_stat.json', 'name': '🇫🇷 Ligue 1'},
            'HOL': {'file': 'data/HOL_stat.json', 'name': '🇳🇱 Eredivisie'},
            'POR': {'file': 'data/POR_stat.json', 'name': '🇵🇹 Primeira Liga'},
            'BEL': {'file': 'data/BEL_stat.json', 'name': '🇧🇪 Pro League'}
        }
        self.league_models = {}
        self.league_performance = {}
        
    def train_all_leagues(self):
        """Tüm ligleri eğit"""
        print("🌍 Multi-League Training Started")
        print("=" * 50)
        
        results = {}
        
        for league_code, league_info in self.league_files.items():
            print(f"\n📊 Training {league_info['name']}...")
            
            try:
                # Create predictor for this league
                predictor = AdvancedGPPredictor()
                
                # Train the model
                success, message = predictor.train_with_kernel_optimization(league_info['file'])
                
                if success:
                    self.league_models[league_code] = predictor
                    
                    # Extract performance metrics from message
                    cv_acc = self._extract_accuracy(message, 'CV')
                    test_acc = self._extract_accuracy(message, 'Test')
                    
                    results[league_code] = {
                        'name': league_info['name'],
                        'status': 'Success',
                        'cv_accuracy': cv_acc,
                        'test_accuracy': test_acc,
                        'message': message
                    }
                    
                    print(f"   ✅ {message}")
                else:
                    results[league_code] = {
                        'name': league_info['name'],
                        'status': 'Failed',
                        'cv_accuracy': 0,
                        'test_accuracy': 0,
                        'message': message
                    }
                    print(f"   ❌ {message}")
                    
            except Exception as e:
                results[league_code] = {
                    'name': league_info['name'],
                    'status': 'Error',
                    'cv_accuracy': 0,
                    'test_accuracy': 0,
                    'message': str(e)
                }
                print(f"   ⚠️ Error: {str(e)}")
        
        self.league_performance = results
        return results
    
    def _extract_accuracy(self, message, metric_type):
        """Mesajdan doğruluk oranını çıkar"""
        try:
            if metric_type in message:
                # Look for percentage after the metric type
                import re
                pattern = f"{metric_type}:?\s*(\d+\.?\d*)%"
                match = re.search(pattern, message)
                if match:
                    return float(match.group(1)) / 100
            return 0.0
        except:
            return 0.0
    
    def compare_league_performance(self):
        """Lig performanslarını karşılaştır"""
        if not self.league_performance:
            return {}
        
        print("\n📈 LEAGUE PERFORMANCE COMPARISON")
        print("=" * 50)
        
        # Sort by test accuracy
        sorted_leagues = sorted(
            self.league_performance.items(),
            key=lambda x: x[1]['test_accuracy'],
            reverse=True
        )
        
        comparison = {}
        
        print(f"{'Rank':<4} {'League':<25} {'CV Acc':<8} {'Test Acc':<8} {'Status'}")
        print("-" * 60)
        
        for rank, (league_code, perf) in enumerate(sorted_leagues, 1):
            status_icon = "✅" if perf['status'] == 'Success' else "❌"
            
            print(f"{rank:<4} {perf['name']:<25} {perf['cv_accuracy']:.1%:<8} {perf['test_accuracy']:.1%:<8} {status_icon}")
            
            comparison[league_code] = {
                'rank': rank,
                'name': perf['name'],
                'cv_accuracy': perf['cv_accuracy'],
                'test_accuracy': perf['test_accuracy'],
                'status': perf['status']
            }
        
        # Calculate statistics
        successful_leagues = [p for p in self.league_performance.values() if p['status'] == 'Success']
        
        if successful_leagues:
            test_accs = [p['test_accuracy'] for p in successful_leagues]
            cv_accs = [p['cv_accuracy'] for p in successful_leagues]
            
            print(f"\n📊 SUMMARY STATISTICS:")
            print(f"   Successful Models: {len(successful_leagues)}/{len(self.league_performance)}")
            print(f"   Average Test Accuracy: {np.mean(test_accs):.1%}")
            print(f"   Best Test Accuracy: {max(test_accs):.1%}")
            print(f"   Worst Test Accuracy: {min(test_accs):.1%}")
            print(f"   Std Dev: {np.std(test_accs):.1%}")
            
            comparison['summary'] = {
                'total_leagues': len(self.league_performance),
                'successful_leagues': len(successful_leagues),
                'avg_test_accuracy': np.mean(test_accs),
                'best_test_accuracy': max(test_accs),
                'worst_test_accuracy': min(test_accs),
                'std_test_accuracy': np.std(test_accs)
            }
        
        return comparison
    
    def analyze_league_characteristics(self):
        """Lig karakteristik analizi"""
        print("\n🔬 LEAGUE CHARACTERISTICS ANALYSIS")
        print("=" * 50)
        
        characteristics = {}
        
        for league_code, league_info in self.league_files.items():
            try:
                # Load league data
                with open(league_info['file'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Basic statistics
                total_matches = len(data)
                
                # Result distribution
                results = {'1': 0, 'X': 0, '2': 0}
                total_goals = 0
                high_scoring_matches = 0
                
                for match in data:
                    try:
                        score_data = match.get('score', {})
                        full_time = score_data.get('fullTime', {})
                        home_score = int(full_time.get('home', 0))
                        away_score = int(full_time.get('away', 0))
                        
                        total_goals += home_score + away_score
                        
                        if home_score + away_score > 3:
                            high_scoring_matches += 1
                        
                        if home_score > away_score:
                            results['1'] += 1
                        elif home_score == away_score:
                            results['X'] += 1
                        else:
                            results['2'] += 1
                    except:
                        continue
                
                # Calculate percentages
                home_win_pct = results['1'] / total_matches if total_matches > 0 else 0
                draw_pct = results['X'] / total_matches if total_matches > 0 else 0
                away_win_pct = results['2'] / total_matches if total_matches > 0 else 0
                avg_goals = total_goals / total_matches if total_matches > 0 else 0
                high_scoring_pct = high_scoring_matches / total_matches if total_matches > 0 else 0
                
                characteristics[league_code] = {
                    'name': league_info['name'],
                    'total_matches': total_matches,
                    'home_win_pct': home_win_pct,
                    'draw_pct': draw_pct,
                    'away_win_pct': away_win_pct,
                    'avg_goals': avg_goals,
                    'high_scoring_pct': high_scoring_pct,
                    'competitiveness': abs(home_win_pct - away_win_pct),  # Lower = more competitive
                    'predictability': max(home_win_pct, draw_pct, away_win_pct)  # Higher = more predictable
                }
                
                print(f"\n{league_info['name']}:")
                print(f"   Matches: {total_matches}")
                print(f"   Home Win: {home_win_pct:.1%}, Draw: {draw_pct:.1%}, Away Win: {away_win_pct:.1%}")
                print(f"   Avg Goals: {avg_goals:.2f}")
                print(f"   High Scoring (>3): {high_scoring_pct:.1%}")
                print(f"   Competitiveness: {characteristics[league_code]['competitiveness']:.1%}")
                
            except Exception as e:
                print(f"   ⚠️ Error analyzing {league_info['name']}: {e}")
                characteristics[league_code] = {'name': league_info['name'], 'error': str(e)}
        
        return characteristics
    
    def find_best_prediction_contexts(self):
        """En iyi tahmin kontekstlerini bul"""
        print("\n🎯 BEST PREDICTION CONTEXTS")
        print("=" * 50)
        
        contexts = {}
        
        for league_code, predictor in self.league_models.items():
            try:
                # Get feature importance
                feature_importance = predictor.calculate_feature_importance_ranking()
                
                # Top 5 most important features
                top_features = feature_importance[:5]
                
                contexts[league_code] = {
                    'name': self.league_files[league_code]['name'],
                    'top_features': top_features,
                    'model_available': True
                }
                
                print(f"\n{self.league_files[league_code]['name']}:")
                print("   Top 5 Critical Features:")
                for feature in top_features:
                    print(f"      {feature['rank']}. {feature['feature']} ({feature['importance_level']})")
                
            except Exception as e:
                contexts[league_code] = {
                    'name': self.league_files[league_code]['name'],
                    'error': str(e),
                    'model_available': False
                }
                print(f"   ⚠️ Error: {e}")
        
        return contexts
    
    def predict_cross_league(self, home_team, away_team, league_code, **kwargs):
        """Belirli lig için tahmin yap"""
        if league_code not in self.league_models:
            return None, 0.0, 0.0, "High", f"Model not available for {league_code}"
        
        predictor = self.league_models[league_code]
        return predictor.predict_with_risk_analysis(home_team, away_team, **kwargs)
    
    def generate_multi_league_report(self):
        """Kapsamlı multi-league raporu"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_comparison': self.compare_league_performance(),
            'league_characteristics': self.analyze_league_characteristics(),
            'prediction_contexts': self.find_best_prediction_contexts(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self):
        """Öneriler oluştur"""
        recommendations = []
        
        if self.league_performance:
            # Find best performing league
            best_league = max(
                self.league_performance.items(),
                key=lambda x: x[1]['test_accuracy'] if x[1]['status'] == 'Success' else 0
            )
            
            if best_league[1]['status'] == 'Success':
                recommendations.append({
                    'type': 'Best Model',
                    'message': f"Use {best_league[1]['name']} model for highest accuracy ({best_league[1]['test_accuracy']:.1%})"
                })
            
            # Find leagues with similar characteristics
            successful_leagues = [(k, v) for k, v in self.league_performance.items() if v['status'] == 'Success']
            
            if len(successful_leagues) >= 2:
                recommendations.append({
                    'type': 'Cross-Validation',
                    'message': f"Cross-validate predictions across {len(successful_leagues)} successful models"
                })
            
            # Model reliability
            accuracies = [v['test_accuracy'] for k, v in self.league_performance.items() if v['status'] == 'Success']
            if accuracies:
                avg_acc = np.mean(accuracies)
                if avg_acc > 0.5:
                    recommendations.append({
                        'type': 'Reliability',
                        'message': f"Models show good average performance ({avg_acc:.1%}). Consider ensemble approach."
                    })
                else:
                    recommendations.append({
                        'type': 'Caution',
                        'message': f"Models show moderate performance ({avg_acc:.1%}). Use with caution and additional analysis."
                    })
        
        return recommendations

def main():
    """Multi-league analiz çalıştır"""
    analyzer = MultiLeagueGPAnalyzer()
    
    # Train all available leagues
    print("Starting multi-league training...")
    training_results = analyzer.train_all_leagues()
    
    # Generate comprehensive report
    report = analyzer.generate_multi_league_report()
    
    # Save report
    with open('multi_league_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Report saved to: multi_league_analysis_report.json")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
