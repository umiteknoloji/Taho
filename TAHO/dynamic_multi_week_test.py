#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Threshold Multi-Week Test
Farklı haftalarda dynamic threshold sisteminin performansını test eder
"""

import json
import numpy as np
from enhanced_no_h2h_gp_dynamic import EnhancedNoH2HGP
import warnings
warnings.filterwarnings('ignore')

def multi_week_dynamic_test(league_file, test_weeks):
    """Birden fazla hafta için dynamic threshold testi"""
    results = {}
    
    print(f"🚀 DYNAMIC THRESHOLD MULTI-WEEK TEST")
    print(f"📁 Liga: {league_file}")
    print(f"📅 Test haftaları: {test_weeks}")
    print("=" * 60)
    
    for week in test_weeks:
        print(f"\n🎯 WEEK {week} TESTING...")
        print("-" * 40)
        
        try:
            # Her hafta için yeni predictor oluştur (dynamic thresholds yeniden hesaplanır)
            predictor = EnhancedNoH2HGP()
            
            # Load and train
            train_data, test_data = predictor.load_and_analyze_data(league_file, week)
            
            if len(test_data) == 0:
                print(f"❌ Week {week}: Test data yok")
                continue
                
            if not predictor.train(train_data):
                print(f"❌ Week {week}: Training başarısız")
                continue
            
            # Test predictions
            correct = 0
            total = 0
            predictions = []
            
            for match in test_data:
                actual_result = predictor._get_match_result(match)
                predicted_result, prob_dict, confidence = predictor.predict_match(match)
                
                if predicted_result:
                    total += 1
                    if predicted_result == actual_result:
                        correct += 1
                    
                    predictions.append({
                        'home': match.get('home', ''),
                        'away': match.get('away', ''),
                        'actual': actual_result,
                        'predicted': predicted_result,
                        'confidence': confidence,
                        'probabilities': prob_dict
                    })
            
            accuracy = correct / total if total > 0 else 0
            
            # Dynamic thresholds bilgisi
            dynamic_info = {
                'thresholds': predictor.dynamic_thresholds,
                'feature_count': len(predictor.feature_names),
                'training_size': len(train_data)
            }
            
            results[week] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'predictions': predictions,
                'dynamic_info': dynamic_info
            }
            
            print(f"✅ Week {week}: {accuracy:.3f} ({correct}/{total})")
            print(f"🔧 Dynamic Features: {len(predictor.feature_names)}")
            print(f"📊 Key Thresholds:")
            print(f"   Home Advantage: {predictor.dynamic_thresholds.get('home_advantage', 0):.3f}")
            print(f"   Defensive Factor: {predictor.dynamic_thresholds.get('defensive_factor', 0):.3f}")
            print(f"   Max Features: {predictor.dynamic_thresholds.get('max_features', 0)}")
            
        except Exception as e:
            print(f"❌ Week {week} Error: {str(e)}")
            results[week] = {'error': str(e)}
    
    return results

def analyze_dynamic_results(results):
    """Dynamic test sonuçlarını analiz et"""
    print(f"\n📊 DYNAMIC THRESHOLD ANALYSIS SUMMARY")
    print("=" * 60)
    
    successful_weeks = [week for week, data in results.items() if 'accuracy' in data]
    
    if not successful_weeks:
        print("❌ Başarılı test yok!")
        return
    
    accuracies = [results[week]['accuracy'] for week in successful_weeks]
    
    print(f"📈 Test edilen haftalar: {len(successful_weeks)}")
    print(f"🎯 Ortalama Accuracy: {np.mean(accuracies):.3f}")
    print(f"📊 Std Deviation: {np.std(accuracies):.3f}")
    print(f"🏆 En iyi hafta: Week {successful_weeks[np.argmax(accuracies)]} ({max(accuracies):.3f})")
    print(f"📉 En kötü hafta: Week {successful_weeks[np.argmin(accuracies)]} ({min(accuracies):.3f})")
    
    print(f"\n📋 WEEKLY BREAKDOWN:")
    print("Week | Accuracy | Matches | Dynamic Thresholds")
    print("-" * 50)
    
    for week in successful_weeks:
        data = results[week]
        accuracy = data['accuracy']
        matches = data['total']
        home_adv = data['dynamic_info']['thresholds'].get('home_advantage', 0)
        features = data['dynamic_info']['feature_count']
        
        print(f"{week:4d} | {accuracy:8.3f} | {matches:7d} | HA:{home_adv:.3f} F:{features}")
    
    # Threshold consistency analysis
    print(f"\n🔧 DYNAMIC THRESHOLD CONSISTENCY:")
    
    # Collect all thresholds
    threshold_keys = set()
    for week in successful_weeks:
        threshold_keys.update(results[week]['dynamic_info']['thresholds'].keys())
    
    for key in sorted(threshold_keys):
        values = []
        for week in successful_weeks:
            if key in results[week]['dynamic_info']['thresholds']:
                values.append(results[week]['dynamic_info']['thresholds'][key])
        
        if values and isinstance(values[0], (int, float)):
            print(f"   {key:20s}: {np.mean(values):8.3f} (±{np.std(values):.3f})")
    
    # Performance correlation with threshold values
    print(f"\n🔍 THRESHOLD IMPACT ANALYSIS:")
    
    # Home advantage vs accuracy correlation
    home_advantages = []
    acc_for_ha = []
    for week in successful_weeks:
        ha = results[week]['dynamic_info']['thresholds'].get('home_advantage', 0)
        acc = results[week]['accuracy']
        home_advantages.append(ha)
        acc_for_ha.append(acc)
    
    if len(home_advantages) > 1:
        correlation = np.corrcoef(home_advantages, acc_for_ha)[0, 1]
        print(f"   Home Advantage vs Accuracy: r={correlation:.3f}")
    
    # Feature count vs accuracy
    feature_counts = [results[week]['dynamic_info']['feature_count'] for week in successful_weeks]
    if len(feature_counts) > 1:
        correlation = np.corrcoef(feature_counts, accuracies)[0, 1]
        print(f"   Feature Count vs Accuracy:   r={correlation:.3f}")

def compare_static_vs_dynamic():
    """Static vs Dynamic threshold karşılaştırması"""
    print(f"\n⚡ STATIC vs DYNAMIC COMPARISON")
    print("=" * 60)
    
    # Test weeks
    test_weeks = [10, 11, 12, 13]
    league_file = "/Users/umitduman/Taho/data/ALM_stat.json"
    
    # Dynamic test
    print("🔧 Testing DYNAMIC thresholds...")
    dynamic_results = multi_week_dynamic_test(league_file, test_weeks)
    
    print(f"\n📊 COMPARISON RESULTS:")
    
    dynamic_accuracies = []
    for week in test_weeks:
        if week in dynamic_results and 'accuracy' in dynamic_results[week]:
            dynamic_accuracies.append(dynamic_results[week]['accuracy'])
    
    if dynamic_accuracies:
        avg_dynamic = np.mean(dynamic_accuracies)
        print(f"🎯 Dynamic Average: {avg_dynamic:.3f}")
        print(f"📊 Dynamic StdDev:  {np.std(dynamic_accuracies):.3f}")
        print(f"🏆 Dynamic Best:    {max(dynamic_accuracies):.3f}")
        print(f"📉 Dynamic Worst:   {min(dynamic_accuracies):.3f}")
        
        # Theoretical static comparison (based on previous tests)
        static_baseline = 0.65  # Estimated from previous static tests
        improvement = avg_dynamic - static_baseline
        print(f"\n💡 IMPROVEMENT ANALYSIS:")
        print(f"   Static Baseline (est):  {static_baseline:.3f}")
        print(f"   Dynamic Average:        {avg_dynamic:.3f}")
        print(f"   Improvement:            {improvement:+.3f}")
        print(f"   Improvement %:          {(improvement/static_baseline)*100:+.1f}%")
    
    return dynamic_results

if __name__ == "__main__":
    # Comprehensive dynamic threshold test
    results = compare_static_vs_dynamic()
    analyze_dynamic_results(results)
