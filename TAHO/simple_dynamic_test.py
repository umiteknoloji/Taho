#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Dynamic Threshold Comparison Test
"""

from enhanced_no_h2h_gp_dynamic import EnhancedNoH2HGP
import numpy as np

def simple_dynamic_test():
    """Basit dynamic threshold testi"""
    print("🎯 SIMPLE DYNAMIC THRESHOLD TEST")
    print("=" * 50)
    
    league_file = "/Users/umitduman/Taho/data/ALM_stat.json"
    test_weeks = [10, 11, 12]
    
    results = []
    
    for week in test_weeks:
        print(f"\n📅 Testing Week {week}...")
        
        try:
            predictor = EnhancedNoH2HGP()
            train_data, test_data = predictor.load_and_analyze_data(league_file, week)
            
            if len(test_data) == 0:
                print(f"   ❌ No test data for week {week}")
                continue
            
            predictor.train(train_data)
            
            # Test
            correct = 0
            total = 0
            
            for match in test_data:
                actual = predictor._get_match_result(match)
                predicted, _, confidence = predictor.predict_match(match)
                
                if predicted:
                    total += 1
                    if predicted == actual:
                        correct += 1
            
            accuracy = correct / total if total > 0 else 0
            
            print(f"   ✅ Week {week}: {accuracy:.3f} ({correct}/{total})")
            print(f"   🔧 Features: {len(predictor.feature_names)}")
            print(f"   🎯 Home Advantage: {predictor.dynamic_thresholds.get('home_advantage', 0):.3f}")
            
            results.append({
                'week': week,
                'accuracy': accuracy,
                'matches': total,
                'home_advantage': predictor.dynamic_thresholds.get('home_advantage', 0),
                'features': len(predictor.feature_names)
            })
            
        except Exception as e:
            print(f"   ❌ Week {week} Error: {str(e)}")
    
    # Summary
    if results:
        accuracies = [r['accuracy'] for r in results]
        print(f"\n📊 SUMMARY:")
        print(f"   Average Accuracy: {np.mean(accuracies):.3f}")
        print(f"   Best Week: {max(results, key=lambda x: x['accuracy'])['week']} ({max(accuracies):.3f})")
        print(f"   Worst Week: {min(results, key=lambda x: x['accuracy'])['week']} ({min(accuracies):.3f})")
        
        print(f"\n🔧 DYNAMIC THRESHOLD ANALYSIS:")
        home_advs = [r['home_advantage'] for r in results]
        features = [r['features'] for r in results]
        
        print(f"   Home Advantage Range: {min(home_advs):.3f} - {max(home_advs):.3f}")
        print(f"   Feature Count Range: {min(features)} - {max(features)}")
        
        # Correlation with accuracy
        if len(accuracies) > 1:
            corr_ha = np.corrcoef(home_advs, accuracies)[0, 1]
            corr_features = np.corrcoef(features, accuracies)[0, 1]
            print(f"   HA vs Accuracy Correlation: {corr_ha:.3f}")
            print(f"   Features vs Accuracy Correlation: {corr_features:.3f}")
    
    return results

if __name__ == "__main__":
    results = simple_dynamic_test()
