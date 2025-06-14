#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Dynamic Threshold Analysis
Farklı haftalarda threshold değişimlerini ve accuracy'yi analiz eder
"""

from enhanced_no_h2h_gp_dynamic import EnhancedNoH2HGP
import json

def analyze_week(week_num):
    """Tek hafta için detailed analiz"""
    print(f"\n🎯 WEEK {week_num} ANALYSIS")
    print("-" * 40)
    
    predictor = EnhancedNoH2HGP()
    train_data, test_data = predictor.load_and_analyze_data(
        '/Users/umitduman/Taho/data/ALM_stat.json', week_num
    )
    
    print(f"📊 Training data: {len(train_data)} matches")
    print(f"🎯 Test data: {len(test_data)} matches")
    
    # Dynamic thresholds
    thresholds = predictor.dynamic_thresholds
    print(f"🔧 Dynamic Thresholds:")
    for key, value in thresholds.items():
        if isinstance(value, list):
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value:.4f}")
    
    # Train model
    if predictor.train(train_data):
        print(f"✅ Model trained successfully")
        print(f"📈 Features used: {len(predictor.feature_names)}")
        
        # Test predictions
        correct = 0
        total = 0
        detailed_results = []
        
        for match in test_data:
            actual = predictor._get_match_result(match)
            predicted, prob_dict, confidence = predictor.predict_match(match)
            
            if predicted:
                total += 1
                is_correct = (predicted == actual)
                if is_correct:
                    correct += 1
                
                detailed_results.append({
                    'home': match.get('home', ''),
                    'away': match.get('away', ''),
                    'actual': actual,
                    'predicted': predicted,
                    'confidence': confidence,
                    'correct': is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        print(f"🎯 Accuracy: {accuracy:.3f} ({correct}/{total})")
        
        # Show predictions
        print(f"📋 Detailed Results:")
        for result in detailed_results:
            status = "✅" if result['correct'] else "❌"
            print(f"   {status} {result['home']} vs {result['away']}")
            print(f"      Actual: {result['actual']}, Predicted: {result['predicted']}, Conf: {result['confidence']:.3f}")
        
        return {
            'week': week_num,
            'accuracy': accuracy,
            'matches': total,
            'thresholds': thresholds,
            'features': len(predictor.feature_names),
            'training_size': len(train_data),
            'results': detailed_results
        }
    else:
        print("❌ Model training failed")
        return {'week': week_num, 'error': 'Training failed'}

def compare_weeks():
    """Farklı haftalarda karşılaştırma yap"""
    print("🚀 DYNAMIC THRESHOLD MULTI-WEEK COMPARISON")
    print("=" * 60)
    
    weeks_to_test = [10, 11, 12, 13]
    results = []
    
    for week in weeks_to_test:
        try:
            result = analyze_week(week)
            if 'accuracy' in result:
                results.append(result)
        except Exception as e:
            print(f"❌ Week {week} error: {str(e)}")
    
    if len(results) >= 2:
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 60)
        
        print("Week | Accuracy | Matches | Home Adv | Max Feat | Training")
        print("-" * 60)
        
        for r in results:
            week = r['week']
            acc = r['accuracy']
            matches = r['matches']
            home_adv = r['thresholds'].get('home_advantage', 0)
            max_feat = r['thresholds'].get('max_features', 0)
            training = r['training_size']
            
            print(f"{week:4d} | {acc:8.3f} | {matches:7d} | {home_adv:8.3f} | {max_feat:8d} | {training:8d}")
        
        # Analysis
        accuracies = [r['accuracy'] for r in results]
        home_advantages = [r['thresholds'].get('home_advantage', 0) for r in results]
        max_features = [r['thresholds'].get('max_features', 0) for r in results]
        
        print(f"\n🔍 ANALYSIS:")
        print(f"   Average Accuracy: {sum(accuracies)/len(accuracies):.3f}")
        print(f"   Best Week: {results[accuracies.index(max(accuracies))]['week']} ({max(accuracies):.3f})")
        print(f"   Worst Week: {results[accuracies.index(min(accuracies))]['week']} ({min(accuracies):.3f})")
        
        print(f"\n🔧 THRESHOLD VARIANCE:")
        print(f"   Home Advantage: {min(home_advantages):.3f} - {max(home_advantages):.3f}")
        print(f"   Max Features: {min(max_features)} - {max(max_features)}")
        
        # Best performing week analysis
        best_week = results[accuracies.index(max(accuracies))]
        print(f"\n🏆 BEST WEEK ANALYSIS (Week {best_week['week']}):")
        print(f"   Accuracy: {best_week['accuracy']:.3f}")
        print(f"   Training Size: {best_week['training_size']}")
        print(f"   Features Used: {best_week['features']}")
        print(f"   Key Thresholds:")
        for key, value in best_week['thresholds'].items():
            if not isinstance(value, list):
                print(f"      {key}: {value:.4f}")
        
        # Show which thresholds correlate with performance
        print(f"\n💡 THRESHOLD IMPACT:")
        
        # Simple correlation analysis
        if len(accuracies) > 2:
            import numpy as np
            
            # Home advantage correlation
            if len(set(home_advantages)) > 1:
                corr = np.corrcoef(home_advantages, accuracies)[0, 1]
                print(f"   Home Advantage vs Accuracy: r = {corr:.3f}")
            
            # Feature count correlation
            feature_counts = [r['features'] for r in results]
            if len(set(feature_counts)) > 1:
                corr = np.corrcoef(feature_counts, accuracies)[0, 1]
                print(f"   Feature Count vs Accuracy: r = {corr:.3f}")
            
            # Training size correlation
            training_sizes = [r['training_size'] for r in results]
            if len(set(training_sizes)) > 1:
                corr = np.corrcoef(training_sizes, accuracies)[0, 1]
                print(f"   Training Size vs Accuracy: r = {corr:.3f}")
    
    return results

if __name__ == "__main__":
    results = compare_weeks()
    
    # Final summary
    if results:
        print(f"\n🎯 FINAL DYNAMIC THRESHOLD SUMMARY:")
        print("=" * 60)
        accuracies = [r['accuracy'] for r in results]
        avg_accuracy = sum(accuracies) / len(accuracies)
        
        print(f"📊 Weeks tested: {len(results)}")
        print(f"🎯 Average accuracy: {avg_accuracy:.3f}")
        print(f"📈 Accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f}")
        
        # Compare with historical static performance
        static_baseline = 0.65  # Estimated from previous tests
        improvement = avg_accuracy - static_baseline
        
        print(f"\n💡 IMPROVEMENT vs STATIC THRESHOLDS:")
        print(f"   Static baseline (estimated): {static_baseline:.3f}")
        print(f"   Dynamic average: {avg_accuracy:.3f}")
        print(f"   Improvement: {improvement:+.3f} ({(improvement/static_baseline)*100:+.1f}%)")
        
        if improvement > 0:
            print(f"   ✅ Dynamic thresholds show improvement!")
        else:
            print(f"   ⚠️  Dynamic thresholds need further optimization")
