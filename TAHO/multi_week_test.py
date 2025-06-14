#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-week accuracy test
Farklı haftalarda accuracy'yi test eder
"""

from no_h2h_gp import NoH2HGP

def test_multiple_weeks():
    """Farklı haftalarda accuracy test et"""
    
    weeks_to_test = [8, 9, 10, 11, 12, 13, 14]
    results_summary = []
    
    for week in weeks_to_test:
        try:
            print(f"\n{'='*60}")
            print(f"🗓️ WEEK {week} TEST")
            print(f"{'='*60}")
            
            predictor = NoH2HGP()
            train_data, test_data = predictor.load_and_analyze_data('data/ALM_stat.json', week)
            
            if len(test_data) == 0:
                print(f"Week {week}: Veri yok")
                continue
                
            predictor.train_model(train_data)
            results = predictor.predict_matches(test_data)
            
            # Calculate accuracy
            correct = sum(1 for r in results if r['is_correct'])
            total = len(results)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            results_summary.append({
                'week': week,
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            })
            
            print(f"📊 Week {week}: {correct}/{total} = {accuracy:.1f}%")
            
            # Show failed predictions
            failed = [r for r in results if not r['is_correct']]
            if failed:
                print(f"❌ Başarısız tahminler:")
                for f in failed:
                    print(f"   {f['home_team']} vs {f['away_team']}: "
                          f"{f['predicted']} → {f['actual']} ({f['home_score']}-{f['away_score']})")
            
        except Exception as e:
            print(f"Week {week} hatası: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 HAFTALIK ACCURACY ÖZETİ")
    print(f"{'='*60}")
    
    total_correct = sum(r['correct'] for r in results_summary)
    total_matches = sum(r['total'] for r in results_summary)
    overall_accuracy = (total_correct / total_matches * 100) if total_matches > 0 else 0
    
    print(f"{'Week':<6} {'Doğru':<6} {'Toplam':<7} {'Accuracy':<10}")
    print("-" * 35)
    
    for r in results_summary:
        print(f"{r['week']:<6} {r['correct']:<6} {r['total']:<7} {r['accuracy']:<10.1f}%")
    
    print("-" * 35)
    print(f"{'GENEL':<6} {total_correct:<6} {total_matches:<7} {overall_accuracy:<10.1f}%")
    
    # Find best and worst weeks
    if results_summary:
        best_week = max(results_summary, key=lambda x: x['accuracy'])
        worst_week = min(results_summary, key=lambda x: x['accuracy'])
        
        print(f"\n🏆 En iyi hafta: Week {best_week['week']} ({best_week['accuracy']:.1f}%)")
        print(f"💥 En kötü hafta: Week {worst_week['week']} ({worst_week['accuracy']:.1f}%)")

if __name__ == "__main__":
    test_multiple_weeks()
