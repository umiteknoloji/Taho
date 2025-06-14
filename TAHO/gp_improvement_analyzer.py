#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GP Improvement Analyzer
Backtest sonuçlarını analiz edip GP iyileştirme önerileri sunar
"""

def analyze_backtest_failures(predictions, target_accuracy=100.0):
    """Backtest başarısızlıklarını analiz et ve iyileştirme önerileri sun"""
    
    total_predictions = len(predictions)
    correct_predictions = sum(1 for p in predictions if p['is_correct'])
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    failures = [p for p in predictions if not p['is_correct']]
    
    analysis = {
        'total_matches': total_predictions,
        'correct_predictions': correct_predictions,
        'failed_predictions': len(failures),
        'accuracy': accuracy,
        'target_reached': accuracy >= target_accuracy,
        'failures': failures,
        'patterns': {},
        'recommendations': []
    }
    
    if not failures:
        analysis['recommendations'] = ["🏆 MÜKEMMEL! %100 doğruluk elde edildi!"]
        return analysis
    
    # Pattern analysis
    patterns = analyze_failure_patterns(failures)
    analysis['patterns'] = patterns
    
    # Generate recommendations
    recommendations = generate_improvement_recommendations(patterns, failures)
    analysis['recommendations'] = recommendations
    
    return analysis

def analyze_failure_patterns(failures):
    """Başarısızlık pattern'lerini analiz et"""
    patterns = {
        'prediction_errors': {},
        'confidence_analysis': {
            'high_confidence_errors': 0,
            'low_confidence_errors': 0,
            'avg_failed_confidence': 0
        },
        'result_type_analysis': {
            'home_wins_missed': 0,
            'draws_missed': 0,
            'away_wins_missed': 0
        },
        'error_types': {
            'predicted_home_actual_away': 0,
            'predicted_home_actual_draw': 0,
            'predicted_away_actual_home': 0,
            'predicted_away_actual_draw': 0,
            'predicted_draw_actual_home': 0,
            'predicted_draw_actual_away': 0
        }
    }
    
    for failure in failures:
        predicted = failure['predicted']
        actual = failure['actual']
        confidence = failure['confidence']
        
        # Prediction error types
        error_key = f"{predicted} -> {actual}"
        patterns['prediction_errors'][error_key] = patterns['prediction_errors'].get(error_key, 0) + 1
        
        # Confidence analysis
        if confidence > 0.8:
            patterns['confidence_analysis']['high_confidence_errors'] += 1
        else:
            patterns['confidence_analysis']['low_confidence_errors'] += 1
        
        # Result type analysis
        if actual == '1':
            patterns['result_type_analysis']['home_wins_missed'] += 1
        elif actual == 'X':
            patterns['result_type_analysis']['draws_missed'] += 1
        elif actual == '2':
            patterns['result_type_analysis']['away_wins_missed'] += 1
        
        # Detailed error types
        if predicted == '1' and actual == '2':
            patterns['error_types']['predicted_home_actual_away'] += 1
        elif predicted == '1' and actual == 'X':
            patterns['error_types']['predicted_home_actual_draw'] += 1
        elif predicted == '2' and actual == '1':
            patterns['error_types']['predicted_away_actual_home'] += 1
        elif predicted == '2' and actual == 'X':
            patterns['error_types']['predicted_away_actual_draw'] += 1
        elif predicted == 'X' and actual == '1':
            patterns['error_types']['predicted_draw_actual_home'] += 1
        elif predicted == 'X' and actual == '2':
            patterns['error_types']['predicted_draw_actual_away'] += 1
    
    # Calculate average failed confidence
    if failures:
        patterns['confidence_analysis']['avg_failed_confidence'] = sum(f['confidence'] for f in failures) / len(failures)
    
    return patterns

def generate_improvement_recommendations(patterns, failures):
    """Pattern'lere göre iyileştirme önerileri oluştur"""
    recommendations = []
    
    # High confidence errors
    high_conf_errors = patterns['confidence_analysis']['high_confidence_errors']
    if high_conf_errors > 0:
        recommendations.append({
            'type': 'overconfidence',
            'priority': 'HIGH',
            'title': '⚠️ Aşırı Güven Problemi',
            'description': f'{high_conf_errors} yüksek güvenli yanlış tahmin var (%80+ güven)',
            'solutions': [
                'Model kalibrasyonu uygula (Platt scaling)',
                'Ensemble voting ile güven skorlarını yumuşat',
                'Uncertainty quantification ekle',
                'Cross-validation ile overconfidence tespit et'
            ]
        })
    
    # Home advantage problems
    home_wins_missed = patterns['result_type_analysis']['home_wins_missed']
    if home_wins_missed >= 2:
        recommendations.append({
            'type': 'home_advantage',
            'priority': 'HIGH',
            'title': '🏠 Ev Sahibi Avantajı Eksikliği',
            'description': f'{home_wins_missed} ev sahibi galibiyeti kaçırıldı',
            'solutions': [
                'Ev sahibi-spesifik feature\'lar ekle',
                'Son N maçta ev sahibi performansı',
                'Ev sahibi vs deplasman gol ortalaması farkı',
                'Takım-spesifik ev sahibi avantajı katsayıları',
                'Stadyum kapasitesi ve atmosfer faktörleri'
            ]
        })
    
    # Draw prediction problems
    draws_missed = patterns['result_type_analysis']['draws_missed']
    predicted_draw_errors = patterns['error_types']['predicted_draw_actual_home'] + patterns['error_types']['predicted_draw_actual_away']
    
    if draws_missed >= 2 or predicted_draw_errors >= 2:
        recommendations.append({
            'type': 'draw_prediction',
            'priority': 'MEDIUM',
            'title': '⚖️ Beraberlik Tahmin Problemi',
            'description': f'{draws_missed} beraberlik kaçırıldı, {predicted_draw_errors} yanlış beraberlik tahmini',
            'solutions': [
                'Beraberlik-spesifik feature\'lar ekle',
                'Defensive strength balance hesapla',
                'Takım gol ortalamaları dengesizliği',
                'Son maçlarda beraberlik eğilimi',
                'H2H beraberlik geçmişi',
                'Beraberlik için ayrı binary classifier'
            ]
        })
    
    # Away team underestimation
    away_wins_missed = patterns['result_type_analysis']['away_wins_missed']
    predicted_away_errors = patterns['error_types']['predicted_away_actual_home']
    
    if predicted_away_errors >= 2:
        recommendations.append({
            'type': 'away_underestimation',
            'priority': 'MEDIUM', 
            'title': '✈️ Deplasman Takımı Hafife Alma',
            'description': f'{predicted_away_errors} deplasman galibiyeti ev sahibi olarak tahmin edildi',
            'solutions': [
                'Deplasman takımı strength faktörünü artır',
                'Deplasman performans trendleri',
                'Ev sahibi takımının son ev sahibi performansı',
                'Deplasman takımının motivasyon faktörleri',
                'Travel distance ve fixture congestion'
            ]
        })
    
    # Overall model improvements
    if len(failures) > len([r for r in recommendations if r['priority'] == 'HIGH']):
        recommendations.append({
            'type': 'model_architecture',
            'priority': 'MEDIUM',
            'title': '🧠 Model Mimarisi İyileştirmeleri',
            'description': 'Genel model performansını artırmak için',
            'solutions': [
                'Gaussian Process kernel optimizasyonu',
                'Feature selection algoritması güncelle',
                'Temporal features ekle (hafta, sezon dönemi)',
                'Player injury ve suspension data',
                'Weather conditions',
                'Recent transfer activity impact',
                'Ensemble model ağırlıklarını optimize et'
            ]
        })
    
    # Data quality improvements
    recommendations.append({
        'type': 'data_quality',
        'priority': 'LOW',
        'title': '📊 Veri Kalitesi İyileştirmeleri',
        'description': 'Daha iyi veri ile model performansını artır',
        'solutions': [
            'Daha fazla geçmiş sezon verisi',
            'Player-level statistics',
            'Tactical formation data',
            'Team news ve motivation factors',
            'Market odds data for calibration',
            'Expected goals (xG) statistics'
        ]
    })
    
    return recommendations

def format_analysis_for_web(analysis):
    """Analizi web arayüzü için formatla"""
    
    if analysis['target_reached']:
        return {
            'status': 'perfect',
            'message': '🏆 MÜKEMMEL! %100 doğruluk elde edildi!',
            'accuracy': 100.0,
            'details': 'Tüm tahminler doğru. Parametreleri kaydedin!'
        }
    
    # Failed matches details
    failed_matches = []
    for failure in analysis['failures']:
        failed_matches.append({
            'match': failure['match'],
            'predicted': failure['predicted'],
            'actual': failure['actual'],
            'confidence': f"{failure['confidence']:.1%}",
            'score': f"{failure['home_score']}-{failure['away_score']}"
        })
    
    # Priority recommendations
    high_priority = [r for r in analysis['recommendations'] if r.get('priority') == 'HIGH']
    medium_priority = [r for r in analysis['recommendations'] if r.get('priority') == 'MEDIUM']
    
    return {
        'status': 'needs_improvement',
        'accuracy': analysis['accuracy'],
        'failed_matches': failed_matches,
        'high_priority_fixes': high_priority,
        'medium_priority_fixes': medium_priority,
        'patterns': analysis['patterns']
    }

# Test function
def test_analysis():
    """Test analiz sistemi"""
    
    # Sample failed predictions
    sample_failures = [
        {
            'match': 'Borussia Dortmund vs Freiburg',
            'predicted': '2', 'actual': '1',
            'confidence': 0.973, 'is_correct': False,
            'home_score': 4, 'away_score': 0
        },
        {
            'match': 'Wolfsburg vs Union Berlin', 
            'predicted': '2', 'actual': '1',
            'confidence': 0.938, 'is_correct': False,
            'home_score': 1, 'away_score': 0
        },
        {
            'match': 'Stuttgart vs Bochum',
            'predicted': 'X', 'actual': '1', 
            'confidence': 0.673, 'is_correct': False,
            'home_score': 2, 'away_score': 0
        }
    ]
    
    analysis = analyze_backtest_failures(sample_failures)
    web_format = format_analysis_for_web(analysis)
    
    print("🔍 ANALİZ SONUÇLARI:")
    print(f"Başarı oranı: {analysis['accuracy']:.1f}%")
    print(f"Başarısız tahmin: {len(analysis['failures'])}")
    
    for rec in analysis['recommendations']:
        if rec.get('priority') == 'HIGH':
            print(f"\n⚠️ {rec['title']}")
            print(f"   {rec['description']}")
    
    return analysis

if __name__ == "__main__":
    test_analysis()
