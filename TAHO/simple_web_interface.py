#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Backtest Web Interface
Sadece backtest odaklı, %100 doğruluk hedefli basit arayüz
"""

from flask import Flask, render_template, request, jsonify
import json
import os
import sys

# Backtest sistemini import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parametric_backtest import ParametricBacktestSystem
from gp_improvement_analyzer import analyze_backtest_failures, format_analysis_for_web
from perfect_accuracy_gp import PerfectAccuracyGP
from no_h2h_gp import NoH2HGP

app = Flask(__name__)
app.secret_key = 'simple_backtest_2025'

def get_available_weeks(data_file):
    """Veri dosyasındaki mevcut hafta sayısını hesapla"""
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tüm hafta numaralarını al
        weeks = [int(match.get('week', 0)) for match in data if match.get('week')]
        
        if weeks:
            max_week = max(weeks)
            print(f"📊 {data_file} - Maksimum hafta: {max_week}")
            return max_week
        else:
            print(f"⚠️ {data_file} - Hafta bilgisi bulunamadı, varsayılan 20 kullanılacak")
            return 20
            
    except Exception as e:
        print(f"❌ Hafta sayısı hesaplanırken hata: {e}")
        return 20  # Varsayılan değer

# Global backtest system
backtest_system = ParametricBacktestSystem()

# Desteklenen veri dosyaları
SUPPORTED_DATA_FILES = {
    'data/TR_stat.json': '🇹🇷 Türkiye Süper Lig',
    'data/ENG_stat.json': '🏴 İngiliz Premier Ligi',
    'data/ESP_stat.json': '🇪🇸 İspanya La Liga',
    'data/ALM_stat.json': '🇩🇪 Alman Bundesliga',
    'data/FRA_stat.json': '🇫🇷 Fransa Ligue 1',
    'data/HOL_stat.json': '🇳🇱 Hollanda Eredivisie',
    'data/POR_stat.json': '🇵🇹 Portekiz Ligi',
    'data/BEL_stat.json': '🇧🇪 Belçika Ligi',
}

@app.route('/')
def index():
    """Ana sayfa - basit backtest arayüzü"""
    return render_template('simple_backtest.html', 
                         data_files=SUPPORTED_DATA_FILES)

@app.route('/api/available-weeks')
def available_weeks():
    """Seçilen veri dosyası için mevcut haftaları döndür"""
    try:
        data_file = request.args.get('data_file', 'data/TR_stat.json')
        
        if not os.path.exists(data_file):
            return jsonify({"error": "Veri dosyası bulunamadı"}), 404
            
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        weeks = []
        total_matches = len(data)
        
        for match in data:
            week = match.get('week', 0)
            if week and week not in weeks:
                weeks.append(week)
        
        weeks.sort()
        
        return jsonify({
            "success": True,
            "weeks": weeks,
            "total_matches": total_matches,
            "data_file": data_file
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-backtest', methods=['POST'])
def run_backtest():
    """Backtest çalıştır - Sadece No H2H GP modeli"""
    try:
        data = request.get_json()
        
        # Parametreleri al
        data_file = data.get('data_file', 'data/TR_stat.json')
        test_week = data.get('test_week', 25)
        
        # Dinamik train_weeks hesapla - test week'ten önce kaç hafta var
        available_weeks = get_available_weeks(data_file)
        train_weeks = min(test_week - 1, available_weeks)  # Test week'ten bir önceki haftalara kadar
        
        # Sabit parametreler - No H2H GP için optimize edilmiş
        min_confidence = 0.0
        min_matches = 0
        
        print(f"🎯 Dinamik Parametreler: test_week={test_week}, train_weeks={train_weeks}, available_weeks={available_weeks}")
        
        # Sadece No H2H GP modeli kullan
        predictor = NoH2HGP()
        train_data, test_data = predictor.load_and_analyze_data(data_file, test_week)
        predictor.train_model(train_data)
        predictions = predictor.predict_matches(test_data)
        
        # Sonuçları uygun formata çevir
        results = {
            'predictions': []
        }
        
        for pred in predictions:
            # Her prediction'ı tuple formatına çevir: (match_name, pred_data, score_tuple)
            match_name = pred['match']
            pred_data = {
                'predicted_result': pred['predicted'],
                'confidence': pred['confidence'],
                'probabilities': pred['probabilities'],
                'actual_result': pred['actual'],
                'is_correct': pred['is_correct']
            }
            score_tuple = (pred['home_score'], pred['away_score'])
            
            results['predictions'].append((match_name, pred_data, score_tuple))
        
        if results:
            # Sonuçları formatla
            formatted_results = format_backtest_results(results, min_matches, min_confidence)
            
            # GP iyileştirme analizi yap
            gp_analysis = None
            if formatted_results['predictions']:
                # GP analyzer için doğru format: orijinal predictions'dan skor bilgisini de al
                analyzer_predictions = []
                for i, pred_formatted in enumerate(formatted_results['predictions']):
                    # Orijinal prediction'dan skor bilgisini al
                    orig_pred = predictions[i] if i < len(predictions) else {}
                    
                    analyzer_pred = {
                        'match': f"{pred_formatted['home_team']} vs {pred_formatted['away_team']}",
                        'predicted': pred_formatted['predicted'],
                        'actual': pred_formatted['actual'],
                        'confidence': pred_formatted['confidence'],
                        'is_correct': pred_formatted['is_correct'],
                        'home_team': pred_formatted['home_team'],
                        'away_team': pred_formatted['away_team'],
                        'home_score': orig_pred.get('home_score', 0),
                        'away_score': orig_pred.get('away_score', 0)
                    }
                    analyzer_predictions.append(analyzer_pred)
                
                gp_analysis = analyze_backtest_failures(analyzer_predictions, target_accuracy=100.0)
                gp_analysis_web = format_analysis_for_web(gp_analysis)
            
            return jsonify({
                "success": True,
                "results": formatted_results,
                "gp_analysis": gp_analysis_web if gp_analysis else None,
                "parameters": {
                    "data_file": data_file,
                    "test_week": test_week,
                    "model": "No H2H GP (Pure Form)",
                    "min_confidence": min_confidence,
                    "train_weeks": train_weeks,
                    "min_matches": min_matches
                }
            })
        else:
            return jsonify({"error": "Backtest sonuç üretemedi"}), 400
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Backtest hatası: {e}")
        print(f"❌ Detaylar: {error_details}")
        return jsonify({"error": f"{str(e)} - Detay: {error_details[:500]}"}), 500

def format_backtest_results(results, min_matches=3, min_confidence=0.7):
    """Backtest sonuçlarını formatla"""
    try:
        # Results bir dict, predictions key'ini kullan
        if not isinstance(results, dict) or 'predictions' not in results:
            return {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0,
                'predictions': [],
                'is_perfect': False,
                'error': 'Invalid results format'
            }
        
        predictions = results['predictions']
        
        # Confidence threshold'a göre filtrele
        filtered_predictions = []
        for pred_item in predictions:
            # Her prediction 3 elemanlı tuple: (match_name, pred_data, score_tuple)
            if len(pred_item) >= 2:
                match_name = pred_item[0]
                pred_data = pred_item[1]
                confidence = pred_data.get('confidence', 0.0)
                
                # Confidence 0-1 arasında değilse yüzde formatında olabilir
                if confidence > 1:
                    confidence = confidence / 100.0
                
                if confidence >= min_confidence:
                    filtered_predictions.append(pred_item)
        
        # Temel istatistikler
        total_predictions = len(filtered_predictions)
        correct_predictions = 0
        predictions_list = []
        
        for pred_item in filtered_predictions:
            match_name = pred_item[0]
            pred_data = pred_item[1]
            # Match name'den takım isimlerini ayır
            if ' vs ' in match_name:
                home_team, away_team = match_name.split(' vs ')
            else:
                home_team = away_team = match_name
            
            # Tahmin vs gerçek karşılaştırması
            predicted = pred_data.get('predicted_result', '')
            confidence = pred_data.get('confidence', 0.0)
            
            # Actual result'ı score tuple'dan çıkar
            actual = ''
            if len(pred_item) > 2 and isinstance(pred_item[2], tuple) and len(pred_item[2]) == 2:
                home_score, away_score = pred_item[2]
                if home_score > away_score:
                    actual = '1'
                elif home_score == away_score:
                    actual = 'X'
                else:
                    actual = '2'
            
            is_correct = (predicted == actual)
            
            if is_correct:
                correct_predictions += 1
            
            predictions_list.append({
                'home_team': home_team,
                'away_team': away_team,
                'predicted': predicted,
                'actual': actual,
                'confidence': confidence,
                'is_correct': is_correct
            })
        
        # Sadece minimum maç sayısını karşılayan sonuçları döndür
        if total_predictions >= min_matches:
            accuracy = round((correct_predictions / total_predictions) * 100, 1) if total_predictions > 0 else 0
            
            return {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'predictions': predictions_list,
                'is_perfect': accuracy == 100.0
            }
        else:
            return {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0,
                'predictions': [],
                'is_perfect': False,
                'error': f'Minimum {min_matches} maç gerekli, sadece {total_predictions} maç bulundu'
            }
            
    except Exception as e:
        return {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0,
            'predictions': [],
            'is_perfect': False,
            'error': str(e)
        }

@app.route('/api/clear-results', methods=['POST'])
def clear_results():
    """Sonuçları temizle"""
    try:
        # Backtest geçmişini temizle
        backtest_system.results_history = {}
        
        return jsonify({
            "success": True,
            "message": "Sonuçlar temizlendi"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Template dizinini oluştur
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    print("🎯 Simple Backtest Web Interface")
    print("📊 Hedef: %100 doğru tahmin")
    print("🌐 Tarayıcıda açın: http://localhost:5002")
    
    app.run(debug=True, host='0.0.0.0', port=5002)
