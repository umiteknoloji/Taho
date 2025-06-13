#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest Web Arayüzü
Flask tabanlı web interface
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import sys
from datetime import datetime
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Parametric backtest ve oynanmamış maç sistemlerini import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parametric_backtest import ParametricBacktestSystem
from predict_unplayed_matches import UnplayedMatchPredictor

app = Flask(__name__)
app.secret_key = 'futbol_backtest_2025'

# Global sistemler - her istek için yeni instance oluşturulacak
# backtest_system = ParametricBacktestSystem()  # Kaldırıldı
unplayed_predictor = UnplayedMatchPredictor()

# Desteklenen veri dosyaları
SUPPORTED_DATA_FILES = {
    'data/ALM_stat.json': '🇩🇪 Alman Ligi (Bundesliga)',
    'data/BEL_stat.json': '🇧🇪 Belçika Ligi',
    'data/BRA_stat.json': '🇧🇷 Brezilya Ligi',
    'data/ENG_stat.json': '🏴 İngiliz Premier Ligi',
    'data/ENG_C_stat.json': '🏴 İngiliz Championship',
    'data/ESP_stat.json': '🇪🇸 İspanya La Liga',
    'data/FRA_stat.json': '🇫🇷 Fransa Ligue 1',
    'data/HOL_stat.json': '🇳🇱 Hollanda Eredivisie',
    'data/HOL_2_stat.json': '🇳🇱 Hollanda 2. Lig',
    'data/POR_stat.json': '🇵🇹 Portekiz Ligi',
    'data/TR_stat.json': '🇹🇷 Türkiye Süper Lig',
    'data/ISVEC_stat.json': '🇸🇪 İsveç Ligi',
    'data/NORVEC_stat.json': '🇳🇴 Norveç Ligi',
    'data/FINLANDIYA_stat.json': '🇫🇮 Finlandiya Ligi',
    'data/JAP_stat.json': '🇯🇵 Japonya J-League',
    'data/GK_stat.json': '🌍 Güney Kore K-League'
}

def validate_data_file(data_file):
    """Veri dosyasının geçerliliğini kontrol et"""
    if data_file not in SUPPORTED_DATA_FILES:
        return False, f"Desteklenmeyen veri dosyası: {data_file}"
    
    if not os.path.exists(data_file):
        return False, f"Veri dosyası bulunamadı: {data_file}"
    
    return True, "OK"

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/api/available-weeks')
def get_available_weeks():
    """Mevcut hafta numaralarını döndür"""
    try:
        data_file = request.args.get('data_file', 'data/ALM_stat.json')
        
        # Veri dosyasını doğrula
        is_valid, message = validate_data_file(data_file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Her istek için yeni instance oluştur (determinizm için)
        backtest_system = ParametricBacktestSystem()
        df = backtest_system.load_data(data_file)
        weeks = backtest_system.get_available_weeks(df)
        # Tüm haftaları int olarak döndür (tip hatası önle)
        weeks = [int(w) for w in weeks]
        
        return jsonify({
            "weeks": weeks,
            "total_matches": len(df),
            "week_range": f"{min(weeks)} - {max(weeks)}",
            "league": SUPPORTED_DATA_FILES.get(data_file, "Bilinmeyen Lig")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-backtest', methods=['POST'])
def run_backtest():
    """Backtest çalıştır"""
    try:
        data = request.get_json()
        test_week = data.get('week')
        train_weeks = data.get('train_weeks')
        data_file = data.get('data_file', 'data/ALM_stat.json')
        
        if not test_week:
            return jsonify({"error": "Hafta numarası gerekli"}), 400
        
        # Veri dosyasını doğrula
        is_valid, message = validate_data_file(data_file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Her istek için yeni instance oluştur (determinizm için)
        backtest_system = ParametricBacktestSystem()
        # Backtest çalıştır
        result = backtest_system.run_backtest(data_file, test_week, train_weeks)

        # Eğer sadece oynanmamış maç tahmini varsa (unplayed_predictions anahtarı)
        if result and 'unplayed_predictions' in result:
            return jsonify({
                "success": True,
                "unplayed_predictions": result['unplayed_predictions'],
                "test_week": test_week,
                "league": SUPPORTED_DATA_FILES.get(data_file, "Bilinmeyen Lig")
            })

        return jsonify({
            "success": True,
            "result": result,
            "summary": result.get('summary', ''),
            "league": SUPPORTED_DATA_FILES.get(data_file, "Bilinmeyen Lig")
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/results-history')
def get_results_history():
    """Backtest geçmişini döndür"""
    try:
        summary = backtest_system.get_results_summary()
        history = backtest_system.results_history
        
        return jsonify({
            "summary": summary,
            "history": history
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Geçmişi temizle"""
    try:
        backtest_system.results_history = {}
        return jsonify({"success": True, "message": "Geçmiş temizlendi"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-chart')
def generate_chart():
    """Performans grafiği oluştur"""
    try:
        if not backtest_system.results_history:
            return jsonify({"error": "Henüz backtest sonucu yok"}), 400
        
        # Veri hazırla
        weeks = []
        accuracies = []
        close_rates = []
        
        for week, result in backtest_system.results_history.items():
            weeks.append(week)
            accuracies.append(result['metrics']['accuracy'] * 100)
            close_rates.append(result['metrics']['close_rate'] * 100)
        
        # Grafik oluştur
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(weeks, accuracies, color='skyblue', alpha=0.7)
        plt.title('Tam Skor Doğruluğu (%)')
        plt.xlabel('Hafta')
        plt.ylabel('Doğruluk (%)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(weeks, close_rates, color='lightgreen', alpha=0.7)
        plt.title('Yakın Tahmin Oranı (%)')
        plt.xlabel('Hafta')
        plt.ylabel('Yakın Tahmin (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Base64'e çevir
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            "chart": f"data:image/png;base64,{img_base64}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/export-results')
def export_results():
    """Sonuçları JSON olarak export et"""
    try:
        if not backtest_system.results_history:
            return jsonify({"error": "Henüz backtest sonucu yok"}), 400
        
        # Export dosyası oluştur
        export_data = {
            "export_time": datetime.now().isoformat(),
            "summary": backtest_system.get_results_summary(),
            "results": backtest_system.results_history
        }
        
        filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Geçici dosya oluştur
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return send_file(filename, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-unplayed', methods=['POST'])
def analyze_unplayed_matches():
    """Oynanmamış maçları analiz et"""
    try:
        data = request.get_json()
        data_file = data.get('data_file', 'data/ALM_stat.json')
        
        # Veri dosyasını doğrula
        is_valid, message = validate_data_file(data_file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        print(f"🔍 Oynanmamış maçlar analiz ediliyor: {SUPPORTED_DATA_FILES.get(data_file, data_file)}")
        
        # Veri yükle ve analiz et
        success = unplayed_predictor.load_and_analyze_data(data_file)
        
        if not success:
            return jsonify({"error": "Veri analizi başarısız"}), 500
        
        # İstatistikler
        played_matches = unplayed_predictor.played_matches
        unplayed_matches = unplayed_predictor.unplayed_matches
        
        stats = {
            "total_matches": len(played_matches) + len(unplayed_matches),
            "played_matches": len(played_matches),
            "unplayed_matches": len(unplayed_matches),
            "analysis_time": datetime.now().isoformat()
        }
        
        # Oynanmamış maçları formatla
        unplayed_list = []
        if len(unplayed_matches) > 0:
            for _, match in unplayed_matches.iterrows():
                unplayed_list.append({
                    "matchday": int(match.get('week', 0)),
                    "home_team": match.get('home', 'N/A'),
                    "away_team": match.get('away', 'N/A'),
                    "date": match.get('date', 'N/A'),
                    "status": "Oynanmamış"
                })
        
        return jsonify({
            "success": True,
            "stats": stats,
            "unplayed_matches": unplayed_list[:20]  # İlk 20 maçı göster
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-unplayed', methods=['POST'])
def predict_unplayed_matches():
    """Oynanmamış maçları tahmin et"""
    try:
        data = request.get_json()
        data_file = data.get('data_file', 'data/ALM_stat.json')
        max_predictions = data.get('max_predictions', 10)
        
        # Veri dosyasını doğrula
        is_valid, message = validate_data_file(data_file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        print(f"🎯 Oynanmamış maç tahminleri yapılıyor: {SUPPORTED_DATA_FILES.get(data_file, data_file)}")
        
        # Veri yükle
        success = unplayed_predictor.load_and_analyze_data(data_file)
        
        if not success:
            return jsonify({"error": "Veri analizi başarısız"}), 500
        
        played_matches = unplayed_predictor.played_matches
        unplayed_matches = unplayed_predictor.unplayed_matches
        
        if len(unplayed_matches) == 0:
            return jsonify({"error": "Oynanmamış maç bulunamadı"}), 404
        
        # Modelleri eğit
        print("🤖 Modeller eğitiliyor...")
        unplayed_predictor.train_models(played_matches)
        
        # Tahminler yap
        predictions = []
        unplayed_sample = unplayed_matches.head(max_predictions)
        
        for idx, match in unplayed_sample.iterrows():
            try:
                # Her maç için tahmin yap
                prediction = unplayed_predictor.predict_match(
                    match['home'], 
                    match['away'],
                    match
                )
                
                predictions.append({
                    "matchday": int(match.get('week', 0)),
                    "home_team": match['home'],
                    "away_team": match['away'],
                    "date": match.get('date', 'N/A'),
                    "predicted_result": prediction.get('predicted_result', 'N/A'),
                    "result_meaning": prediction.get('result_meaning', 'N/A'),
                    "home_win_prob": f"{prediction.get('home_win_prob', 0)*100:.1f}%",
                    "draw_prob": f"{prediction.get('draw_prob', 0)*100:.1f}%",
                    "away_win_prob": f"{prediction.get('away_win_prob', 0)*100:.1f}%",
                    "confidence": f"{prediction.get('confidence', 0)*100:.1f}%"
                })
                
            except Exception as e:
                print(f"❌ Tahmin hatası: {e}")
                predictions.append({
                    "matchday": int(match.get('week', 0)),
                    "home_team": match['home'],
                    "away_team": match['away'],
                    "date": match.get('date', 'N/A'),
                    "predicted_result": "Hata",
                    "result_meaning": "Tahmin Edilemedi",
                    "home_win_prob": "N/A",
                    "draw_prob": "N/A", 
                    "away_win_prob": "N/A",
                    "confidence": "N/A"
                })
        
        return jsonify({
            "success": True,
            "predictions": predictions,
            "total_unplayed": len(unplayed_matches),
            "predicted_count": len(predictions),
            "prediction_time": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-advanced-backtest', methods=['POST'])
def run_advanced_backtest():
    """Gelişmiş parametric backtest çalıştır"""
    try:
        data = request.get_json()
        test_week = data.get('week')
        data_file = data.get('data_file', 'data/ALM_stat.json')
        model_type = data.get('model_type', 'ensemble')
        feature_type = data.get('feature_type', 'enhanced')
        
        if not test_week:
            return jsonify({"error": "Hafta numarası gerekli"}), 400
        
        # Veri dosyasını doğrula
        is_valid, message = validate_data_file(data_file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        print(f"🚀 Gelişmiş backtest başlatılıyor - Hafta {test_week} ({SUPPORTED_DATA_FILES.get(data_file, data_file)})")
        
        # Gelişmiş backtest çalıştır
        from parametric_backtest import ParametricBacktestSystem
        advanced_system = ParametricBacktestSystem()
        
        result = advanced_system.run_parametric_backtest(
            data_file=data_file,
            test_week=test_week,
            model_type=model_type,
            feature_type=feature_type
        )
        
        return jsonify({
            "success": True,
            "result": result,
            "model_type": model_type,
            "feature_type": feature_type,
            "test_week": test_week
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Template dizinini oluştur
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    print("🌐 Web arayüzü başlatılıyor...")
    print("📱 Tarayıcıda açın: http://localhost:5005")
    
    app.run(debug=True, host='0.0.0.0', port=5005)
