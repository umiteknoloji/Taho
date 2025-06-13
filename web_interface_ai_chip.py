#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Chip Gaussian Process Web Interface
M2 ve M3 Neural Engine'leri kullanan Gaussian Process tahmin sistemi için web arayüzü
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import threading
import time
from datetime import datetime
import numpy as np

from ai_chip_predictor import AIChipUnplayedMatchPredictor
from gp_ai_chip_manager import AIChipGaussianProcessor

app = Flask(__name__)

# Global variables
predictor = None
training_status = {"status": "idle", "progress": 0, "message": "Hazır"}
available_leagues = {}

def scan_available_leagues():
    """Mevcut lig dosyalarını tara"""
    global available_leagues
    data_dir = "data"
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('_stat.json'):
                league_code = file.replace('_stat.json', '')
                league_names = {
                    'ALM': 'Almanya Bundesliga',
                    'ENG': 'İngiltere Premier League', 
                    'ESP': 'İspanya La Liga',
                    'FRA': 'Fransa Ligue 1',
                    'ITA': 'İtalya Serie A',
                    'TUR': 'Türkiye Süper Lig',
                    'BRA': 'Brezilya Serie A',
                    'ARG': 'Arjantin Primera',
                    'NED': 'Hollanda Eredivisie',
                    'POR': 'Portekiz Primeira'
                }
                
                available_leagues[league_code] = {
                    'name': league_names.get(league_code, league_code),
                    'file': file,
                    'path': os.path.join(data_dir, file)
                }

# Başlangıçta ligleri tara
scan_available_leagues()

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('ai_chip_index.html')

@app.route('/api/leagues')
def get_leagues():
    """Mevcut ligleri listele"""
    return jsonify(available_leagues)

@app.route('/api/neural-engine-status')
def get_neural_engine_status():
    """Neural Engine durumlarını al"""
    try:
        if predictor:
            stats = predictor.get_neural_engine_status()
        else:
            # Geçici AI chip manager ile durum kontrol et
            temp_manager = AIChipGaussianProcessor()
            m2_status = temp_manager.get_m2_neural_load()
            m3_status = temp_manager.get_m3_neural_load()
            
            stats = {
                'M2_Neural_Engine': {
                    'available': m2_status.get('available', False),
                    'neural_load': m2_status.get('neural_load', 100),
                    'cpu_usage': m2_status.get('cpu_usage', 0),
                    'status': 'Optimal' if m2_status.get('neural_load', 100) < 70 else 'Busy'
                },
                'M3_Neural_Engine': {
                    'available': True,
                    'neural_load': m3_status.get('neural_load', 50),
                    'cpu_usage': m3_status.get('cpu_usage', 0),
                    'status': 'Optimal' if m3_status.get('neural_load', 50) < 70 else 'Busy'
                },
                'training_history': []
            }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Model eğitimi başlat"""
    global predictor, training_status
    
    try:
        data = request.get_json()
        league_code = data.get('league')
        
        if league_code not in available_leagues:
            return jsonify({'error': 'Geçersiz lig kodu'}), 400
        
        # Training status güncelle
        training_status = {"status": "starting", "progress": 0, "message": "Eğitim başlatılıyor..."}
        
        def train_worker():
            global predictor, training_status
            
            try:
                training_status = {"status": "loading", "progress": 10, "message": "Veri yükleniyor..."}
                
                # Predictor oluştur
                predictor = AIChipUnplayedMatchPredictor()
                
                # Veri yükle
                league_file = available_leagues[league_code]['path']
                if not predictor.load_and_analyze_data(league_file):
                    training_status = {"status": "error", "progress": 0, "message": "Veri yükleme başarısız"}
                    return
                
                training_status = {"status": "training", "progress": 50, "message": "AI Chip'lerde Gaussian Process eğitimi..."}
                
                # Model eğit
                if predictor.train_models():
                    training_status = {"status": "completed", "progress": 100, "message": "Eğitim tamamlandı"}
                else:
                    training_status = {"status": "error", "progress": 0, "message": "Model eğitimi başarısız"}
                    
            except Exception as e:
                training_status = {"status": "error", "progress": 0, "message": f"Hata: {str(e)}"}
        
        # Background thread'de eğit
        training_thread = threading.Thread(target=train_worker)
        training_thread.start()
        
        return jsonify({'message': 'Eğitim başlatıldı', 'status': 'started'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-status')
def get_training_status():
    """Eğitim durumunu al"""
    return jsonify(training_status)

@app.route('/api/predict-unplayed', methods=['POST'])
def predict_unplayed_matches():
    """Oynanmamış maçları tahmin et"""
    global predictor
    
    try:
        if not predictor:
            return jsonify({'error': 'Model henüz eğitilmedi'}), 400
        
        # Tahminleri yap
        predictions = predictor.predict_unplayed_matches()
        
        # Sonuçları formatla
        formatted_predictions = []
        for pred in predictions:
            formatted_predictions.append({
                'home_team': pred.get('home_team', 'N/A'),
                'away_team': pred.get('away_team', 'N/A'),
                'predicted_result': pred.get('predicted_result', 'X'),
                'result_meaning': pred.get('result_meaning', 'Bilinmeyen'),
                'home_win_prob': round(pred.get('home_win_prob', 0), 1),
                'draw_prob': round(pred.get('draw_prob', 0), 1),
                'away_win_prob': round(pred.get('away_win_prob', 0), 1),
                'confidence': round(pred.get('confidence', 0), 1),
                'method': pred.get('method', 'GP_Neural_Engine')
            })
        
        return jsonify({
            'success': True,
            'predictions': formatted_predictions,
            'total_matches': len(formatted_predictions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-single', methods=['POST'])
def predict_single_match():
    """Tek maç tahmini"""
    global predictor
    
    try:
        if not predictor:
            return jsonify({'error': 'Model henüz eğitilmedi'}), 400
        
        data = request.get_json()
        home_team = data.get('home_team', '').strip()
        away_team = data.get('away_team', '').strip()
        week = data.get('week')
        
        if not home_team or not away_team:
            return jsonify({'error': 'Takım isimleri gerekli'}), 400
        
        # Tahmin yap
        prediction = predictor.predict_single_match(home_team, away_team, week)
        
        return jsonify({
            'success': True,
            'prediction': {
                'home_team': prediction.get('home_team', home_team),
                'away_team': prediction.get('away_team', away_team),
                'predicted_result': prediction.get('predicted_result', 'X'),
                'result_meaning': prediction.get('result_meaning', 'Bilinmeyen'),
                'home_win_prob': round(prediction.get('home_win_prob', 0), 1),
                'draw_prob': round(prediction.get('draw_prob', 0), 1),
                'away_win_prob': round(prediction.get('away_win_prob', 0), 1),
                'confidence': round(prediction.get('confidence', 0), 1),
                'method': prediction.get('method', 'GP_Neural_Engine')
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-performance')
def get_system_performance():
    """Sistem performansını al"""
    try:
        if not predictor:
            return jsonify({'error': 'Model henüz yüklenmedi'}), 400
        
        # Neural Engine stats
        neural_stats = predictor.get_neural_engine_status()
        
        # Training history
        training_history = neural_stats.get('training_history', [])
        
        # Performance metrics
        performance = {
            'neural_engines': {
                'M2': neural_stats.get('M2_Neural_Engine', {}),
                'M3': neural_stats.get('M3_Neural_Engine', {})
            },
            'training_history': training_history,
            'model_info': {
                'algorithm': 'Gaussian Process Classification',
                'kernels': 'RBF, Matern32, Matern52, Combined',
                'optimization': 'Neural Engine Optimized',
                'distributed': True
            }
        }
        
        return jsonify(performance)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset-system', methods=['POST'])
def reset_system():
    """Sistemi sıfırla"""
    global predictor, training_status
    
    try:
        predictor = None
        training_status = {"status": "idle", "progress": 0, "message": "Hazır"}
        
        return jsonify({'success': True, 'message': 'Sistem sıfırlandı'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🧠 AI Chip Gaussian Process Web Interface başlatılıyor...")
    print("⚡ M2 ve M3 Neural Engine'leri destekleniyor")
    print("🌐 Web arayüzü: http://localhost:5025")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5025,
        debug=False,
        threaded=True
    )
