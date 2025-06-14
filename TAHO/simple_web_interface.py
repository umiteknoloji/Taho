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

# Çalışma dizinini ayarla
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Backtest sistemini import et
sys.path.append(current_dir)
from parametric_backtest import ParametricBacktestSystem
from gp_improvement_analyzer import analyze_backtest_failures, format_analysis_for_web
from perfect_accuracy_gp import PerfectAccuracyGP
from no_h2h_gp import NoH2HGP
from perfect_prediction_system import PerfectPredictor

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
    'data/ALM_stat_test.json': '🧪 Test Bundesliga (Oynanmamış)',
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
        
        # PerfectPredictor modeli kullan - doğru parametrelerle
        predictor = PerfectPredictor(data_file)  # data_file'ı constructor'a ver
        predictions = predictor.predict_week(test_week)  # Sadece week parametresi
        
        # Sonuçları uygun formata çevir
        results = {
            'predictions': []
        }
        
        for pred in predictions:
            # Gerçek sonuçları bul
            match_name = f"{pred['home_team']} vs {pred['away_team']}"
            home_score, away_score, actual_result = _get_actual_result(data_file, test_week, pred['home_team'], pred['away_team'])
            
            # Oynanmamış maçlar için farklı işlem
            if actual_result is None:
                # Oynanmamış maç - sadece tahmin göster
                pred_data = {
                    'predicted_result': pred['prediction'],
                    'confidence': pred['confidence'],
                    'probabilities': pred.get('probabilities', {'1': 0.33, 'X': 0.33, '2': 0.33}),
                    'actual_result': 'Oynanmamış',
                    'is_correct': None  # Değerlendirilemez
                }
                score_tuple = (None, None)
            else:
                # Oynanmış maç - normal karşılaştırma
                pred_data = {
                    'predicted_result': pred['prediction'],
                    'confidence': pred['confidence'],
                    'probabilities': pred.get('probabilities', {'1': 0.33, 'X': 0.33, '2': 0.33}),
                    'actual_result': actual_result,
                    'is_correct': pred['prediction'] == actual_result
                }
                score_tuple = (home_score, away_score)
            
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
                    "model": "PerfectPredictor (Best System)",
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
            
            # Oynanmamış maç kontrolü
            actual_result = pred_data.get('actual_result')
            if actual_result == 'Oynanmamış':
                # Oynanmamış maç - sadece tahmin göster
                actual = 'Oynanmamış'
                is_correct = None  # Değerlendirilemez
            else:
                # Oynanmış maç - score tuple'dan gerçek sonucu çıkar
                actual = actual_result or ''
                if len(pred_item) > 2 and isinstance(pred_item[2], tuple) and len(pred_item[2]) == 2:
                    home_score, away_score = pred_item[2]
                    if home_score is not None and away_score is not None:
                        if home_score > away_score:
                            actual = '1'
                        elif home_score == away_score:
                            actual = 'X'
                        else:
                            actual = '2'
                
                is_correct = (predicted == actual) if actual != 'Oynanmamış' else None
            
            # Sadece oynanmış maçları doğruluk hesaplamasına dahil et
            if is_correct is not None and is_correct:
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
            # Doğruluk hesaplamasını sadece oynanmış maçlara göre yap
            played_matches = [p for p in predictions_list if p['is_correct'] is not None]
            played_count = len(played_matches)
            
            if played_count > 0:
                accuracy = round((correct_predictions / played_count) * 100, 1)
            else:
                accuracy = 0  # Hiç oynanmış maç yok
            
            return {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'played_matches': played_count,
                'accuracy': accuracy,
                'predictions': predictions_list,
                'is_perfect': accuracy == 100.0 if played_count > 0 else False
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

def perfect_predict_match(home_team, away_team, target_week):
    """Perfect prediction algoritması"""
    try:
        # Veri yükle
        with open('data/ALM_stat.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Takım form analizi
        home_form = get_team_form(home_team, data, target_week)
        away_form = get_team_form(away_team, data, target_week)
        
        # Perfect scoring system
        home_score = 0
        away_score = 0
        draw_score = 0
        
        # 1. Ev sahibi avantajı (30%)
        home_advantage = calculate_home_advantage(home_team, data, target_week)
        home_score += home_advantage * 30
        
        # 2. Form analizi (25%)
        home_form_score = (home_form['win_ratio'] * 0.6 + home_form['points_per_game'] / 3 * 0.4) * 25
        away_form_score = (away_form['win_ratio'] * 0.6 + away_form['points_per_game'] / 3 * 0.4) * 25
        
        home_score += home_form_score
        away_score += away_form_score
        
        # 3. Gol farkı analizi (20%)
        home_goal_diff = (home_form['goals_for_avg'] - home_form['goals_against_avg']) * 10
        away_goal_diff = (away_form['goals_for_avg'] - away_form['goals_against_avg']) * 10
        
        home_score += max(0, home_goal_diff)
        away_score += max(0, away_goal_diff)
        
        # 4. Momentum analizi (15%)
        if home_form['recent_trend'] == 'IMPROVING':
            home_score += 15
        elif home_form['recent_trend'] == 'DECLINING':
            home_score -= 10
            
        if away_form['recent_trend'] == 'IMPROVING':
            away_score += 15
        elif away_form['recent_trend'] == 'DECLINING':
            away_score -= 10
        
        # 5. Beraberlik analizi (10%)
        score_diff = abs(home_score - away_score)
        if score_diff < 10:
            draw_score += 20
        
        if home_form['draw_ratio'] > 0.3 or away_form['draw_ratio'] > 0.3:
            draw_score += 15
            
        # Özel durumlar
        big_teams = ['Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen']
        if home_team in big_teams and away_team in big_teams:
            draw_score += 10  # Büyük maç beraberlik eğilimi
        
        # Sonuç belirleme
        max_score = max(home_score, away_score, draw_score)
        
        if max_score == draw_score and draw_score > 25:
            prediction = 'X'
            confidence = min(100, draw_score)
        elif home_score > away_score:
            prediction = '1'
            confidence = min(100, home_score)
        else:
            prediction = '2'
            confidence = min(100, away_score)
        
        return {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'home_score': round(home_score, 1),
            'away_score': round(away_score, 1),
            'draw_score': round(draw_score, 1),
            'algorithm': 'Perfect Prediction System'
        }
        
    except Exception as e:
        return {
            'prediction': '1',
            'confidence': 50.0,
            'error': str(e),
            'algorithm': 'Fallback'
        }

def get_team_form(team_name, data, target_week, num_matches=5):
    """Takım form analizi"""
    team_matches = []
    
    for match in data:
        week = match.get('week')
        if week is None or week >= target_week:
            continue
            
        home_team = match.get('home')
        away_team = match.get('away')
        
        if team_name in [home_team, away_team]:
            team_matches.append({
                'week': week,
                'match': match,
                'is_home': team_name == home_team
            })
    
    # Haftaya göre sırala
    team_matches.sort(key=lambda x: x['week'], reverse=True)
    recent_matches = team_matches[:num_matches]
    
    if not recent_matches:
        return {
            'form': '',
            'goals_for_avg': 0.0,
            'goals_against_avg': 0.0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'win_ratio': 0.0,
            'draw_ratio': 0.0,
            'points_per_game': 0.0,
            'recent_trend': 'STABLE'
        }
    
    form = []
    total_goals_for = 0
    total_goals_against = 0
    wins = draws = losses = 0
    
    for match_info in recent_matches:
        match = match_info['match']
        is_home = match_info['is_home']
        
        score_data = match.get('score', {})
        if not score_data or 'fullTime' not in score_data:
            continue
            
        h_goals = int(score_data['fullTime']['home'])
        a_goals = int(score_data['fullTime']['away'])
        
        if is_home:
            goals_for = h_goals
            goals_against = a_goals
        else:
            goals_for = a_goals
            goals_against = h_goals
        
        total_goals_for += goals_for
        total_goals_against += goals_against
        
        if goals_for > goals_against:
            form.append('W')
            wins += 1
        elif goals_for < goals_against:
            form.append('L')
            losses += 1
        else:
            form.append('D')
            draws += 1
    
    num_matches = len(recent_matches)
    if num_matches == 0:
        return {
            'form': '',
            'goals_for_avg': 0.0,
            'goals_against_avg': 0.0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'win_ratio': 0.0,
            'draw_ratio': 0.0,
            'points_per_game': 0.0,
            'recent_trend': 'STABLE'
        }
    
    # Trend analizi
    recent_trend = 'STABLE'
    if len(form) >= 3:
        recent_form = form[:3]
        win_count = recent_form.count('W')
        loss_count = recent_form.count('L')
        
        if win_count >= 2:
            recent_trend = 'IMPROVING'
        elif loss_count >= 2:
            recent_trend = 'DECLINING'
    
    return {
        'form': ''.join(form),
        'goals_for_avg': total_goals_for / num_matches,
        'goals_against_avg': total_goals_against / num_matches,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_ratio': wins / num_matches,
        'draw_ratio': draws / num_matches,
        'points_per_game': (wins * 3 + draws) / num_matches,
        'recent_trend': recent_trend
    }

def calculate_home_advantage(home_team, data, target_week):
    """Ev sahibi avantajını hesapla"""
    home_matches = []
    
    for match in data:
        week = match.get('week')
        if week is None or week >= target_week:
            continue
            
        if match.get('home') == home_team:
            score_data = match.get('score', {})
            if score_data and 'fullTime' in score_data:
                h_goals = int(score_data['fullTime']['home'])
                a_goals = int(score_data['fullTime']['away'])
                
                if h_goals > a_goals:
                    result = 'W'
                elif h_goals < a_goals:
                    result = 'L'
                else:
                    result = 'D'
                
                home_matches.append(result)
    
    if not home_matches:
        return 0.5
    
    wins = home_matches.count('W')
    total = len(home_matches)
    
    return wins / total

@app.route('/perfect_predict', methods=['POST'])
def perfect_predict():
    """Perfect prediction endpoint"""
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        week = data.get('week', 13)  # Default next week
        
        if not home_team or not away_team:
            return jsonify({'error': 'home_team and away_team required'}), 400
        
        prediction = perfect_predict_match(home_team, away_team, week)
        
        return jsonify({
            'match': f"{home_team} vs {away_team}",
            'week': week,
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'details': {
                'home_score': prediction['home_score'],
                'away_score': prediction['away_score'],
                'draw_score': prediction['draw_score']
            },
            'algorithm': prediction['algorithm'],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/perfect_week', methods=['POST'])
def perfect_week():
    """Bir hafta için perfect predictions"""
    try:
        data = request.get_json()
        week = data.get('week', 13)
        
        # Veri yükle
        with open('data/ALM_stat.json', 'r', encoding='utf-8') as f:
            matches_data = json.load(f)
        
        week_matches = [m for m in matches_data if m.get('week') == week]
        predictions = []
        
        for match in week_matches:
            home_team = match.get('home')
            away_team = match.get('away')
            
            if home_team and away_team:
                prediction = perfect_predict_match(home_team, away_team, week)
                predictions.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence']
                })
        
        return jsonify({
            'week': week,
            'predictions': predictions,
            'total_matches': len(predictions),
            'algorithm': 'Perfect Prediction System',
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _get_actual_result(data_file, week, home_team, away_team):
    """Gerçek maç sonucunu bul"""
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for match in data:
            if (match.get('week') == week and 
                match.get('home') == home_team and 
                match.get('away') == away_team):
                
                score = match.get('score', {}).get('fullTime', {})
                
                # Eğer skor bilgisi yoksa oynanmamış maç
                if not score or score.get('home') is None or score.get('away') is None:
                    return None, None, None  # Oynanmamış maç
                
                home_score = int(score.get('home', 0))
                away_score = int(score.get('away', 0))
                
                if home_score > away_score:
                    result = '1'
                elif home_score < away_score:
                    result = '2'
                else:
                    result = 'X'
                
                return home_score, away_score, result
        
        return None, None, None  # Maç bulunamadı veya oynanmamış
    except:
        return None, None, None

if __name__ == '__main__':
    # Template dizinini oluştur
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    print("🎯 Simple Backtest Web Interface")
    print("📊 Hedef: %100 doğru tahmin")
    print("🌐 Tarayıcıda açın: http://localhost:5002")
    
    app.run(debug=True, host='0.0.0.0', port=5002)
