"""
Evrensel Takım Analizi Modülü
Bu modül farklı lig formatlarındaki verilerden takımların oyun stilini ve performansını analiz eder.

Özellikler:
- Evrensel veri formatı desteği (DataNormalizer kullanarak)
- Gelişmiş takım profilleme
- Oyun stili kümeleme analizi
- Performans karşılaştırmaları
- Takım güçlü/zayıf yönlerini belirleme
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import json
import logging
from core.data_normalizer import DataNormalizer

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeamAnalysisModel:
    """
    Gelişmiş Takım Analizi Modeli
    
    Bu sınıf takımların oyun stillerini, güçlü/zayıf yönlerini ve performanslarını
    evrensel veri formatında analiz eder.
    """
    
    def __init__(self, n_clusters=5, clustering_method='agglomerative'):
        """
        Model parametrelerini başlat
        
        Args:
            n_clusters: Küme sayısı
            clustering_method: Kümeleme yöntemi ('agglomerative', 'kmeans')
        """
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.normalizer = DataNormalizer()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # %95 varyans koruma
        
        # Model bileşenleri
        self.cluster_model = None
        self.team_profiles = {}
        self.cluster_descriptions = {}
        self.feature_names = []
        self.is_trained = False
        
        # Oyun stili kategorileri
        self.playing_style_features = {
            'offensive': ['total_shots', 'shots_on_target', 'corners', 'crosses', 'attack_intensity'],
            'defensive': ['fouls', 'yellow_cards', 'defensive_actions', 'blocks'],
            'possession': ['possession', 'successful_passes', 'pass_accuracy', 'possession_dominance'],
            'physical': ['fouls', 'yellow_cards', 'red_cards', 'aggressive_play'],
            'technical': ['pass_accuracy', 'shot_accuracy', 'cross_accuracy', 'technical_skill']
        }
    
    def _prepare_team_features(self, normalized_data: List[Dict]) -> Tuple[Dict, List]:
        """
        Takım bazında özellikleri hazırla
        
        Args:
            normalized_data: Normalleştirilmiş maç verileri
            
        Returns:
            team_stats: Takım istatistikleri
            feature_names: Özellik isimleri
        """
        team_stats = {}
        
        for match in normalized_data:
            home_team = match['home']
            away_team = match['away']
            
            # Her takım için ev sahibi ve deplasman istatistiklerini topla
            for team, location in [(home_team, 'home'), (away_team, 'away')]:
                if team not in team_stats:
                    team_stats[team] = {'home': [], 'away': [], 'all': []}
                
                # Temel istatistikler
                match_stats = []
                
                # Skor ve performans
                goals_for = match[f'{location}_score']
                goals_against = match[f'{"away" if location == "home" else "home"}_score']
                match_stats.extend([goals_for, goals_against])
                
                # Oyun istatistikleri
                stats_fields = [
                    f'possession_{location}', f'total_shots_{location}', 
                    f'shots_on_target_{location}', f'corners_{location}',
                    f'fouls_{location}', f'offsides_{location}'
                ]
                
                # Sarı/kırmızı kartlar için güvenli erişim
                yellow_cards = match.get(f'yellow_cards_{location}', 0)
                red_cards = match.get(f'red_cards_{location}', 0)
                
                for field in stats_fields:
                    value = match.get(field, 0)
                    if isinstance(value, str) and value.replace('.', '').isdigit():
                        value = float(value)
                    elif not isinstance(value, (int, float)):
                        value = 0
                    match_stats.append(value)
                
                # Kart istatistikleri
                match_stats.extend([yellow_cards, red_cards])
                
                # Türetilmiş özellikler
                possession = match.get(f'possession_{location}', 0.5)
                total_shots = match.get(f'total_shots_{location}', 0)
                shots_on_target = match.get(f'shots_on_target_{location}', 0)
                
                # Skor farkı
                goal_difference = goals_for - goals_against
                match_stats.append(goal_difference)
                
                # Şut doğruluğu
                shot_accuracy = (shots_on_target / total_shots * 100) if total_shots > 0 else 0
                match_stats.append(shot_accuracy)
                
                # Hücum yoğunluğu
                attack_intensity = total_shots + match.get(f'corners_{location}', 0)
                match_stats.append(attack_intensity)
                
                # Hâkimiyet (possession 0-1 arası olduğu için 0.5'ten farkı alıyoruz)
                possession_dominance = (possession - 0.5) * 100  # -50 ile +50 arası
                match_stats.append(possession_dominance)
                
                team_stats[team][location].append(match_stats)
                team_stats[team]['all'].append(match_stats)
        
        # Özellik isimleri
        feature_names = [
            'goals_for', 'goals_against', 'possession', 'total_shots',
            'shots_on_target', 'corners', 'fouls', 'offsides',
            'yellow_cards', 'red_cards', 'goal_difference', 
            'shot_accuracy', 'attack_intensity', 'possession_dominance'
        ]
        
        return team_stats, feature_names
    
    def _calculate_team_averages(self, team_stats: Dict) -> Dict:
        """
        Takım ortalama istatistiklerini hesapla
        """
        team_averages = {}
        
        for team, stats in team_stats.items():
            team_averages[team] = {}
            
            for location in ['home', 'away', 'all']:
                if stats[location]:
                    avg_stats = np.mean(stats[location], axis=0)
                    team_averages[team][location] = avg_stats
                else:
                    team_averages[team][location] = np.zeros(len(self.feature_names))
        
        return team_averages
    
    def _create_feature_matrix(self, team_averages: Dict) -> Tuple[np.ndarray, List[str]]:
        """
        Kümeleme için özellik matrisi oluştur
        """
        teams = list(team_averages.keys())
        feature_matrix = []
        
        for team in teams:
            # Genel performans istatistiklerini kullan
            team_features = team_averages[team]['all']
            feature_matrix.append(team_features)
        
        return np.array(feature_matrix), teams
    
    def _perform_clustering(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Kümeleme analizi gerçekleştir
        """
        # Veriyi ölçeklendir
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # PCA uygula (isteğe bağlı)
        if scaled_features.shape[1] > 10:
            scaled_features = self.pca.fit_transform(scaled_features)
        
        # Kümeleme yöntemi seç
        if self.clustering_method == 'kmeans':
            self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        else:
            self.cluster_model = AgglomerativeClustering(n_clusters=self.n_clusters)
        
        # Kümeleme uygula
        cluster_labels = self.cluster_model.fit_predict(scaled_features)
        
        # Silhouette skorunu hesapla
        if len(np.unique(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            logger.info(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return cluster_labels
    
    def _analyze_clusters(self, feature_matrix: np.ndarray, cluster_labels: np.ndarray, teams: List[str]):
        """
        Küme özelliklerini analiz et
        """
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_teams = [teams[i] for i in range(len(teams)) if cluster_mask[i]]
            
            if len(cluster_teams) == 0:
                continue
            
            # Küme ortalaması
            cluster_features = feature_matrix[cluster_mask]
            cluster_mean = np.mean(cluster_features, axis=0)
            
            # Küme özelliklerini belirle
            top_features_idx = np.argsort(cluster_mean)[-3:][::-1]
            top_features = [(self.feature_names[i], cluster_mean[i]) for i in top_features_idx]
            
            # Oyun stili belirleme
            playing_style = self._determine_playing_style(cluster_mean)
            
            self.cluster_descriptions[cluster_id] = {
                'teams': cluster_teams,
                'team_count': len(cluster_teams),
                'top_features': top_features,
                'playing_style': playing_style,
                'avg_stats': cluster_mean
            }
    
    def _determine_playing_style(self, features: np.ndarray) -> str:
        """
        Özellik vektörüne göre oyun stilini belirle
        """
        feature_dict = dict(zip(self.feature_names, features))
        
        # Farklı oyun stillerinin skorlarını hesapla
        style_scores = {}
        
        # Hücum odaklı
        if feature_dict.get('total_shots', 0) > 10 and feature_dict.get('attack_intensity', 0) > 15:
            style_scores['Hücum Odaklı'] = feature_dict.get('total_shots', 0) + feature_dict.get('attack_intensity', 0)
        
        # Defans odaklı
        if feature_dict.get('goals_against', 0) < 1.2 and feature_dict.get('fouls', 0) > 12:
            style_scores['Defans Odaklı'] = 20 - feature_dict.get('goals_against', 0) + feature_dict.get('fouls', 0) / 2
        
        # Top hakimiyeti
        if feature_dict.get('possession', 0) > 52:
            style_scores['Topa Sahip Olma'] = feature_dict.get('possession', 0)
        
        # Dengeli oyun
        possession = feature_dict.get('possession', 50)
        if 48 <= possession <= 52:
            style_scores['Dengeli Oyun'] = 25 - abs(possession - 50)
        
        # Fiziksel oyun
        if feature_dict.get('fouls', 0) > 15 or feature_dict.get('yellow_cards', 0) > 2:
            style_scores['Fiziksel Oyun'] = feature_dict.get('fouls', 0) + feature_dict.get('yellow_cards', 0) * 3
        
        return max(style_scores, key=style_scores.get) if style_scores else 'Belirsiz'
    
    def train(self, data_path: str):
        """
        Veri dosyasından model eğit
        
        Args:
            data_path: JSON veri dosya yolu
        """
        try:
            logger.info(f"Takım analizi eğitimi başlatılıyor: {data_path}")
            
            # Veriyi normalize et
            normalized_df = self.normalizer.normalize_dataset(data_path)
            normalized_data = normalized_df.to_dict('records')
            logger.info(f"{len(normalized_data)} maç verisi normalleştirildi")
            
            # Takım özelliklerini hazırla
            team_stats, self.feature_names = self._prepare_team_features(normalized_data)
            logger.info(f"{len(team_stats)} takım analiz edildi")
            
            # Takım ortalamalarını hesapla
            team_averages = self._calculate_team_averages(team_stats)
            
            # Özellik matrisini oluştur
            feature_matrix, teams = self._create_feature_matrix(team_averages)
            
            # Kümeleme analizi yap
            cluster_labels = self._perform_clustering(feature_matrix)
            
            # Takım profillerini oluştur
            for i, team in enumerate(teams):
                self.team_profiles[team] = {
                    'cluster_label': int(cluster_labels[i]),
                    'avg_stats': team_averages[team],
                    'feature_values': feature_matrix[i],
                    'playing_style': None  # Sonra belirlenecek
                }
            
            # Küme analizini yap
            self._analyze_clusters(feature_matrix, cluster_labels, teams)
            
            # Her takım için oyun stilini belirle
            for team in self.team_profiles:
                cluster_id = self.team_profiles[team]['cluster_label']
                if cluster_id in self.cluster_descriptions:
                    self.team_profiles[team]['playing_style'] = self.cluster_descriptions[cluster_id]['playing_style']
            
            self.is_trained = True
            logger.info("Takım analizi eğitimi tamamlandı")
            
        except Exception as e:
            logger.error(f"Eğitim hatası: {str(e)}")
            raise
    
    def train_from_data(self, normalized_data: List[Dict]):
        """
        Normalleştirilmiş veriden model eğit
        
        Args:
            normalized_data: Normalleştirilmiş maç verileri listesi
        """
        try:
            logger.info(f"Takım analizi eğitimi başlatılıyor: {len(normalized_data)} maç")
            
            # Takım özelliklerini hazırla
            team_stats, self.feature_names = self._prepare_team_features(normalized_data)
            logger.info(f"{len(team_stats)} takım analiz edildi")
            
            # Takım ortalamalarını hesapla
            team_averages = self._calculate_team_averages(team_stats)
            
            # Özellik matrisini oluştur
            feature_matrix, teams = self._create_feature_matrix(team_averages)
            
            # Kümeleme analizi yap
            cluster_labels = self._perform_clustering(feature_matrix)
            
            # Takım profillerini oluştur
            for i, team in enumerate(teams):
                self.team_profiles[team] = {
                    'cluster_label': int(cluster_labels[i]),
                    'avg_stats': team_averages[team],
                    'feature_values': feature_matrix[i],
                    'playing_style': None
                }
            
            # Küme analizini yap
            self._analyze_clusters(feature_matrix, cluster_labels, teams)
            
            # Her takım için oyun stilini belirle
            for team in self.team_profiles:
                cluster_id = self.team_profiles[team]['cluster_label']
                if cluster_id in self.cluster_descriptions:
                    self.team_profiles[team]['playing_style'] = self.cluster_descriptions[cluster_id]['playing_style']
            
            self.is_trained = True
            logger.info("Takım analizi eğitimi tamamlandı")
            
        except Exception as e:
            logger.error(f"Eğitim hatası: {str(e)}")
            raise
    
    def get_team_profile(self, team: str) -> Dict[str, Any]:
        """
        Takım profilini getir
        
        Args:
            team: Takım adı
            
        Returns:
            Detaylı takım profili
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi. Önce train() metodunu çağırın.")
        
        if team not in self.team_profiles:
            available_teams = list(self.team_profiles.keys())
            raise ValueError(f"Takım bulunamadı: {team}. Mevcut takımlar: {available_teams[:5]}...")
        
        profile = self.team_profiles[team].copy()
        cluster_id = profile['cluster_label']
        
        # Detaylı analiz ekle
        avg_stats = profile['avg_stats']['all']
        feature_dict = dict(zip(self.feature_names, avg_stats))
        
        # Güçlü yönleri belirle
        strengths = []
        weaknesses = []
        
        # Gol atma
        if feature_dict.get('goals_for', 0) > 1.5:
            strengths.append("Etkili Gol Atma")
        elif feature_dict.get('goals_for', 0) < 1.0:
            weaknesses.append("Gol Atma Zorluğu")
        
        # Defans
        if feature_dict.get('goals_against', 0) < 1.0:
            strengths.append("Güçlü Defans")
        elif feature_dict.get('goals_against', 0) > 1.8:
            weaknesses.append("Zayıf Defans")
        
        # Şut doğruluğu
        if feature_dict.get('shot_accuracy', 0) > 35:
            strengths.append("Yüksek Şut Doğruluğu")
        elif feature_dict.get('shot_accuracy', 0) < 25:
            weaknesses.append("Düşük Şut Doğruluğu")
        
        # Top hakimiyeti
        if feature_dict.get('possession', 0) > 55:
            strengths.append("Top Hakimiyeti")
        elif feature_dict.get('possession', 0) < 45:
            weaknesses.append("Top Hakimiyeti Eksikliği")
        
        # Disiplin
        if feature_dict.get('fouls', 0) < 10:
            strengths.append("Disiplinli Oyun")
        elif feature_dict.get('fouls', 0) > 16:
            weaknesses.append("Disiplinsiz Oyun")
        
        profile.update({
            'detailed_stats': feature_dict,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'cluster_info': self.cluster_descriptions.get(cluster_id, {}),
            'team_summary': self._generate_team_summary(feature_dict, strengths, weaknesses)
        })
        
        return profile
    
    def _generate_team_summary(self, stats: Dict, strengths: List[str], weaknesses: List[str]) -> str:
        """
        Takım için özet açıklama oluştur
        """
        summary_parts = []
        
        # Oyun stili
        possession = stats.get('possession', 50)
        goals_per_game = stats.get('goals_for', 0)
        
        if possession > 55:
            summary_parts.append("topa sahip olmayı seven")
        elif possession < 45:
            summary_parts.append("direkt oyunu tercih eden")
        else:
            summary_parts.append("dengeli")
        
        if goals_per_game > 2:
            summary_parts.append("ve hücum odaklı")
        elif goals_per_game < 1:
            summary_parts.append("ve defans odaklı")
        
        summary = f"Bu takım {' '.join(summary_parts)} bir oyun tarzına sahip."
        
        if strengths:
            summary += f" Güçlü yönleri: {', '.join(strengths[:2])}."
        
        return summary
    
    def compare_teams(self, team1: str, team2: str) -> Dict[str, Any]:
        """
        İki takımı karşılaştır
        
        Args:
            team1: İlk takım
            team2: İkinci takım
            
        Returns:
            Karşılaştırma raporu
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi.")
        
        if team1 not in self.team_profiles or team2 not in self.team_profiles:
            raise ValueError("Takımlardan biri bulunamadı.")
        
        profile1 = self.get_team_profile(team1)
        profile2 = self.get_team_profile(team2)
        
        stats1 = profile1['detailed_stats']
        stats2 = profile2['detailed_stats']
        
        comparison = {
            'team1': team1,
            'team2': team2,
            'team1_advantages': [],
            'team2_advantages': [],
            'key_differences': {},
            'prediction_factors': []
        }
        
        # Karşılaştırma kategorileri
        categories = {
            'Gol Atma': 'goals_for',
            'Defans': 'goals_against',  # Ters çevirilecek
            'Top Hakimiyeti': 'possession',
            'Şut Doğruluğu': 'shot_accuracy',
            'Hücum Yoğunluğu': 'attack_intensity'
        }
        
        for category, stat_key in categories.items():
            val1 = stats1.get(stat_key, 0)
            val2 = stats2.get(stat_key, 0)
            
            if stat_key == 'goals_against':  # Defans için düşük değer iyi
                if val1 < val2:
                    comparison['team1_advantages'].append(category)
                elif val2 < val1:
                    comparison['team2_advantages'].append(category)
            else:  # Diğerleri için yüksek değer iyi
                if val1 > val2:
                    comparison['team1_advantages'].append(category)
                elif val2 > val1:
                    comparison['team2_advantages'].append(category)
            
            comparison['key_differences'][category] = {
                team1: round(val1, 2),
                team2: round(val2, 2),
                'difference': round(abs(val1 - val2), 2)
            }
        
        # Tahmin faktörleri
        if len(comparison['team1_advantages']) > len(comparison['team2_advantages']):
            comparison['prediction_factors'].append(f"{team1} daha avantajlı görünüyor")
        elif len(comparison['team2_advantages']) > len(comparison['team1_advantages']):
            comparison['prediction_factors'].append(f"{team2} daha avantajlı görünüyor")
        else:
            comparison['prediction_factors'].append("Takımlar dengeli görünüyor")
        
        return comparison
    
    def get_cluster_analysis(self) -> Dict[str, Any]:
        """
        Küme analizi sonuçlarını getir
        
        Returns:
            Küme analizi raporu
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi.")
        
        analysis = {
            'total_clusters': self.n_clusters,
            'clustering_method': self.clustering_method,
            'clusters': self.cluster_descriptions,
            'summary': {}
        }
        
        # Küme özeti
        for cluster_id, desc in self.cluster_descriptions.items():
            analysis['summary'][f'Küme_{cluster_id}'] = {
                'takım_sayısı': desc['team_count'],
                'oyun_stili': desc['playing_style'],
                'öne_çıkan_özellik': desc['top_features'][0][0] if desc['top_features'] else 'Belirsiz'
            }
        
        return analysis
    
    def get_league_overview(self) -> Dict[str, Any]:
        """
        Lig genel görünümü
        
        Returns:
            Lig analizi
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi.")
        
        all_stats = []
        playing_styles = []
        
        for team_profile in self.team_profiles.values():
            all_stats.append(team_profile['feature_values'])
            playing_styles.append(team_profile.get('playing_style', 'Belirsiz'))
        
        stats_array = np.array(all_stats)
        
        # Lig ortalamaları
        league_averages = {}
        for i, feature in enumerate(self.feature_names):
            league_averages[feature] = {
                'ortalama': round(np.mean(stats_array[:, i]), 2),
                'std': round(np.std(stats_array[:, i]), 2),
                'min': round(np.min(stats_array[:, i]), 2),
                'max': round(np.max(stats_array[:, i]), 2)
            }
        
        # Oyun stili dağılımı
        from collections import Counter
        style_distribution = Counter(playing_styles)
        
        return {
            'total_teams': len(self.team_profiles),
            'league_averages': league_averages,
            'playing_style_distribution': dict(style_distribution),
            'most_common_style': style_distribution.most_common(1)[0] if style_distribution else ('Belirsiz', 0)
        }
    
    def predict_match_outcome(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """
        Maç sonucu tahmini (basit takım profili analizi)
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takımı
            
        Returns:
            Tahmin raporu
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi.")
        
        if home_team not in self.team_profiles or away_team not in self.team_profiles:
            raise ValueError("Takımlardan biri bulunamadı.")
        
        home_profile = self.get_team_profile(home_team)
        away_profile = self.get_team_profile(away_team)
        
        home_stats = home_profile['avg_stats']['home']  # Ev sahibi performansı
        away_stats = away_profile['avg_stats']['away']  # Deplasman performansı
        
        # Basit skor tahmini (geliştirilmesi gerekir)
        home_attack = (home_stats[2] + home_stats[3]) / 2  # shots + shots_on_target
        away_defense = away_stats[1]  # goals_against
        
        away_attack = (away_stats[2] + away_stats[3]) / 2
        home_defense = home_stats[1]
        
        # Ev sahibi avantajı ekleme
        home_advantage = 0.3
        
        expected_home_goals = max(0, (home_attack / 10) - (away_defense / 10) + home_advantage)
        expected_away_goals = max(0, (away_attack / 10) - (home_defense / 10))
        
        # Sonuç tahmini
        if expected_home_goals > expected_away_goals + 0.5:
            prediction = "Ev Sahibi Galibiyeti"
        elif expected_away_goals > expected_home_goals + 0.5:
            prediction = "Deplasman Galibiyeti"
        else:
            prediction = "Beraberlik"
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'prediction': prediction,
            'expected_score': {
                'home': round(expected_home_goals, 1),
                'away': round(expected_away_goals, 1)
            },
            'confidence_factors': {
                'home_form': home_profile['detailed_stats']['goals_for'],
                'away_form': away_profile['detailed_stats']['goals_for'],
                'home_defense': home_profile['detailed_stats']['goals_against'],
                'away_defense': away_profile['detailed_stats']['goals_against']
            },
            'key_matchup': self.compare_teams(home_team, away_team)
        }
    
    def save_model(self, file_path: str):
        """
        Modeli kaydet
        
        Args:
            file_path: Kayıt dosya yolu
        """
        import pickle
        
        model_data = {
            'team_profiles': self.team_profiles,
            'cluster_descriptions': self.cluster_descriptions,
            'feature_names': self.feature_names,
            'n_clusters': self.n_clusters,
            'clustering_method': self.clustering_method,
            'is_trained': self.is_trained
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model kaydedildi: {file_path}")
    
    def load_model(self, file_path: str):
        """
        Modeli yükle
        
        Args:
            file_path: Model dosya yolu
        """
        import pickle
        
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.team_profiles = model_data['team_profiles']
        self.cluster_descriptions = model_data['cluster_descriptions']
        self.feature_names = model_data['feature_names']
        self.n_clusters = model_data['n_clusters']
        self.clustering_method = model_data['clustering_method']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model yüklendi: {file_path}")


# Örnek kullanım ve test fonksiyonları
def test_team_analysis():
    """
    Test fonksiyonu
    """
    # Model oluştur
    model = TeamAnalysisModel(n_clusters=5)
    
    # Test verileri
    print("Test verileri hazırlanıyor...")
    
    return model


if __name__ == "__main__":
    # Test çalıştır
    model = test_team_analysis()
    print("TeamAnalysisModel hazır!")
