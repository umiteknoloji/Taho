"""
Evrensel Futbol Verisi Normalizasyon Modülü
Bu modül farklı lig formatlarındaki verileri ortak bir formata dönüştürür.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataNormalizer:
    """
    Farklı lig veri formatlarını normalize eden sınıf.
    """
    
    def __init__(self):
        """
        Normalizasyon kurallarını tanımla.
        """
        # Alan eşlemeleri - farklı lig formatlarından ortak alan adlarına
        self.field_mappings = {
            # Temel bilgiler
            'week': ['week', 'hafta', 'round'],
            'date': ['date', 'tarih', 'maç_tarihi'],
            'home': ['home', 'ev_sahibi', 'home_team'],
            'away': ['away', 'deplasman', 'away_team'],
            'attendance': ['attendance', 'seyirci', 'katılım'],
            
            # Skor bilgileri
            'score.fullTime.home': ['score.fullTime.home', 'final_skor.ev'],
            'score.fullTime.away': ['score.fullTime.away', 'final_skor.deplasman'],
            'score.halfTime.home': ['score.halfTime.home', 'yarı_skor.ev'],
            'score.halfTime.away': ['score.halfTime.away', 'yarı_skor.deplasman'],
            
            # İstatistik alanları - normalleştirilmiş isimler
            'possession_home': ['Topla Oynama.home', 'Ball Possession.home', 'Possession.home'],
            'possession_away': ['Topla Oynama.away', 'Ball Possession.away', 'Possession.away'],
            'total_shots_home': ['Toplam Şut.home', 'Total Shots.home', 'Shots.home'],
            'total_shots_away': ['Toplam Şut.away', 'Total Shots.away', 'Shots.away'],
            'shots_on_target_home': ['İsabetli Şut.home', 'Shots on Target.home', 'On Target.home'],
            'shots_on_target_away': ['İsabetli Şut.away', 'Shots on Target.away', 'On Target.away'],
            'corners_home': ['Korner.home', 'Köşe Vuruşu.home', 'Corner Kicks.home'],
            'corners_away': ['Korner.away', 'Köşe Vuruşu.away', 'Corner Kicks.away'],
            'fouls_home': ['Faul.home', 'Fauller.home', 'Fouls.home'],
            'fouls_away': ['Faul.away', 'Fauller.away', 'Fouls.away'],
            'offsides_home': ['Ofsayt.home', 'Ofsaytlar.home', 'Offsides.home'],
            'offsides_away': ['Ofsayt.away', 'Ofsaytlar.away', 'Offsides.away'],
            'passes_home': ['Başarılı Paslar.home', 'Successful Passes.home'],
            'passes_away': ['Başarılı Paslar.away', 'Successful Passes.away'],
            'pass_accuracy_home': ['Pas Başarı(_).home', 'Pass Accuracy.home'],
            'pass_accuracy_away': ['Pas Başarı(_).away', 'Pass Accuracy.away'],
            'crosses_home': ['Orta.home', 'Crosses.home'],
            'crosses_away': ['Orta.away', 'Crosses.away'],
            'hit_post_home': ['Direkten Dönen.home', 'Hit Post.home'],
            'hit_post_away': ['Direkten Dönen.away', 'Hit Post.away']
        }
        
        # Standart alan adları listesi
        self.standard_fields = [
            'week', 'date', 'home', 'away', 'attendance',
            'home_score', 'away_score', 'home_ht_score', 'away_ht_score',
            'possession_home', 'possession_away',
            'total_shots_home', 'total_shots_away',
            'shots_on_target_home', 'shots_on_target_away',
            'corners_home', 'corners_away',
            'fouls_home', 'fouls_away',
            'offsides_home', 'offsides_away',
            'passes_home', 'passes_away',
            'pass_accuracy_home', 'pass_accuracy_away',
            'crosses_home', 'crosses_away',
            'hit_post_home', 'hit_post_away'
        ]
        
    def detect_league_format(self, data: List[Dict]) -> str:
        """
        Veri formatını otomatik olarak algıla.
        
        Args:
            data: Ham veri listesi
            
        Returns:
            str: Algılanan lig formatı ('german', 'belgian', 'generic')
        """
        if not data or len(data) == 0:
            return 'generic'
            
        sample = data[0]
        
        # Alman ligi özellikleri
        if ('stats' in sample and 
            'Başarılı Paslar' in sample.get('stats', {}) and
            'Pas Başarı(_)' in sample.get('stats', {})):
            logger.info("Alman ligi formatı algılandı")
            return 'german'
            
        # Belçika ligi özellikleri
        if ('stats' in sample and 
            'Direkten Dönen' in sample.get('stats', {}) and
            'attendance' in sample):
            logger.info("Belçika ligi formatı algılandı")
            return 'belgian'
            
        logger.info("Genel format olarak algılandı")
        return 'generic'
    
    def normalize_percentage(self, value: str) -> float:
        """
        Yüzde değerini sayıya çevir.
        
        Args:
            value: Yüzde stringi (örn: "%45")
            
        Returns:
            float: Normalize edilmiş değer (0.45)
        """
        if not value:
            return 0.0
            
        try:
            if isinstance(value, str) and value.startswith('%'):
                return float(value[1:]) / 100.0
            return float(value) / 100.0 if float(value) > 1 else float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def normalize_cross_stat(self, value: str) -> Dict[str, float]:
        """
        Orta istatistiğini normalize et (örn: "5/15" -> {'successful': 5, 'total': 15})
        
        Args:
            value: Orta stringi
            
        Returns:
            Dict: Başarılı ve toplam orta sayısı
        """
        if not value or not isinstance(value, str):
            return {'successful': 0.0, 'total': 0.0}
            
        try:
            if '/' in value:
                parts = value.split('/')
                return {
                    'successful': float(parts[0]),
                    'total': float(parts[1])
                }
            else:
                # Sadece sayı varsa
                return {'successful': float(value), 'total': float(value)}
        except (ValueError, IndexError):
            return {'successful': 0.0, 'total': 0.0}
    
    def extract_stat_value(self, stats_dict: Dict, field_name: str, team: str) -> Any:
        """
        İstatistik sözlüğünden belirli bir alanı çıkar.
        
        Args:
            stats_dict: İstatistik sözlüğü
            field_name: Alan adı
            team: Takım ('home' veya 'away')
            
        Returns:
            Any: Çıkarılan değer
        """
        if not stats_dict or field_name not in stats_dict:
            return None
            
        field_data = stats_dict[field_name]
        
        if isinstance(field_data, dict):
            return field_data.get(team)
        
        return field_data
    
    def normalize_match_data(self, match_data: Dict, league_format: str) -> Dict:
        """
        Tek bir maç verisini normalize et.
        
        Args:
            match_data: Ham maç verisi
            league_format: Lig formatı
            
        Returns:
            Dict: Normalize edilmiş maç verisi
        """
        normalized = {}
        
        # Temel bilgileri normalize et
        normalized['week'] = match_data.get('week', '')
        normalized['date'] = match_data.get('date', '')
        normalized['home'] = match_data.get('home', '')
        normalized['away'] = match_data.get('away', '')
        normalized['attendance'] = match_data.get('attendance', 0)
        
        # Skor bilgilerini normalize et
        score = match_data.get('score', {})
        full_time = score.get('fullTime', {})
        half_time = score.get('halfTime', {})
        
        normalized['home_score'] = int(full_time.get('home', 0)) if full_time.get('home') else 0
        normalized['away_score'] = int(full_time.get('away', 0)) if full_time.get('away') else 0
        normalized['home_ht_score'] = int(half_time.get('home', 0)) if half_time.get('home') else 0
        normalized['away_ht_score'] = int(half_time.get('away', 0)) if half_time.get('away') else 0
        
        # İstatistikleri normalize et
        stats = match_data.get('stats', {})
        
        if stats:
            # Topla oynama
            possession_home = self.extract_stat_value(stats, 'Topla Oynama', 'home')
            possession_away = self.extract_stat_value(stats, 'Topla Oynama', 'away')
            
            normalized['possession_home'] = self.normalize_percentage(possession_home)
            normalized['possession_away'] = self.normalize_percentage(possession_away)
            
            # Şutlar
            normalized['total_shots_home'] = int(self.extract_stat_value(stats, 'Toplam Şut', 'home') or 0)
            normalized['total_shots_away'] = int(self.extract_stat_value(stats, 'Toplam Şut', 'away') or 0)
            
            normalized['shots_on_target_home'] = int(self.extract_stat_value(stats, 'İsabetli Şut', 'home') or 0)
            normalized['shots_on_target_away'] = int(self.extract_stat_value(stats, 'İsabetli Şut', 'away') or 0)
            
            # Kornerler
            corner_field = 'Korner' if 'Korner' in stats else 'Köşe Vuruşu'
            normalized['corners_home'] = int(self.extract_stat_value(stats, corner_field, 'home') or 0)
            normalized['corners_away'] = int(self.extract_stat_value(stats, corner_field, 'away') or 0)
            
            # Fauller
            foul_field = 'Faul' if 'Faul' in stats else 'Fauller'
            normalized['fouls_home'] = int(self.extract_stat_value(stats, foul_field, 'home') or 0)
            normalized['fouls_away'] = int(self.extract_stat_value(stats, foul_field, 'away') or 0)
            
            # Ofsaytlar
            offside_field = 'Ofsayt' if 'Ofsayt' in stats else 'Ofsaytlar'
            normalized['offsides_home'] = int(self.extract_stat_value(stats, offside_field, 'home') or 0)
            normalized['offsides_away'] = int(self.extract_stat_value(stats, offside_field, 'away') or 0)
            
            # Alman ligi özel alanları
            if league_format == 'german':
                normalized['passes_home'] = int(self.extract_stat_value(stats, 'Başarılı Paslar', 'home') or 0)
                normalized['passes_away'] = int(self.extract_stat_value(stats, 'Başarılı Paslar', 'away') or 0)
                
                pass_acc_home = self.extract_stat_value(stats, 'Pas Başarı(_)', 'home')
                pass_acc_away = self.extract_stat_value(stats, 'Pas Başarı(_)', 'away')
                
                normalized['pass_accuracy_home'] = self.normalize_percentage(pass_acc_home)
                normalized['pass_accuracy_away'] = self.normalize_percentage(pass_acc_away)
                
                cross_home = self.extract_stat_value(stats, 'Orta', 'home')
                cross_away = self.extract_stat_value(stats, 'Orta', 'away')
                
                cross_home_data = self.normalize_cross_stat(cross_home)
                cross_away_data = self.normalize_cross_stat(cross_away)
                
                normalized['crosses_successful_home'] = cross_home_data['successful']
                normalized['crosses_total_home'] = cross_home_data['total']
                normalized['crosses_successful_away'] = cross_away_data['successful']
                normalized['crosses_total_away'] = cross_away_data['total']
            
            # Belçika ligi özel alanları
            if league_format == 'belgian':
                normalized['hit_post_home'] = int(self.extract_stat_value(stats, 'Direkten Dönen', 'home') or 0)
                normalized['hit_post_away'] = int(self.extract_stat_value(stats, 'Direkten Dönen', 'away') or 0)
        
        # Eksik alanları varsayılan değerlerle doldur
        for field in self.standard_fields:
            if field not in normalized:
                if field.endswith('_home') or field.endswith('_away'):
                    normalized[field] = 0.0
                else:
                    normalized[field] = None
        
        return normalized
    
    def normalize_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Tüm veri setini normalize et.
        
        Args:
            file_path: JSON dosya yolu
            
        Returns:
            pd.DataFrame: Normalize edilmiş DataFrame
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                logger.warning(f"Boş veri dosyası: {file_path}")
                return pd.DataFrame()
            
            # Format algıla
            league_format = self.detect_league_format(data)
            logger.info(f"Algılanan format: {league_format}")
            
            # Her maçı normalize et
            normalized_matches = []
            for match in data:
                if match.get('score'):  # Sadece tamamlanmış maçları al
                    normalized_match = self.normalize_match_data(match, league_format)
                    normalized_matches.append(normalized_match)
            
            df = pd.DataFrame(normalized_matches)
            logger.info(f"Normalize edildi: {len(df)} maç verisi")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Dosya bulunamadı: {file_path}")
            return pd.DataFrame()
        except json.JSONDecodeError:
            logger.error(f"JSON parse hatası: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Normalizasyon hatası: {str(e)}")
            return pd.DataFrame()
    
    def get_feature_columns(self, league_format: str = None) -> List[str]:
        """
        ML modelleri için kullanılacak özellik sütunlarını döndür.
        
        Args:
            league_format: Lig formatı (optional)
            
        Returns:
            List[str]: Özellik sütun adları
        """
        base_features = [
            'possession_home', 'possession_away',
            'total_shots_home', 'total_shots_away',
            'shots_on_target_home', 'shots_on_target_away',
            'corners_home', 'corners_away',
            'fouls_home', 'fouls_away',
            'offsides_home', 'offsides_away'
        ]
        
        if league_format == 'german':
            base_features.extend([
                'passes_home', 'passes_away',
                'pass_accuracy_home', 'pass_accuracy_away',
                'crosses_successful_home', 'crosses_total_home',
                'crosses_successful_away', 'crosses_total_away'
            ])
        elif league_format == 'belgian':
            base_features.extend([
                'hit_post_home', 'hit_post_away'
            ])
        
        return base_features
    
    def merge_datasets(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Birden fazla lig verisini birleştir.
        
        Args:
            file_paths: Dosya yolları listesi
            
        Returns:
            pd.DataFrame: Birleştirilmiş DataFrame
        """
        all_dataframes = []
        
        for file_path in file_paths:
            df = self.normalize_dataset(file_path)
            if not df.empty:
                # Lig bilgisini ekle
                if 'ALM' in file_path:
                    df['league'] = 'German'
                elif 'BEL' in file_path:
                    df['league'] = 'Belgian'
                else:
                    df['league'] = 'Unknown'
                    
                all_dataframes.append(df)
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            logger.info(f"Birleştirildi: {len(combined_df)} toplam maç verisi")
            return combined_df
        
        return pd.DataFrame()

# Test fonksiyonu
def test_normalizer():
    """
    Normalizasyon modülünü test et.
    """
    normalizer = DataNormalizer()
    
    # Alman ligi testi
    german_df = normalizer.normalize_dataset('/Users/umitduman/Taho/futbol_skor_tahmin_projesi_final_updated/data/ALM_stat.json')
    print(f"Alman ligi: {len(german_df)} maç")
    print("Sütunlar:", german_df.columns.tolist())
    
    # Belçika ligi testi
    belgian_df = normalizer.normalize_dataset('/Users/umitduman/Taho/futbol_skor_tahmin_projesi_final_updated/data/BEL_stat.json')
    print(f"Belçika ligi: {len(belgian_df)} maç")
    print("Sütunlar:", belgian_df.columns.tolist())
    
    # Birleştirme testi
    combined_df = normalizer.merge_datasets([
        '/Users/umitduman/Taho/futbol_skor_tahmin_projesi_final_updated/data/ALM_stat.json',
        '/Users/umitduman/Taho/futbol_skor_tahmin_projesi_final_updated/data/BEL_stat.json'
    ])
    print(f"Birleştirilmiş: {len(combined_df)} maç")

if __name__ == "__main__":
    test_normalizer()
