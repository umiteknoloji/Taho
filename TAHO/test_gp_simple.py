#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np

print("🎯 GP FUTBOL TAHMİN SİSTEMİ - TEST")
print("=" * 50)

# Test data loading
try:
    with open('data/TR_stat.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Data loaded: {len(data)} matches")
    
    # Test parsing first few matches
    matches = []
    for i, match in enumerate(data[:5]):
        score_data = match.get('score', {})
        full_time = score_data.get('fullTime', {})
        home_score = int(full_time.get('home', 0))
        away_score = int(full_time.get('away', 0))
        
        if home_score > away_score:
            result = '1'
        elif home_score == away_score:
            result = 'X'
        else:
            result = '2'
            
        print(f"Match {i+1}: {match.get('home', '')} {home_score}-{away_score} {match.get('away', '')} -> {result}")
        
        matches.append({
            'home': match.get('home', ''),
            'away': match.get('away', ''),
            'home_score': home_score,
            'away_score': away_score,
            'result': result,
            'week': match.get('week', 0)
        })
    
    df = pd.DataFrame(matches)
    print(f"\n📊 DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Basic stats
    result_counts = df['result'].value_counts()
    print(f"\nResult distribution in sample: {dict(result_counts)}")
    
    print("\n✅ Basic functionality works!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
