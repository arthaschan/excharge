"""充电站网络模拟数据生成器
100个充电站 × 20个充电桩/站 × 365天
故障率：1%（每设备每年约88小时故障）
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

OUT = '/Users/arthas/.qclaw/workspace-dhj4e57a67drnnbd/simulated_data'
NUM_STATIONS = 100
PILES_PER_STATION = 20
TOTAL_PILES = NUM_STATIONS * PILES_PER_STATION
DAYS = 365
FAULT_RATE = 0.01  # 1% of intervals
FEATURES_PER_HOUR = 4  # 15min granularity
HOURS_PER_DAY = 24
TOTAL_HOURS = DAYS * HOURS_PER_DAY  # 8760

print(f"Total piles: {TOTAL_PILES}")
print(f"Total hours per pile: {TOTAL_HOURS}")
print(f"Full dataset size: {TOTAL_PILES * TOTAL_HOURS / 1e6:.1f}M records")

# ============================================================
# 1. Station & Pile Inventory
# ============================================================
stations = []
for i in range(1, NUM_STATIONS + 1):
    station_id = f"ST{i:04d}"
    lat = round(random.uniform(22.5, 22.9), 6)  # Shenzhen-ish
    lon = round(random.uniform(113.8, 114.3), 6)
    district = random.choice(['宝安','南山','福田','罗湖','龙岗','龙华','光明','坪山','盐田'])
    stations.append({'station_id': station_id, 'district': district, 'lat': lat, 'lon': lon})

stations_df = pd.DataFrame(stations)
stations_df.to_csv(f'{OUT}/stations.csv', index=False)
print(f"Stations: {len(stations_df)}")

piles = []
for i, st in enumerate(stations):
    for j in range(1, PILES_PER_STATION + 1):
        pile_id = f"{st['station_id']}-P{j:02d}"
        power_rating = random.choice([60, 120, 180, 240, 360])  # kW
        install_date = pd.Timestamp('2023-01-01') + timedelta(days=random.randint(0, 365))
        piles.append({
            'pile_id': pile_id, 'station_id': st['station_id'],
            'power_rating_kw': power_rating, 'district': st['district'],
            'install_date': install_date.strftime('%Y-%m-%d')
        })

piles_df = pd.DataFrame(piles)
piles_df.to_csv(f'{OUT}/piles.csv', index=False)
print(f"Piles: {len(piles_df)}")

# ============================================================
# 2. Fault Records (站长标注用)
# ============================================================
FAULT_TYPES = {
    'contact_poor': {'label': '接触不良', 'prob': 0.35, 'desc': '充电枪头接触不良/松动'},
    'overheat': {'label': '过温保护', 'prob': 0.25, 'desc': '设备过热保护触发'},
    'comm_fault': {'label': '通讯故障', 'prob': 0.20, 'desc': '模块通讯中断/丢包'},
    'hw_damage': {'label': '硬件损坏', 'prob': 0.10, 'desc': '功率模块/继电器/屏幕损坏'},
    'power_abnormal': {'label': '功率异常', 'prob': 0.05, 'desc': '输出功率异常波动'},
    'cable_damage': {'label': '线缆问题', 'prob': 0.05, 'desc': '线缆老化/破损/断股'},
}

# Fault generation: each pile has ~1% chance/hour, generate fault events
fault_records = []
start_date = pd.Timestamp('2025-08-01')

# Use yearly fault probability: 1% of 365 days = ~3.65 days/pile
# Split into events: average 2-5 fault events per pile per year
for pile in piles_df.itertuples():
    # Number of faults this year (Poisson with lambda ~3-4)
    n_faults = np.random.poisson(3.5)
    n_faults = max(1, min(10, n_faults))  # 1-10 faults/year
    
    # Generate fault types for this pile
    fault_types_pool = random.choices(
        list(FAULT_TYPES.keys()),
        weights=[FT['prob'] for FT in FAULT_TYPES.values()],
        k=n_faults
    )
    
    for ft in fault_types_pool:
        ft_info = FAULT_TYPES[ft]
        # Fault time in 2025-08-01 ~ 2026-07-31
        fault_day = random.randint(0, 364)
        fault_hour = random.randint(0, 23)
        fault_start = start_date + timedelta(days=fault_day, hours=fault_hour)
        
        # Fault duration: 30min to 72 hours, most 1-8 hours
        duration_hours = max(0.5, np.random.exponential(4))
        duration_hours = min(72, duration_hours)
        fault_end = fault_start + timedelta(hours=duration_hours)
        
        # Repair measure
        measures = {
            'contact_poor': random.choice(['更换充电枪头','清洁接头','紧固接线端子']),
            'overheat': random.choice(['清理散热片','检查风扇','降低功率运行']),
            'comm_fault': random.choice(['重启通讯模块','更换4G/5G模块','检查天线连接']),
            'hw_damage': random.choice(['更换功率模块','更换控制板','返厂维修']),
            'power_abnormal': random.choice(['校准功率表','更换传感器','固件升级']),
            'cable_damage': random.choice(['更换充电线缆','修补线缆外皮','更换连接器']),
        }
        
        fault_records.append({
            'pile_id': pile.pile_id,
            'station_id': pile.station_id,
            'fault_start_time': fault_start.strftime('%Y-%m-%d %H:%M'),
            'fault_end_time': fault_end.strftime('%Y-%m-%d %H:%M'),
            'fault_type': ft_info['label'],
            'fault_desc': ft_info['desc'],
            'repair_measure': measures[ft],
            'duration_hours': round(duration_hours, 1),
        })

faults_df = pd.DataFrame(fault_records)
faults_df = faults_df.sort_values(['station_id', 'pile_id', 'fault_start_time']).reset_index(drop=True)
faults_df.to_csv(f'{OUT}/fault_records.csv', index=False)

print(f"Fault records: {len(faults_df)}")
print(f"Fault type distribution:\n{faults_df['fault_type'].value_counts()}")
print(f"Piles affected: {faults_df['pile_id'].nunique()}/{TOTAL_PILES}")

# ============================================================
# 3. Charging Time Series (sampled, not full 17.5M rows)
# ============================================================
# Generate daily summaries per pile (365 days × 2000 piles ≈ 730K rows)
# With sampled hourly data for 3 stations (60 piles × 8760 ≈ 525K rows)

print("\nGenerating daily summaries...")
daily_records = []
for pile in piles_df.itertuples():
    # Each pile has a typical daily charging profile
    # Peak hours: 10-12, 14-16, 19-21
    base_daily_kwh = round(random.uniform(200, 800), 0)
    base_daily_sessions = random.randint(8, 30)
    
    for d in range(365):
        date = (start_date + timedelta(days=d)).strftime('%Y-%m-%d')
        
        # Weekend/holiday adjustment
        wday = (start_date + timedelta(days=d)).dayofweek
        weekend_factor = 0.85 if wday >= 5 else 1.0
        
        # Seasonal adjustment: summer higher
        month = (start_date + timedelta(days=d)).month
        season_factor = 1.15 if month in [6,7,8] else (0.85 if month in [12,1,2] else 1.0)
        
        # Weather effect (random noise)
        weather_factor = np.random.normal(1.0, 0.1)
        weather_factor = max(0.5, min(1.5, weather_factor))
        
        total_kwh = base_daily_kwh * weekend_factor * season_factor * weather_factor
        total_sessions = max(0, int(base_daily_sessions * weekend_factor + np.random.normal(0, 2)))
        
        # Fault check: was this pile faulty today?
        fault_today = faults_df[
            (faults_df['pile_id'] == pile.pile_id) &
            (pd.to_datetime(faults_df['fault_start_time']).dt.strftime('%Y-%m-%d') <= date) &
            (pd.to_datetime(faults_df['fault_end_time']).dt.strftime('%Y-%m-%d') >= date)
        ]
        
        has_fault = len(fault_today) > 0
        fault_type_today = fault_today.iloc[0]['fault_type'] if has_fault else None
        
        # If fault, reduce charging
        if has_fault:
            total_kwh *= random.uniform(0.0, 0.3)  # most sessions fail
            total_sessions = max(0, int(total_sessions * 0.2))
        
        daily_records.append({
            'pile_id': pile.pile_id,
            'station_id': pile.station_id,
            'date': date,
            'total_kwh': round(total_kwh, 2),
            'total_sessions': total_sessions,
            'avg_power_kw': round(total_kwh / 24, 2),
            'peak_power_kw': round(total_kwh / 24 * random.uniform(1.5, 2.5), 2),
            'has_fault': has_fault,
            'fault_type': fault_type_today if has_fault else None,
        })

daily_df = pd.DataFrame(daily_records)
daily_df.to_parquet(f'{OUT}/daily_summaries.parquet', index=False)
print(f"Daily summaries: {len(daily_df)} rows, {len(daily_df['pile_id'].unique())} piles")
print(f"Fault days: {daily_df['has_fault'].sum()} ({100*daily_df['has_fault'].sum()/len(daily_df):.2f}%)")

# ============================================================
# 4. Hourly data for 3 stations (for model training demos)
# ============================================================
print("\nGenerating hourly data for 3 stations (60 piles × 8760h)...")
sample_stations = stations[:3]
sample_piles = [p for p in piles if p['station_id'] in [s['station_id'] for s in sample_stations]]

hourly_records = []
base_timestamps = [start_date + timedelta(hours=h) for h in range(TOTAL_HOURS)]

for pile in sample_piles:
    pile_faults = faults_df[faults_df['pile_id'] == pile['pile_id']]
    
    for ts in base_timestamps:
        hour = ts.hour
        month = ts.month
        
        # Build realistic charging pattern
        # Night (0-5): low, Morning peak (8-11): medium, Noon (12-14): high, Afternoon (15-17): medium, Evening (18-21): high
        hourly_profile = {
            (0,5): 0.05, (6,7): 0.2, (8,9,10,11): 0.6, (12,13,14): 0.7,
            (15,16,17): 0.5, (18,19,20,21): 0.75, (22,23): 0.3
        }
        base_factor = 0.1
        for hrs, factor in hourly_profile.items():
            if hour in (hrs if isinstance(hrs, tuple) else [hrs]):
                base_factor = factor
                break
        
        potential_power = pile['power_rating_kw'] * base_factor * np.random.uniform(0.7, 1.3)
        
        # Check if fault at this hour
        ts_str = ts.strftime('%Y-%m-%d %H:%M')
        is_fault = False
        ft = None
        for _, f in pile_faults.iterrows():
            f_start = pd.Timestamp(f['fault_start_time'])
            f_end = pd.Timestamp(f['fault_end_time'])
            if f_start <= ts <= f_end:
                is_fault = True
                ft = f['fault_type']
                potential_power *= random.uniform(0.0, 0.1)  # nearly zero during fault
                break
        
        # Add noise
        power = max(0, potential_power + np.random.normal(0, 2))
        voltage = round(380 * (0.95 + np.random.normal(0, 0.01)), 1) if power > 1 else 0
        current = round(power / voltage * 1000 if voltage > 0 else 0, 1)
        temp = round(25 + power / pile['power_rating_kw'] * 30 + np.random.normal(0, 2), 1)
        efficiency = round(min(0.98, 0.92 + 0.06 * (1 - power/pile['power_rating_kw']) + np.random.normal(0, 0.005)), 4) if power > 5 else 0
        
        hourly_records.append({
            'pile_id': pile['pile_id'], 'station_id': pile['station_id'],
            'timestamp': ts_str,
            'active_power_kw': round(power, 2),
            'voltage_v': voltage,
            'current_a': current,
            'temperature_c': temp,
            'efficiency': efficiency,
            'power_factor': round(0.92 + np.random.normal(0, 0.02), 3) if power > 1 else 0,
            'cumulative_kwh': 0,  # placeholder, will aggregate
            'is_fault': is_fault,
            'fault_type': ft,
        })

hourly_df = pd.DataFrame(hourly_records)
hourly_df.to_parquet(f'{OUT}/hourly_sample_3stations.parquet', index=False)
print(f"Hourly sample: {len(hourly_df)} rows")
print(f"Fault hours in sample: {hourly_df['is_fault'].sum()} ({100*hourly_df['is_fault'].sum()/len(hourly_df):.3f}%)")
print(f"Fault type distribution (hourly):\n{hourly_df['fault_type'].value_counts(dropna=False)}")

# ============================================================
# 5. Summary
# ============================================================
print("\n" + "="*60)
print("DATA GENERATION COMPLETE")
print(f"  stations.csv:            {len(stations_df)} stations")
print(f"  piles.csv:               {len(piles_df)} piles")
print(f"  fault_records.csv:       {len(faults_df)} fault events")
print(f"  daily_summaries.parquet: {len(daily_df):,} rows ({round(len(daily_df)*4/1024, 1)} KB est)")
print(f"  hourly_sample:           {len(hourly_df):,} rows ({round(len(hourly_df)*4/1024, 1)} KB est)")
print(f"\n  Fault rate: {faults_df['pile_id'].nunique()}/{TOTAL_PILES} piles affected")
print(f"  Median faults/pile: {faults_df.groupby('pile_id').size().median():.0f}")
