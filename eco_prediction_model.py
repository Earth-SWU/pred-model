import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 데이터 로드
data_path = "final_fully_adjusted_data.csv"
data = pd.read_csv(data_path)

# 날짜 변환
date_cols = ['completed_at', 'session_start', 'session_end', 'start_time', 'end_time']
for col in date_cols:
    data[col] = pd.to_datetime(data[col], errors='coerce')

# 세션 지속 시간 계산 (분 단위)
data['session_duration'] = (data['session_end'] - data['session_start']).dt.total_seconds() / 60

# 활동 빈도 계산 (일일 미션 수행 횟수)
activity_counts = data.groupby(['user_id', 'date'])['activity_type_y'].count().reset_index()
activity_counts.rename(columns={'activity_type_y': 'daily_activity_count'}, inplace=True)
data = pd.merge(data, activity_counts, on=['user_id', 'date'], how='left')

# ✅ **총 미션 수행 횟수 계산**
total_mission_count = data.groupby('user_id')['mission_id'].count().reset_index()
total_mission_count.rename(columns={'mission_id': 'total_mission_count'}, inplace=True)
data = pd.merge(data, total_mission_count, on='user_id', how='left')

# ✅ **총 클릭 횟수 계산**
total_clicks = data[data['activity_type_x'] == 'Button Click'].groupby('user_id')['activity_type_x'].count().reset_index()
total_clicks.rename(columns={'activity_type_x': 'total_clicks'}, inplace=True)
data = pd.merge(data, total_clicks, on='user_id', how='left')
data['total_clicks'].fillna(0, inplace=True)  # 클릭 없는 유저는 0으로 설정

# 미션별 탄소 절감량 매핑
carbon_reduction_map = {
    '텀블러 사용하기': 0.3,
    '대중교통 이용하기': 1.2,
    '잔반 없는 식사 인증하기': 0.5,
    '5000걸음 걷기': 0.7,
    '환경 문제 OX 퀴즈 풀기': 0.2,
    '소비 내역 인증하기': 0.4
}
data['carbon_reduction'] = data['name_x'].map(carbon_reduction_map)

# ✅ **사용자별 총 탄소 절감량 계산**
weekly_carbon_reduction = data.groupby('user_id')['carbon_reduction'].sum().reset_index()
weekly_carbon_reduction.rename(columns={'carbon_reduction': 'total_weekly_carbon_reduction'}, inplace=True)
data = pd.merge(data, weekly_carbon_reduction, on='user_id', how='left')

# ✅ **🚀 🔥 여기서 X를 생성해야 함 (칼럼이 존재한 이후!)**
X = data[['total_mission_count', 'total_clicks']]
y = data['total_weekly_carbon_reduction']

# 데이터 분할 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=30, max_depth=7, min_samples_split=10, random_state=42)
rf_model.fit(X_train, y_train)

# 예측 수행
data['predicted_carbon_reduction'] = rf_model.predict(X)

# ✅ **사용자별 환경 기여도 순위 계산**
data['percentile_rank'] = data['predicted_carbon_reduction'].rank(pct=True) * 100

def categorize_user(percentile):
    if percentile >= 75:
        return "상위 25% - 훌륭한 에코스텝러! 🌱"
    elif percentile >= 50:
        return "상위 50% - 좋은 참여, 더 나아갈 수 있어요! 🚀"
    else:
        return "하위 50% - 작은 실천으로 더 큰 변화를! 🌎"

data['eco_category'] = data['percentile_rank'].apply(categorize_user)

# ✅ **사용자 연령대 및 지역 추가**
data['age_group'] = np.random.choice(['20대 초반', '20대 후반', '30대'], size=len(data))
data['region'] = np.random.choice(['서울', '부산', '대구', '광주', '대전'], size=len(data))

# ✅ **그룹별 평균 탄소 절감량 계산**
group_avg = data.groupby(['age_group', 'region'])['predicted_carbon_reduction'].mean().reset_index()
group_avg.rename(columns={'predicted_carbon_reduction': 'group_avg_carbon_reduction'}, inplace=True)
data = pd.merge(data, group_avg, on=['age_group', 'region'], how='left')

# ✅ **사용자별 평균 대비 차이 계산**
data['carbon_reduction_diff'] = data['predicted_carbon_reduction'] - data['group_avg_carbon_reduction']

# ✅ **사용자 맞춤 피드백 생성**
def generate_feedback(row):
    if row['carbon_reduction_diff'] > 0:
        return f"당신은 같은 연령대({row['age_group']}) {row['region']} 지역 사용자들보다 더 많은 환경 기여를 하고 있어요! 🎉"
    else:
        return f"비슷한 사용자들은 더 활발히 환경 활동을 하고 있어요! {row['region']}에서 더 많은 미션을 수행해보는 건 어떨까요? 🌱"

data['personal_feedback'] = data.apply(generate_feedback, axis=1)

# ✅ **결과 CSV 파일 저장 (칼럼 순서 유지)**
columns_order = [
    'user_id', 'name_x', 'carbon_reduction', 'total_weekly_carbon_reduction',
    'total_mission_count', 'total_clicks', 'predicted_carbon_reduction',
    'percentile_rank', 'eco_category', 'age_group', 'region',
    'group_avg_carbon_reduction', 'carbon_reduction_diff', 'personal_feedback'
]
data = data[columns_order]

data.to_csv("final_user_eco_analysis.csv", index=False, encoding="utf-8-sig")

print("✅ 모델 실행 완료! `final_user_eco_analysis.csv` 파일이 생성되었습니다.")
