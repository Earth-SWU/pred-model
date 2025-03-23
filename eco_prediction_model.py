import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib  # 모델 저장 및 로드를 위한 라이브러리

# 데이터 로드
data_path = "Test_Training_Data.csv"
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

# 총 미션 수행 횟수 계산
total_mission_count = data.groupby('user_id')['mission_id'].count().reset_index()
total_mission_count.rename(columns={'mission_id': 'total_mission_count'}, inplace=True)
data = pd.merge(data, total_mission_count, on='user_id', how='left')

# 미션별 탄소 절감량 매핑 수정
carbon_reduction_map = {
    '텀블러 사용하기': 0.05,
    '3000걸음 줄이기': 0.2,
    '전자 영수증 업로드 미션': 0.1
}
data['carbon_reduction'] = data['name_x'].map(carbon_reduction_map)

# 사용자별 총 탄소 절감량 계산
weekly_carbon_reduction = data.groupby('user_id')['carbon_reduction'].sum().reset_index()
weekly_carbon_reduction.rename(columns={'carbon_reduction': 'total_weekly_carbon_reduction'}, inplace=True)
data = pd.merge(data, weekly_carbon_reduction, on='user_id', how='left')

# X와 y 생성
X = data[['total_mission_count']]
y = data['total_weekly_carbon_reduction']

# 데이터 분할 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=30, max_depth=7, min_samples_split=10, random_state=42)
rf_model.fit(X_train, y_train)

# 모델 저장
joblib.dump(rf_model, 'rf_model.joblib')
print("모델 저장 완료!")

# 예측 수행
data['predicted_carbon_reduction'] = rf_model.predict(X)

# 사용자별 환경 기여도 순위 계산
data['percentile_rank'] = data['predicted_carbon_reduction'].rank(pct=True) * 100

def categorize_user(percentile, total_missions):
    if total_missions < 7:  # Assuming a week's data collection
        return "만나게 되어 반가워요, 아직 더 많은 스텝이 필요해요!"
    elif percentile >= 75:
        return "상위 25% - 훌륭한 에코스텝러! 🌱"
    elif percentile >= 50:
        return "스탠다드 에코스텝러 - 좋은 참여를 보이고 있네요, 그러나 더 나아갈 수 있어요! 🚀"
    else:
        return "하위 50% - 작은 실천으로 더 큰 변화를! 🌎"

data['eco_category'] = data.apply(lambda x: categorize_user(x['percentile_rank'], x['total_mission_count']), axis=1)

# 결과 CSV 파일 저장
data.to_csv("final_user_eco_analysis.csv", index=False, encoding="utf-8-sig")

print("✅ 모델 실행 완료! `final_user_eco_analysis.csv` 파일이 생성되었습니다.")
