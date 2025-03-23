from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import joblib
from datetime import datetime

# FastAPI 객체 생성
app = FastAPI()

# 학습된 모델 로드
model = joblib.load("rf_model.joblib")

# 요청 데이터 모델 정의 (입력 컬럼 전체 포함, 점(.) 필드 alias 처리)
class UserRecord(BaseModel):
    id_x: int
    user_id: int
    mission_id: int
    completed_at: str
    id_y: int
    name_x: str
    description: str
    carbon_reduction: float = 0.0
    name_y: str
    email: str
    created_at: str
    id_x_1: int = Field(..., alias="id_x.1")  # JSON: id_x.1
    timestamp: str
    activity_type_x: str
    page_id: Optional[str] = None
    button_id: Optional[str] = None
    id_y_1: int = Field(..., alias="id_y.1")  # JSON: id_y.1
    session_start: str
    session_end: str
    duration: str
    activity_type_y: str
    start_time: str
    end_time: str
    date: str

    class Config:
        allow_population_by_field_name = True  # alias로 매핑 허용

# 예측 API
@app.post("/predict/")
async def predict(user_data: List[UserRecord]):
    try:
        # 입력 데이터를 DataFrame으로 변환 (alias 이름 사용)
        df = pd.DataFrame([record.dict(by_alias=True) for record in user_data])

        # 날짜/시간 컬럼 파싱
        date_cols = ['completed_at', 'session_start', 'session_end', 'start_time', 'end_time']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        # 세션 시간 계산
        df['session_duration'] = (df['session_end'] - df['session_start']).dt.total_seconds() / 60

        # 날짜 컬럼 파싱
        df['date'] = pd.to_datetime(df['date'])

        # 일일 활동 횟수 계산
        activity_counts = df.groupby(['user_id', 'date'])['activity_type_y'].count().reset_index()
        activity_counts.rename(columns={'activity_type_y': 'daily_activity_count'}, inplace=True)
        df = pd.merge(df, activity_counts, on=['user_id', 'date'], how='left')

        # 사용자별 총 미션 횟수 계산
        total_missions = df.groupby('user_id')['mission_id'].count().reset_index()
        total_missions.rename(columns={'mission_id': 'total_mission_count'}, inplace=True)
        df = pd.merge(df, total_missions, on='user_id', how='left')

        # 미션 이름 → 탄소 절감량 매핑
        carbon_reduction_map = {
            '텀블러 사용하기': 0.05,
            '3000걸음 줄이기': 0.2,
            '전자 영수증 업로드 미션': 0.1
        }
        df['carbon_reduction'] = df['name_x'].map(carbon_reduction_map)

        # 사용자별 총 탄소 절감량 계산
        weekly_cr = df.groupby('user_id')['carbon_reduction'].sum().reset_index()
        weekly_cr.rename(columns={'carbon_reduction': 'total_weekly_carbon_reduction'}, inplace=True)
        df = pd.merge(df, weekly_cr, on='user_id', how='left')

        # 모델 예측
        X = df[['total_mission_count']]
        df['predicted_carbon_reduction'] = model.predict(X)

        # 기여도 퍼센타일 계산
        df['percentile_rank'] = df['predicted_carbon_reduction'].rank(pct=True) * 100

        # 환경 카테고리 분류 함수
        def categorize(percentile, total_missions):
            if total_missions < 7:
                return "만나게 되어 반가워요, 아직 더 많은 스텝이 필요해요!"
            elif percentile >= 75:
                return "상위 25% - 훌륭한 에코스텝러! 🌱"
            elif percentile >= 50:
                return "스탠다드 에코스텝러 - 좋은 참여를 보이고 있네요, 그러나 더 나아갈 수 있어요! 🚀"
            else:
                return "하위 50% - 작은 실천으로 더 큰 변화를! 🌎"

        df['eco_category'] = df.apply(lambda x: categorize(x['percentile_rank'], x['total_mission_count']), axis=1)

        # 결과 반환 (JSON)
        result = df[[
            'user_id', 'total_mission_count', 'total_weekly_carbon_reduction',
            'predicted_carbon_reduction', 'percentile_rank', 'eco_category'
        ]]

        return result.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
