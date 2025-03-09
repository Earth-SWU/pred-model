from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
import uvicorn

app = FastAPI()

# ✅ 기존 데이터 로드
data_path = "final_fully_adjusted_data.csv"
data = pd.read_csv(data_path)

# ✅ total_mission_count 생성 (user_id별 미션 수행 횟수)
data['total_mission_count'] = data.groupby('user_id')['mission_id'].transform('count')

# ✅ total_clicks 생성 (user_id별 버튼 클릭 횟수)
data['total_clicks'] = data[data['activity_type_x'] == 'Button Click'].groupby('user_id')['activity_type_x'].transform('count')
data['total_clicks'].fillna(0, inplace=True)  # 클릭 없는 유저는 0으로 설정

# ✅ 모델 학습
X = data[['total_mission_count', 'total_clicks']]
y = data['carbon_reduction']  # 탄소 절감량 예측

rf_model = RandomForestRegressor(n_estimators=30, max_depth=7, min_samples_split=10, random_state=42)
rf_model.fit(X, y)

# ✅ 요청 받을 데이터 모델 정의
class UserMissionData(BaseModel):
    user_id: int
    total_mission_count: int
    total_clicks: int

# ✅ 예측 API 엔드포인트
@app.post("/predict/")
async def predict_carbon(data: UserMissionData):
    try:
        input_data = np.array([[data.total_mission_count, data.total_clicks]])
        predicted_carbon = rf_model.predict(input_data)[0]

        return {
            "user_id": data.user_id,
            "predicted_carbon_reduction": predicted_carbon,
            "message": "환경 기여도 예측 성공!"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ✅ FastAPI 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)