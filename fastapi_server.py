from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

# ✅ 모델 학습 (임시로 간단한 데이터 사용)
rf_model = RandomForestRegressor(n_estimators=30, max_depth=7, min_samples_split=10, random_state=42)
X_train = np.array([[10, 20], [30, 40], [50, 60]])
y_train = np.array([0.1, 0.5, 0.9])
rf_model.fit(X_train, y_train)

# ✅ 요청 받을 데이터 모델 정의
class UserMissionData(BaseModel):
    user_id: int
    total_mission_count: int
    total_clicks: int

# ✅ 임시 데이터베이스 (user_id에 대한 추가 정보)
USER_DATA = {
    300: {
        "name_x": "텀블러 사용하기",
        "carbon_reduction": 0.3,
        "total_weekly_carbon_reduction": 86.4,
        "total_mission_count": 756,
        "total_clicks": 756.0,
        "percentile_rank": 77.80717225161669,
        "eco_category": "상위 25% - 훌륭한 에코스텝러! 🌱",
        "age_group": "20대 후반",
        "region": "광주",
        "group_avg_carbon_reduction": 58.64651162790715,
        "carbon_reduction_diff": 27.753488372093294,
        "personal_feedback": "당신은 같은 연령대(20대 후반) 광주 지역 사용자들보다 더 많은 환경 기여를 하고 있어요! 🎉"
    }
}

# ✅ 예측 API 엔드포인트 (예측값 + 추가 데이터 반환)
@app.post("/predict/")
async def predict_carbon(data_input: UserMissionData):
    try:
        # ✅ user_id에 해당하는 추가 데이터 찾기
        user_info = USER_DATA.get(data_input.user_id)
        
        if user_info is None:
            raise HTTPException(status_code=404, detail=f"User ID {data_input.user_id} not found")

        # ✅ 예측 수행
        input_data = np.array([[data_input.total_mission_count, data_input.total_clicks]])
        predicted_carbon = rf_model.predict(input_data)[0]

        # ✅ 응답 데이터 구성
        response = {
            "user_id": data_input.user_id,
            "name_x": user_info["name_x"],
            "carbon_reduction": user_info["carbon_reduction"],
            "total_weekly_carbon_reduction": user_info["total_weekly_carbon_reduction"],
            "total_mission_count": user_info["total_mission_count"],
            "total_clicks": user_info["total_clicks"],
            "predicted_carbon_reduction": predicted_carbon,
            "percentile_rank": user_info["percentile_rank"],
            "eco_category": user_info["eco_category"],
            "age_group": user_info["age_group"],
            "region": user_info["region"],
            "group_avg_carbon_reduction": user_info["group_avg_carbon_reduction"],
            "carbon_reduction_diff": user_info["carbon_reduction_diff"],
            "personal_feedback": user_info["personal_feedback"],
            "message": "환경 기여도 예측 성공!"
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ✅ FastAPI 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)