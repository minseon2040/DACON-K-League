# ⚽ K리그 경기 내 최종 패스 좌표 예측 AI 모델 개발


## Project Overview
- 선수 트래킹 데이터 기반 최종 좌표 예측 (Regression Task)
- 기간: 2025.12.01 ~ 2026.01.12
- 팀 프로젝트 (3인)

## Modeling
- CatBoost, LightGBM 기반 tabular modeling
- 비선형 spatial pattern 학습을 위한 MLP 추가
- 거리 구간을 나눈 후 weighted ensemble 적용
- Cross Validation 기반 하이퍼파라미터 튜닝

## Results
- Evaluation Metric: Euclidean Distance  
- Public Rank: 47th  
- Private Rank: 22nd  
- **Final Standing: 22nd out of 937 teams (Top 4%)**

https://dacon.io/competitions/official/236647/overview/description
