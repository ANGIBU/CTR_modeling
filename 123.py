import pandas as pd
from sklearn.model_selection import KFold
from training import train_model  # 가정: 모델 학습 함수

# 학습 데이터 불러오기
train_df = pd.read_parquet('data/train.parquet')

# K-Fold 설정 (예: 5개 폴드)
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# 폴드별 학습 및 검증
for fold, (train_index, val_index) in enumerate(kf.split(train_df)):
    print(f"--- Fold {fold+1}/{N_SPLITS} 학습 시작 ---")
    
    # 데이터 분할
    X_train, X_val = train_df.iloc[train_index], train_df.iloc[val_index]
    y_train, y_val = train_df['target'].iloc[train_index], train_df['target'].iloc[val_index]
    
    # 모델 학습 함수 호출 (train_model 함수는 이미 작성되었다고 가정)
    model = train_model(X_train, y_train, X_val, y_val)
    
    # 성능 평가 (evaluation.py의 함수 사용)
    # evaluation_result = evaluate_model(model, X_val, y_val)
    # print(f"Fold {fold+1} 성능: {evaluation_result}")
    
    # 단일 폴드 테스트를 위해 첫 번째 폴드만 실행 후 중단
    # if fold == 0:
    #     break