import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# データの準備（仮想データの生成）
np.random.seed(42)
data_size = 1000

# 1. メーカーリストと出現確率の定義
MAKER_LIST = ['トヨタ', 'ホンダ', '日産', 'BMW', 'マツダ', 'スバル', 'メルセデス', 'アウディ']
PROBABILITIES = [0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.2, 0.15] # 合計は 1.0

# 特徴量（説明変数）のリスト
data = {
    '走行距離_km': np.random.randint(10000, 150000, data_size),
    '年式': np.random.randint(2015, 2025, data_size),
    'メーカー': np.random.choice(MAKER_LIST, data_size, p=PROBABILITIES), 
    '状態_評価': np.random.randint(1, 6, data_size), # 5が最高
}
df = pd.DataFrame(data)

# 価格（目的変数）の生成ロジック
df['価格'] = (
    2500000  # 基本価格
    - (df['走行距離_km'] * 8) 
    - ((2025 - df['年式']) * 150000)
    + df['状態_評価'] * 50000
    + df['メーカー'].apply(lambda x: 
        500000 if x in ['BMW', 'メルセデス', 'アウディ'] else 
        150000 if x == 'トヨタ' else
        50000 if x in ['ホンダ', 'スバル'] else 
        0) 
    + np.random.randn(data_size) * 100000 # ノイズ
).clip(lower=100000) 

# 特徴量と目的変数の設定
X = df[['走行距離_km', '年式', 'メーカー', '状態_評価']] 
y = df['価格']

# 前処理パイプラインの作成
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['メーカー'])
    ],
    remainder='passthrough' # メーカー以外の列をそのまま通過させる
)

# モデルパイプラインの作成（RandomForestRegressorを使用）
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

# モデルの学習
model_pipeline.fit(X, y)

# 💡 NEW: 特徴量の名前と重要度を取得
# モデルの特徴量重要度と、前処理後の特徴量の名前を結合
feature_names = model_pipeline['preprocessor'].get_feature_names_out()
importances = model_pipeline['regressor'].feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})


# モデルと特徴量重要度データを joblib 形式で保存
joblib.dump(model_pipeline, 'car_price_predictor_model.joblib')
joblib.dump(feature_importance_df, 'feature_importance.joblib') # NEW FILE
print("モデル 'car_price_predictor_model.joblib' を保存しました。")
print("特徴量重要度データ 'feature_importance.joblib' を保存しました。")