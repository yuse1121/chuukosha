import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 特徴量重要度グラフ用

plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['axes.unicode_minus'] = False

# ========== 協調フィルタリング用データと準備 ==========
MAKER_OPTIONS = ['トヨタ', 'ホンダ', '日産', 'BMW', 'マツダ', 'スバル', 'メルセデス', 'アウディ', 'その他'] 

# 日本語表示用マッピング
MAKER_MAPPING = {
    'トヨタ': 'トヨタ', 'ホンダ': 'ホンダ', '日産': '日産', 'BMW': 'BMW', 'マツダ': 'マツダ', 'スバル': 'スバル',
    'メルセデス': 'メルセデス・ベンツ', 'アウディ': 'アウディ',
}
# selectboxに表示するオプションリスト (例: トヨタ (toyota))
DISPLAY_OPTIONS = [f"{eng} ({MAKER_MAPPING[eng]})" for eng in MAKER_OPTIONS if eng != 'その他'] + ['その他']

# 仮想のユーザー興味データ
recommendation_data = {
    'トヨタ': {'UserA': 5, 'UserB': 1, 'UserC': 4, 'UserD': 5, 'UserE': 2},
    'ホンダ': {'UserA': 4, 'UserB': 5, 'UserC': 2, 'UserD': 4, 'UserE': 5},
    '日産': {'UserA': 1, 'UserB': 3, 'UserC': 5, 'UserD': 2, 'UserE': 4},
    'BMW': {'UserA': 5, 'UserB': 1, 'UserC': 5, 'UserD': 5, 'UserE': 1},
    'マツダ': {'UserA': 2, 'UserB': 4, 'UserC': 3, 'UserD': 1, 'UserE': 5},
    'スバル': {'UserA': 3, 'UserB': 5, 'UserC': 3, 'UserD': 3, 'UserE': 4},
    'メルセデス': {'UserA': 5, 'UserB': 2, 'UserC': 5, 'UserD': 4, 'UserE': 1}, 
    'アウディ': {'UserA': 4, 'UserB': 1, 'UserC': 4, 'UserD': 5, 'UserE': 2},    
    'その他': {'UserA': 1, 'UserB': 1, 'UserC': 1, 'UserD': 1, 'UserE': 1},
}
interest_df = pd.DataFrame(recommendation_data).fillna(0)
# ==========================================================


# 1. モデルと重要度データの読み込み
try:
    # ⚠️ Streamlit Cloudの環境パスに合わせて読み込みパスを指定します
    BASE_PATH = "Streamlit_chuuko/"  # あなたのリポジトリのルートフォルダ名 (画像から確認)
    
    model_pipeline = joblib.load(BASE_PATH + 'car_price_predictor_model.joblib')
    feature_importance_df = joblib.load(BASE_PATH + 'feature_importance.joblib') 

except FileNotFoundError:
    st.error("【最終エラー】モデルファイルが見つかりません。パスを確認してください。")
    st.stop()
except Exception as e:
    st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
    st.stop()

# 2. アプリのレイアウト設定
st.title("🚗 中古車価格予測・推薦アプリ")
st.markdown("### 回帰分析と協調フィルタリングのデモ")
st.markdown("---")

# 3. ユーザー入力エリア
st.header("予測条件の入力")

col1, col2 = st.columns(2)

with col1:
    # ⚠️ 修正箇所: ここで maker_display に結果を格納
    maker_display = st.selectbox(
        'メーカー',
        options=DISPLAY_OPTIONS
    )

    current_year = 2025
    year_options = list(range(2015, current_year + 1))
    year = st.selectbox(
        '年式 (製造年)',
        options=sorted(year_options, reverse=True)
    )

with col2:
    mileage = st.number_input(
        '走行距離 (km)',
        min_value=1000,
        max_value=300000,
        value=50000,
        step=5000,
        help="1,000 kmから300,000 kmの範囲で入力してください。"
    )

    condition = st.slider(
        '商品の状態評価 (1:悪い ~ 5:最高)',
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )


st.markdown("---")

# 4. 予測ボタンと処理
if st.button('価格を予測する & 関連車種を推薦する', type='primary'):
    
    # ⚠️ 修正箇所: maker_display からモデルが必要とする英語/日本語名（キー）を抽出
    if '(' in maker_display:
        # 例: "トヨタ (toyota)" -> "トヨタ" を抽出
        maker = maker_display.split(' ')[0]
    else:
        # 例: "その他" の場合は "その他" を使用
        maker = maker_display

    # ユーザー入力をDataFrameに格納 
    input_data = pd.DataFrame({
        '走行距離_km': [mileage],
        '年式': [year],
        'メーカー': [maker], # モデルが学習したキー（例: 'トヨタ'）を使用
        '状態_評価': [condition], 
    })
    
    try:
        # --- (A) 回帰分析：予測の実行 ---
        predicted_price = model_pipeline.predict(input_data)[0]
        
        # 予測価格の表示
        st.subheader("✅ 予測価格 (回帰分析)")
        formatted_price = f"¥{int(round(predicted_price, -3)):,}" 
        st.success(f"## 予測される販売価格は **{formatted_price}** です")
        st.caption("※ 予測は仮想データで学習したランダムフォレストモデルの結果です。")


        # --- (B) NEW: 価格の妥当性評価 ---
        st.markdown("---")
        st.subheader("💰 価格の妥当性評価")
        
        # 妥当な基準価格を計算 (年式と状態が良いほど高くなる単純ロジック)
        base_value = (year - 2015) * 50000 + condition * 10000 
        
        if predicted_price > (1.2 * base_value):
            st.warning("この価格は、同条件の市場平均より**かなり高め**かもしれません。")
        elif predicted_price > (1.05 * base_value):
            st.info("この価格は、市場平均より**やや高め**です。")
        elif predicted_price < (0.8 * base_value):
            st.info("この価格は、同条件の市場平均より**割安**かもしれません。")
        else:
            st.success("この価格は、**市場価値として妥当な範囲**です。")
        st.markdown("---")


        # --- (C) 協調フィルタリング：推薦の実行 ---
        st.subheader("👥 関連車種の推薦 (協調フィルタリング)")
        target_maker = maker # 抽出したメーカー名を使用
        
        if target_maker in interest_df.columns:
            # ターゲット車種と他の車種との相関（類似度）を計算
            correlations = interest_df.corrwith(interest_df[target_maker]).sort_values(ascending=False)
            
            # ターゲット車種自身と、'その他'、NaN（データ不足）を除外
            recommendations = correlations.drop(target_maker, errors='ignore').dropna()
            recommendations = recommendations.drop('その他', errors='ignore')
            
            top_recommendations = recommendations.head(3)
            
            if top_recommendations.empty:
                st.info("推薦できる他のメーカー情報がありません。")
            else:
                target_maker_jp = MAKER_MAPPING.get(target_maker, target_maker)
                st.info(f"この **{target_maker_jp}** に興味を持つユーザーは、以下のメーカーにも関心を持っています。")
                
                rec_list = []
                for rank, (rec_maker_eng, score) in enumerate(top_recommendations.items(), 1):
                    rec_maker_jp = MAKER_MAPPING.get(rec_maker_eng, rec_maker_eng)
                    
                    if score > 0.8:
                        intensity = "非常に強い関心"
                    elif score > 0.4:
                        intensity = "強い関心"
                    elif score > 0:
                        intensity = "一般的な関心"
                    else:
                        intensity = "低い関心 (対立傾向)"

                    rec_list.append(f"{rank}. **{rec_maker_jp}** (関心度: {score:.2f} - {intensity})")
                
                st.markdown('\n'.join(rec_list))

        else:
            st.warning("このメーカーの推薦データは現在不足しています。")


        # --- (D) NEW: 特徴量の重要度グラフの表示 ---
        st.markdown("---")
        st.subheader("📊 予測への貢献度 (特徴量重要度)")
        
        df_plot = feature_importance_df.copy()
        
        # 影響度の低いOne-Hot Encodingされたメーカーの列を除外して、トップ5を表示
        df_plot['feature_clean'] = df_plot['feature'].apply(lambda x: x.split('__')[1] if '__' in x else x)
        
        # Top 5を可視化
        df_plot = df_plot.sort_values('importance', ascending=False).head(5)
        
        # グラフの描画
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='importance', y='feature_clean', data=df_plot, ax=ax, palette='viridis')

# ⚠️ 修正・確認箇所: 日本語のラベルが正しく渡されているか確認
# plt.rcParams['font.family'] = 'IPAGothic' が適用されている前提で

        ax.set_title('予測に影響を与えた上位の特徴量', fontsize=14) 
        ax.set_xlabel('重要度 (%)') # 日本語
        ax.set_ylabel('')
        st.pyplot(fig)


    except Exception as e:
        st.error(f"予測または推薦処理中にエラーが発生しました。エラー: {e}")


st.markdown("---")

