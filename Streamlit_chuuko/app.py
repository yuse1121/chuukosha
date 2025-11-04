import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # グラフ描画ライブラリ

# ========== 協調フィルタリング用データと準備 ==========
MAKER_OPTIONS = ['トヨタ', 'ホンダ', '日産', 'BMW', 'マツダ', 'スバル', 'メルセデス', 'アウディ', 'その他'] 

# 表示用と内部処理用のマッピング
MAKER_MAPPING = {
    'トヨタ': 'toyota', 'ホンダ': 'honda', '日産': 'nissan', 'BMW': 'bmw', 'マツダ': 'mazda', 'スバル': 'subaru',
    'メルセデス': 'mercedes-benz', 'アウディ': 'audi',
}
# selectboxに表示するオプションリスト (例: トヨタ (toyota))
DISPLAY_OPTIONS = [f"{jp} ({MAKER_MAPPING[jp]})" for jp in MAKER_OPTIONS if jp != 'その他'] + ['その他']

# 仮想のユーザー興味データ (日本語メーカー名を使用)
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
# ⚠️ Streamlit Cloudでモデルファイルが見つからない問題に対処するため、パスを直接指定
try:
    # ⚠️ 修正: リポジトリ名 (Streamlit_chuuko) をフォルダ名としてパスに含める
    
    # Cloud環境のパス: /mount/src/chuukosha/Streamlit_chuuko/
    BASE_REPO_FOLDER = "Streamlit_chuuko/"
    
    model_pipeline = joblib.load(BASE_REPO_FOLDER + 'car_price_predictor_model.joblib')
    feature_importance_df = joblib.load(BASE_REPO_FOLDER + 'feature_importance.joblib') 

except FileNotFoundError:
    st.error("モデルまたは特徴量ファイルが見つかりません。train_model.pyを再実行してください。")
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
    # 選択された文字列から、モデルが必要とする日本語メーカー名（例: 'トヨタ'）を抽出
    maker_display = st.selectbox(
        'メーカー',
        options=DISPLAY_OPTIONS
    )
    # 内部処理用のキーを抽出 (例: 'トヨタ (toyota)' -> 'トヨタ')
    maker = maker_display.split(' ')[0] 


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
    
    # ユーザー入力をDataFrameに格納 
    input_data = pd.DataFrame({
        '走行距離_km': [mileage],
        '年式': [year],
        'メーカー': [maker], # 日本語メーカー名がモデルのキーとして使われます
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


        # --- (B) 価格の妥当性評価 ---
        st.markdown("---")
        st.subheader("💰 価格の妥当性評価")
        
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
        target_maker = maker 
        
        if target_maker in interest_df.columns:
            correlations = interest_df.corrwith(interest_df[target_maker]).sort_values(ascending=False)
            
            recommendations = correlations.drop(target_maker, errors='ignore').dropna()
            recommendations = recommendations.drop('その他', errors='ignore')
            
            top_recommendations = recommendations.head(3)
            
            if top_recommendations.empty:
                st.info("推薦できる他のメーカー情報がありません。")
            else:
                target_maker_jp = target_maker # 日本語名
                st.info(f"この **{target_maker_jp}** に興味を持つユーザーは、以下のメーカーにも関心を持っています。")
                
                rec_list = []
                for rank, (rec_maker_jp, score) in enumerate(top_recommendations.items(), 1):
                    
                    if score > 0.8: intensity = "非常に強い関心"
                    elif score > 0.4: intensity = "強い関心"
                    elif score > 0: intensity = "一般的な関心"
                    else: intensity = "低い関心 (対立傾向)"

                    rec_list.append(f"{rank}. **{rec_maker_jp}** (関心度: {score:.2f} - {intensity})")
                
                st.markdown('\n'.join(rec_list))

        else:
            st.warning("このメーカーの推薦データは現在不足しています。")


        # --- (D) 特徴量の重要度グラフの表示 ---
        st.markdown("---")
        st.subheader("📊 予測への貢献度 (特徴量重要度)")
        
        df_plot = feature_importance_df.copy()
        
        # ⚠️ 日本語ラベルを英語ラベルにマッピング (文字化け回避)
        FEATURE_LABEL_MAPPING_EN = {
            '走行距離_km': 'Mileage (km)',
            '年式': 'Year',
            '状態_評価': 'Condition Score',
        }

# 'feature_clean' 列を生成し、不要なプレフィックスや日本語を削除・変換
        df_plot['feature_clean'] = df_plot['feature'].apply(lambda x: 
    # 'remainder__走行距離_km' -> 'Mileage (km)' に変換
            FEATURE_LABEL_MAPPING_EN.get(x.replace('remainder__', ''), 
        # 'cat__トヨタ' -> 'TOYOTA' に変換
            x.replace('cat__', '').upper() if x.startswith('cat__') else 
            x # その他の場合はそのまま
        )
    )

# Top 5を可視化
        df_plot = df_plot.sort_values('importance', ascending=False).head(5)

# グラフ描画 (日本語フォント設定は不要)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='importance', y='feature_clean', data=df_plot, ax=ax, palette='viridis')

# 英語ラベルを設定
        ax.set_title('Top Features Influencing Price', fontsize=14)
        ax.set_xlabel('Importance (%)')
        ax.set_ylabel('') # Y軸のFeatureラベルは不要

# Y軸の目盛りラベルは df_plot['feature_clean'] の値（きれいな英語名）を使用
        ax.set_yticklabels(df_plot['feature_clean'].tolist())
        ax.tick_params(axis='y', labelsize=10) # サイズ調整

        st.pyplot(fig)


    except Exception as e:
        st.error(f"予測または推薦処理中にエラーが発生しました。エラー: {e}")

st.markdown("---")


