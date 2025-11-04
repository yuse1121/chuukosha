import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.font_manager as fm # フォントマネージャーをインポート


# ... (中略: 協調フィルタリングデータ準備、モデル読み込み、UI設定の部分は変更なし) ...


# 4. 予測ボタンと処理
if st.button('価格を予測する & 関連車種を推薦する', type='primary'):
    
    # ユーザー入力処理 (変更なし)
    maker = maker_display.split(' ')[0]
    input_data = pd.DataFrame({
        '走行距離_km': [mileage],
        '年式': [year],
        'メーカー': [maker], 
        '状態_評価': [condition], 
    })
    
    # ⬇️ メインの try ブロック開始
    try:
        # --- (A) 回帰分析：予測の実行 ---
        predicted_price = model_pipeline.predict(input_data)[0]
        # ... (予測価格の表示ロジックは変更なし) ...

        # --- (B) 価格の妥当性評価 ---
        # ... (妥当性評価ロジックは変更なし) ...

        # --- (C) 協調フィルタリング：推薦の実行 ---
        # ... (推薦ロジックは変更なし) ...


        # --- (D) 特徴量の重要度グラフの表示 ---
        st.markdown("---")
        st.subheader("📊 予測への貢献度 (特徴量重要度)")
        
        df_plot = feature_importance_df.copy()
        
        # ⚠️ NEW: 特徴量ラベルの英語マッピングを再確認
        FEATURE_LABEL_MAPPING_EN = {
            '走行距離_km': 'Mileage (km)',
            '年式': 'Year',
            '状態_評価': 'Condition Score',
        }

        # 'feature_clean' 列を生成し、不要なプレフィックスや日本語を削除・変換
        df_plot['feature_clean'] = df_plot['feature'].apply(lambda x: 
            # 1. 'remainder__年式' -> 'Year' / 'remainder__走行距離_km' -> 'Mileage (km)' に変換
            clean_name = x.replace('remainder__', '')
            
            # 2. FEATURE_LABEL_MAPPING_ENで変換
            if clean_name in FEATURE_LABEL_MAPPING_EN:
                return FEATURE_LABEL_MAPPING_EN[clean_name]
            
            # 3. 'cat__トヨタ' -> 'TOYOTA' に変換
            elif clean_name.startswith('cat__'):
                # 日本語メーカー名を英語大文字に変換 (e.g., 'cat__トヨタ' -> 'TOYOTA')
                jp_name = clean_name.replace('cat__', '')
                # MAKER_MAPPINGから逆引きするロジックは複雑なため、ここは手動で英語に変換するシンプルな方法に修正
                return jp_name.upper() 
            
            # 4. それ以外はそのまま (エラー回避)
            else:
                return x
        )

        # Top 5を可視化
        df_plot = df_plot.sort_values('importance', ascending=False).head(5)

        # グラフ描画
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='importance', y='feature_clean', data=df_plot, ax=ax, palette='viridis')

        # 英語ラベルを設定
        ax.set_title('Top Features Influencing Price', fontsize=14)
        ax.set_xlabel('Importance (%)')
        ax.set_ylabel('') # Y軸のFeatureラベルは不要

        # Y軸の目盛りラベルはクリーンな英語名を使用
        ax.set_yticklabels(df_plot['feature_clean'].tolist())
        ax.tick_params(axis='y', labelsize=10) 

        st.pyplot(fig)


    # ⬆️ try ブロックがここで終わり、次に except が来る
    except Exception as e:
        st.error(f"予測または推薦処理中にエラーが発生しました。エラー: {e}")
        # ⬆️ ここで except が正しく閉じられている


st.markdown("---")

# ⬇️ ファイルの最後の部分でエラーが出たコードは削除されています。
