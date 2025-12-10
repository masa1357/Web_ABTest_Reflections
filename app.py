import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


import hashlib
import random
import pandas as pd
import streamlit as st

# ===== パスと定数 =====
RESULTS_PATH = Path("results/abtest_results.jsonl")
BASELINE_PATH = Path("data/phase1_responses_to15step_20251203.json")
ADVICE_PATH = Path("data/phase2_advice_to10step_20251210.json")
MAX_ITEMS = 10
RATING_SCALE = [
    "A が強く良い",
    "A がやや良い",
    "どちらとも言えない",
    "B がやや良い",
    "B が強く良い",
]


# ===== データロード =====
def load_json_dict(path: Path) -> Dict[str, Any]:
    """JSONファイルを辞書としてロードする。"""
    if not path.exists():
        st.error(f"データファイルが見つかりません: {path}")
        st.stop()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_items(max_items: int = MAX_ITEMS) -> pd.DataFrame:
    """baseline応答とstudent adviceをペアにしたDataFrameを返す。"""
    baseline = load_json_dict(BASELINE_PATH)
    advice = load_json_dict(ADVICE_PATH)

    common_ids = sorted(set(baseline.keys()) & set(advice.keys()))
    if not common_ids:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for idx, user_key in enumerate(common_ids[:max_items]):
        base_entry = baseline[user_key]
        advice_entry = advice[user_key]
        records.append(
            {
                "item_index": idx,
                "source_userid": base_entry.get("userid", user_key),
                "baseline_grade": base_entry.get("grade", ""),
                "baseline_response": base_entry.get("response", ""),
                "student_advice_title": advice_entry.get("student_advice_title", ""),
                "student_advice_body": advice_entry.get("student_advice_body", ""),
                "student_grade": advice_entry.get("grade", ""),
            }
        )

    df = pd.DataFrame(records).sort_values("item_index").reset_index(drop=True)
    return df


# ===== 結果の読み書き =====
def load_answered_indices(user_id: str) -> set[int]:
    """すでに回答済みの item_index の集合を返す。"""
    if not RESULTS_PATH.exists():
        return set()

    df = pd.read_json(RESULTS_PATH, lines=True)
    df_user = df[df["user_id"] == user_id]
    return set(df_user["item_index"].tolist())


def save_response(record: Dict[str, Any]) -> None:
    """回答1件を JSON Lines 形式で追記保存する。"""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_next_index(all_indices: List[int], answered: set[int]) -> int | None:
    """未回答の item_index を小さい順に1つ返す。"""
    for idx in all_indices:
        if idx not in answered:
            return idx
    return None


# ===== Streamlit アプリ本体 =====
def main():
    st.set_page_config(page_title="Feedback A/B Test", layout="wide")

    df_items = load_items()
    if df_items.empty:
        st.error("比較対象となるデータが見つかりません。dataフォルダのJSONを確認してください。")
        return

    page = st.sidebar.selectbox("ページ選択", ["説明", "評価画面"])

    if page == "説明":
        st.title("A/Bテスト評価アプリ")
        st.markdown(
            f"""
このアプリでは、「情報科学」を受講した学生に対するフィードバックを比較評価していただきます。

- 評価するフィードバックは {len(df_items)} 件ございます。
- 回答前に簡単なパーソナリティ設問へ回答してもらいます。
- 各サンプルについて6項目（有用性/可読性/説得力/行動可能性/ハルシネーション/総合評価）で、A・Bいずれが優れているか、もしくは同程度かを選択いただきます。
- {len(df_items)} 件すべて回答すると終了です。同じ名前で再アクセスすれば途中から再開できます。

"""
        )
        st.info("説明は以上です。サイドバーの「ページ選択」から「評価画面」を選択して、評価を開始してください。")
        return

    st.title("フィードバック比較評価")
    st.markdown("同じ名前でアクセスすると、途中から評価を再開できます。")

    user_id = st.text_input("任意の名前を記入してください...", value="", placeholder="例: worker001 / student_A など").strip()
    if not user_id:
        st.warning("名前を入力すると評価を開始できます。")
        st.stop()

    # 参加者IDが変わったら状態を初期化
    if st.session_state.get("current_user_id") != user_id:
        st.session_state.current_user_id = user_id
        st.session_state.current_index = None
        st.session_state.survey_answers = None
        for widget_key in ("kyushu_student", "info_course_taken", "info_grade_text"):
            st.session_state.pop(widget_key, None)

    st.subheader("パーソナリティ質問")
    kyushu_options = ["-- 選択してください --", "はい", "いいえ"]
    kyushu_student = st.selectbox("九州大学の学生ですか？", kyushu_options, index=0, key="kyushu_student")

    info_course_options = ["-- 選択してください --", "はい", "いいえ"]
    info_course_taken = st.selectbox(
        "情報科学の講義を受講したことがありますか？",
        info_course_options,
        index=0,
        key="info_course_taken",
    )
    info_course_grade = st.selectbox(
        "受講していたときの成績（任意）",
        options=["未回答", "A", "B", "C", "D", "F"],
        index=0,
        key="info_grade_text",
    )

    if kyushu_student == kyushu_options[0] or info_course_taken == info_course_options[0]:
        st.warning("必須のアンケート項目に回答してください。")
        st.stop()

    survey_answers = {
        "kyushu_student": kyushu_student,
        "info_course_taken": info_course_taken,
        "info_course_grade": info_course_grade,
    }
    st.session_state.survey_answers = survey_answers

    answered = load_answered_indices(user_id)
    all_indices = list(df_items["item_index"])
    if st.session_state.get("current_index") is None:
        st.session_state.current_index = get_next_index(all_indices, answered)

    current_index = st.session_state.current_index
    if current_index is None:
        st.success("この参加者IDでは10件すべての比較が完了しています。ご協力ありがとうございました。")
        st.stop()

    row = df_items.loc[df_items["item_index"] == current_index].iloc[0]
    st.markdown("---")
    st.subheader(f"サンプル {current_index + 1} / {len(df_items)}")
    st.caption(f"{len(answered)} 件回答済み / 全 {len(df_items)} 件")

    # 表示位置をユーザーID + item_index で決定（再現性あり）
    seed = int.from_bytes(
        hashlib.sha256(f"{user_id}_{current_index}".encode("utf-8")).digest()[:8], "big"
    )
    rng = random.Random(seed)
    baseline_on_left = rng.choice([True, False])

    # 左右の表示内容（A=baseline, B=student）
    if baseline_on_left:
        left_title = "アドバイスA"
        left_content = row["baseline_response"]
        right_title = "アドバイスB"
        right_content = f"**{row['student_advice_title']}**\n\n{row['student_advice_body']}"
        # 選択肢は A が左 → RATING_SCALE のまま
        local_options = RATING_SCALE
    else:
        left_title = "アドバイスA"
        left_content = f"**{row['student_advice_title']}**\n\n{row['student_advice_body']}"
        right_title = "アドバイスB"
        right_content = row["baseline_response"]
        # 選択肢を左が表示B（学生）になるように入れ替え
        local_options = RATING_SCALE[::-1]
        
    content_col, flow_col = st.columns([3, 2])

    with content_col:
        st.markdown("#### フィードバック本文")
        st.markdown(f"###### 学生の成績: {row['baseline_grade']}")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(f"##### {left_title}")
            st.write(left_content)
        with col_right:
            st.markdown(f"##### {right_title}")
            # right_content はすでにフォーマット済みの文字列
            st.markdown(right_content)

    with flow_col:
        st.markdown("#### 評価")

        questions = [
            (
                "usefulness",
                "ステップ1：有用性",
                "あなたが学生だった場合、どちらのフィードバックが役に立つと思いますか？",
            ),
            ("readability", "ステップ2：可読性", "どちらが読みやすいと感じますか？"),
            ("persuasiveness", "ステップ3：説得力", "どちらの根拠が明確だと思いますか？"),
            ("actionability", "ステップ4：行動可能性", "成績向上のための行動が明確に示されていますか？"),
            ("hallucination", "ステップ5：ハルシネーション評価", "データに基づく回答に見えますか？"),
            ("overall", "ステップ6：総合評価", "総合的に見て、どちらが良いと思いますか？"),
        ]

        participant_key = str(abs(hash(user_id)))
        responses: Dict[str, str] = {}
        for field, title, description in questions:
            st.markdown(f"**{title}**")
            # 左: "Aが良い" 、中央: 5つの○/●ボタン、右: "Bが良い"
            key = f"{field}_{current_index}_{participant_key}"
            if key not in st.session_state:
                # デフォルトは中央（インデックス2 = 3番目）
                st.session_state[key] = RATING_SCALE[2]

            # カラム配置: [label, btn1, btn2, btn3, btn4, btn5, label]
            cols = st.columns([1, 1, 1, 1, 1, 1, 1])
            cols[0].markdown("**Aが良い**", unsafe_allow_html=True)

            # 各ボタンの表示シンボル（外側は大きめ、中央は小さめ）
            for i in range(5):
                col = cols[i + 1]
                option = RATING_SCALE[i]
                selected = st.session_state.get(key) == option

                # シンボル選択（見た目の差を Unicode で調整）
                sym = "⬤" if selected else "〇"
                btn_key = f"btn_{key}_{i}"

                # コールバックを作成（閉包の値を固定）
                def _make_cb(k=key, opt=option):
                    def _cb():
                        st.session_state[k] = opt
                    return _cb

                # ボタン表示（ラベルはシンボルのみ）
                col.button(sym, key=btn_key, on_click=_make_cb())

            cols[6].markdown("**Bが良い**", unsafe_allow_html=True)

            responses[field] = st.session_state.get(key, RATING_SCALE[2])
            st.divider()

        st.markdown("**ステップ7：コメント（任意）**")
        comment = st.text_area(
            "各解答について、改善点や感想があればご記入ください。",
            height=120,
            key=f"comment_{current_index}_{participant_key}",
        )

        if st.button("評価を保存して次へ ▶", use_container_width=True):
            record: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "item_index": int(row["item_index"]),
                "source_userid": row["source_userid"],
                "baseline_grade": row["baseline_grade"],
                "student_grade": row["student_grade"],
                "baseline_response": row["baseline_response"],
                "student_advice_title": row["student_advice_title"],
                "student_advice_body": row["student_advice_body"],
                "kyushu_student": survey_answers["kyushu_student"],
                "info_course_taken": survey_answers["info_course_taken"],
                "info_course_grade": survey_answers["info_course_grade"],
                "comment": comment,
            }
            for field, _title, _desc in questions:
                record[f"q_{field}"] = responses[field]

            save_response(record)
            st.success("保存しました。")

            answered = load_answered_indices(user_id)
            next_idx = get_next_index(all_indices, answered)
            st.session_state.current_index = next_idx
            st.rerun()


if __name__ == "__main__":
    main()
