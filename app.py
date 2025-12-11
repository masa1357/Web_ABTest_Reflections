import json
import pandas as pd
import streamlit as st
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ===== パスと定数 =====
RESULTS_PATH = Path("results/abtest_results.jsonl")
BASELINE_PATH = Path("data/phase1_responses_to10step_20251211.json")
ADVICE_PATH = Path("data/phase2_advice_to10step_20251210.json")

# ABテスト対象とする userid のサンプル一覧（順序は後でシャッフルされる）
correct_uid = [
    'C-2021-2_U40',
    # 'C-2022-1_U75',
    # 'C-2021-1_U76',
    # 'C-2021-2_U88',
    # 'C-2021-2_U35',
    # 'C-2021-2_U148',
    'C-2021-2_U133',
    # 'C-2021-2_U76',
    # 'C-2022-1_U85',
    # 'C-2021-2_U57',
    # 'C-2021-2_U42',
    'C-2022-1_U80',
    # 'C-2021-2_U85',
    'C-2021-1_U46',
    # 'C-2021-1_U52',
    # 'C-2021-2_U113',
    'C-2021-2_U164',
    # 'C-2022-1_U86',
    # 'C-2021-2_U140',
    'C-2021-2_U81',
    # 'C-2021-1_U78',
    'C-2021-1_U66',
    'C-2021-2_U12',
    'C-2021-2_U82',
    # 'C-2021-2_U134',
    'C-2021-1_U17',
    # 'C-2021-2_U4',
    # 'C-2021-2_U91',
    # 'C-2021-2_U166',
    # 'C-2021-1_U21',
]

incorrect_uid = [
    'C-2021-2_U115',
    'C-2022-1_U52',
    'C-2021-1_U43',
    'C-2022-1_U6',
    'C-2022-1_U29',
    'C-2022-1_U61',
    'C-2021-1_U97',
    'C-2022-1_U92',
    'C-2021-2_U149',
    'C-2021-2_U1']

use_uid = correct_uid + incorrect_uid

# correct_uid をすべて出題するのが基本。部分出題したい場合は max_items を明示的に指定する。
MAX_ITEMS = len(use_uid)
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


# ...existing code...
def load_items(max_items: int = MAX_ITEMS, *, user_id: str | None = None) -> pd.DataFrame:
    """baseline応答とstudent adviceをペアにしたDataFrameを返す。
    - 基本は correct_uid に含まれる userid のみを対象とし、なければ共通集合を使う
    - user_id を指定した場合は、その文字列に基づく決定的な乱数シードで出題順をシャッフルする
    """
    baseline = load_json_dict(BASELINE_PATH)
    advice = load_json_dict(ADVICE_PATH)

    # ユーザID候補を収集して map: userid -> entry を作成
    def build_user_map(d: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        m: Dict[str, Dict[str, Any]] = {}
        for top_key, entry in d.items():
            # 候補: トップレベルキー、entry 内の userid / user_id
            candidates = [top_key]
            if isinstance(entry, dict):
                for fld in ("userid", "user_id"):
                    val = entry.get(fld)
                    if isinstance(val, str) and val:
                        candidates.append(val)
            # 先に見つかったものを優先して登録（重複 userid があれば最初のもの）
            for uid in candidates:
                if uid not in m:
                    m[uid] = entry
        return m

    base_map = build_user_map(baseline)
    advice_map = build_user_map(advice)

    # 共通の userid を取得（まずトップレベル同一キー集合を優先）
    top_keys_common = sorted(set(baseline.keys()) & set(advice.keys()))
    if top_keys_common:
        common_userids = top_keys_common
    else:
        # トップレベル一致がなければ entry 内 userid でマッチさせる
        common_userids = sorted(set(base_map.keys()) & set(advice_map.keys()))

    # use_uid にあるものを優先し、重複を排除
    prioritized_uids: List[str] = []
    if use_uid:
        for uid in use_uid:
            if uid in common_userids and uid not in prioritized_uids:
                prioritized_uids.append(uid)
    else:
        prioritized_uids = list(common_userids)

    if not prioritized_uids:
        return pd.DataFrame()

    items_order = prioritized_uids[:max_items]
    if user_id:
        seed = int.from_bytes(
            hashlib.sha256(f"order_{user_id}".encode("utf-8")).digest()[:8], "big"
        )
        rng = random.Random(seed)
        rng.shuffle(items_order)

    records: List[Dict[str, Any]] = []
    for idx, uid in enumerate(items_order):
        base_entry = base_map.get(uid, {})
        advice_entry = advice_map.get(uid, {})
        records.append(
            {
                "item_index": idx,
                "source_userid": base_entry.get(
                    "userid", base_entry.get("user_id", uid)
                ),
                "baseline_grade": base_entry.get("grade", ""),
                "baseline_response": base_entry.get("response", "")
                or base_entry.get("text", ""),
                "student_advice_title": advice_entry.get("student_advice_title", "")
                or advice_entry.get("title", ""),
                "student_advice_body": advice_entry.get("student_advice_body", "")
                or advice_entry.get("body", ""),
                "student_grade": advice_entry.get("grade", ""),
            }
        )

    df = pd.DataFrame(records).sort_values("item_index").reset_index(drop=True)
    return df


# ===== 結果の読み書き =====
def load_answered_indices(user_id: str) -> set[int]:
    """すでに回答済みの item_index の集合を返す。results の列名差異に頑健に対応する。"""
    if not RESULTS_PATH.exists():
        return set()

    try:
        df = pd.read_json(RESULTS_PATH, lines=True)
    except Exception:
        # JSONL 読み込みに失敗したら行ごとにパースして探索
        records = []
        try:
            with RESULTS_PATH.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return set()

        indices = set()
        ITEM_CANDIDATES = ("item_index", "item", "index", "itemIdx", "itemId")
        for rec in records:
            # user_id 相当フィールドが一致するか探す
            matched = False
            for k in (
                "user_id",
                "userid",
                "user",
                "worker_id",
                "participant",
                "source_userid",
            ):
                if k in rec and rec[k] == user_id:
                    matched = True
                    break
            if not matched:
                # 値レベルで一致を探す（稀なケース）
                for v in rec.values():
                    if isinstance(v, str) and v == user_id:
                        matched = True
                        break
            if not matched:
                continue
            # item index を探して追加
            for k in ITEM_CANDIDATES:
                if k in rec:
                    try:
                        indices.add(int(rec[k]))
                        break
                    except Exception:
                        continue
        return indices

    # DataFrame 読み込み成功時：候補列名を探す
    USER_CANDIDATES = [
        "user_id",
        "userid",
        "user",
        "worker_id",
        "participant",
        "source_userid",
    ]
    ITEM_CANDIDATES = ["item_index", "item", "index", "itemIdx", "itemId"]

    user_col = next((c for c in USER_CANDIDATES if c in df.columns), None)
    item_col = next((c for c in ITEM_CANDIDATES if c in df.columns), None)

    if user_col is not None and item_col is not None:
        try:
            df_user = df[df[user_col] == user_id]
            return set(int(x) for x in df_user[item_col].dropna().astype(int).tolist())
        except Exception:
            return set()

    # 列名が混在している場合はレコード単位で探索
    indices = set()
    for _, row in df.iterrows():
        matched = False
        if user_col is not None:
            val = row.get(user_col)
            if isinstance(val, str) and val == user_id:
                matched = True
        else:
            for c, val in row.items():
                if isinstance(val, str) and val == user_id:
                    matched = True
                    break
        if not matched:
            continue

        if item_col is not None:
            try:
                indices.add(int(row.get(item_col)))
                continue
            except Exception:
                pass
        for c in ITEM_CANDIDATES:
            if c in row.index:
                try:
                    indices.add(int(row[c]))
                    break
                except Exception:
                    continue
        try:
            if "item_index" in row.index and pd.notna(row["item_index"]):
                indices.add(int(row["item_index"]))
        except Exception:
            pass

    return indices


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
    st.set_page_config(
        page_title="Feedback A/B Test", layout="wide", initial_sidebar_state="collapsed"
    )

    df_preview = load_items()
    if df_preview.empty:
        st.error(
            "比較対象となるデータが見つかりません。dataフォルダのJSONを確認してください。"
        )
        return

    page = st.sidebar.selectbox("ページ選択", ["説明", "評価画面"])

    if page == "説明":
        st.title("A/Bテスト評価アプリ")
        st.markdown(
            f"""
このアプリでは、「情報科学」を受講した学生に対するフィードバックを比較評価していただきます。

- 評価するフィードバックは {len(df_preview)} 件あります。
- 所要時間は15~20分程度を想定しています。
- 回答前に簡単なパーソナリティ設問へ回答してもらいます。
- 各サンプルについて7項目（正確さ/可読性/説得力/行動可能性/ハルシネーション/有用性/総合評価）で、A・Bいずれが優れているか、もしくは同程度かを選択いただきます。
- {len(df_preview)} 件すべて回答すると終了です。同じ名前でアクセスすれば途中から再開できます。

"""
        )
        st.info(
            "説明は以上です。サイドバーの「ページ選択」から「評価画面」を選択して、評価を開始してください。"
        )
        return

    st.title("フィードバック比較評価")
    st.markdown("同じ名前でアクセスすると、途中から評価を再開できます。")

    user_id = st.text_input(
        "任意の名前を記入してください...",
        value="",
        placeholder="例: worker001 / student_A など",
    ).strip()
    # 赤文字で，名前を忘れると途中から再開できない旨を表示
    st.markdown(
        "<span style='color:red;'>注意! 名前を忘れると途中から再開できません!</span>", unsafe_allow_html=True
    ) 
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
    kyushu_student = st.selectbox(
        "九州大学の学生ですか？", kyushu_options, index=0, key="kyushu_student"
    )

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

    if (
        kyushu_student == kyushu_options[0]
        or info_course_taken == info_course_options[0]
    ):
        st.warning("必須のアンケート項目に回答してください。")
        st.stop()

    survey_answers = {
        "kyushu_student": kyushu_student,
        "info_course_taken": info_course_taken,
        "info_course_grade": info_course_grade,
    }
    st.session_state.survey_answers = survey_answers

    # 参加者ごとに決定的にシャッフルされた順序でアイテムを取得
    df_items = load_items(user_id=user_id)
    if df_items.empty:
        st.error(
            "正しくロードできる評価対象がありません。userid のリストや JSON の内容をご確認ください。"
        )
        st.stop()

    answered = load_answered_indices(user_id)
    all_indices = list(df_items["item_index"])
    if st.session_state.get("current_index") is None:
        st.session_state.current_index = get_next_index(all_indices, answered)

    current_index = st.session_state.current_index
    if current_index is None:
        st.success(
            f"この参加者IDでは全 {len(df_items)} 件の比較が完了しています。ご協力ありがとうございました。"
        )
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
        left_title = "フィードバックA"
        left_content = row["baseline_response"]
        right_title = "フィードバックB"
        right_content = (
            f"##### **{row['student_advice_title']}**\n\n{row['student_advice_body']}"
        )
        local_options = RATING_SCALE
    else:
        left_title = "フィードバックA"
        left_content = (
            f"**{row['student_advice_title']}**\n\n{row['student_advice_body']}"
        )
        right_title = "フィードバックB"
        right_content = row["baseline_response"]
        local_options = RATING_SCALE[::-1]

    content_col, flow_col = st.columns([3.5, 1.5])

    with content_col:
        st.markdown("#### フィードバック本文")
        st.markdown(f"###### 学生の成績: {row['baseline_grade']}")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(f"##### {left_title} \n====================")
            st.write(left_content)
        with col_right:
            st.markdown(f"##### {right_title} \n====================")
            st.markdown(right_content)

    with flow_col:
        st.markdown("#### 評価")

        questions = [
            (
                # 正確さ：学生の成績と発言内容の整合性を確認してもらう
                "accuracy",
                "ステップ1：正確さ",
                "学生の成績とフィードバック内の発言傾向は一致していると思いますか？",
            ),
            ("readability", "ステップ2：可読性", "どちらが読みやすいと感じますか？"),
            (
                "persuasiveness",
                "ステップ3：説得力",
                "どちらの根拠が明確だと思いますか？",
            ),
            (
                "actionability",
                "ステップ4：行動可能性",
                "成績向上のための行動が明確に示されていますか？",
            ),
            (
                "hallucination",
                "ステップ5：ハルシネーション評価",
                "データに基づく回答に見えますか？",
            ),
            (
                "usefulness",
                "ステップ6：有用性",
                "あなたが学生だった場合、どちらのフィードバックが役に立つと思いますか？",
            ),
            (
                "overall",
                "ステップ7：総合評価",
                "総合的に見て、どちらが良いと思いますか？",
            ),
        ]

        participant_key = str(abs(hash(user_id)))
        responses: Dict[str, str] = {}
        with st.container(height=800):
            with st.form(key=f"eval_form_{current_index}_{participant_key}"):
                for field, title, description in questions:
                    st.markdown(f"**{title}**")
                    st.caption(description)

                    # 左: "Aが良い" 、中央: 5つの○/●ボタン、右: "Bが良い"
                    key = f"{field}_{current_index}_{participant_key}"
                    if key not in st.session_state:
                        # デフォルトは中央（インデックス2 = 3番目）
                        st.session_state[key] = local_options[2]

                    # カラム配置: [label, btn1, btn2, btn3, btn4, btn5, label]
                    cols = st.columns([1.3, 0.8, 0.8, 0.8, 0.8, 0.8, 1.3]
                                    )
                    cols[0].markdown("**Aが良い**", unsafe_allow_html=True)

                    slider_value = st.select_slider(
                        "評価スコア",
                        options=local_options,
                        value=st.session_state.get(key, local_options[2]),
                        key=f"slider_{key}",
                        label_visibility="collapsed",
                        format_func=lambda _: "",
                    )
                    st.session_state[key] = slider_value

                    cols[6].markdown("**Bが良い**", unsafe_allow_html=True)

                    # 選択された値を正規化して保存（baseline_on_left=Falseのとき逆転させる）
                    selected_value = st.session_state.get(key, local_options[2])
                    if baseline_on_left:
                        responses[field] = selected_value
                    else:
                        # local_options が逆転しているため、RATING_SCALE での正規値に戻す
                        idx = local_options.index(selected_value)
                        responses[field] = RATING_SCALE[idx]

                    st.divider()

                st.markdown("**ステップ8：コメント（任意）**")
                comment = st.text_area(
                    "各解答について、改善点や感想があればご記入ください。",
                    height=120,
                    key=f"comment_{current_index}_{participant_key}",
                )

                # 【修正点2】余白の確保（ギリギリで見切れるのを防ぐ）
                st.write("") 
                st.write("")

                # 【修正点3】通常の st.button ではなく st.form_submit_button を使う
                submitted = st.form_submit_button("評価を保存して次へ ▶", use_container_width=True)

                if submitted:
                    # formが送信されたタイミングで値を収集して保存処理を行う
                    
                    # スライダーの値を回収
                    for field, _, _ in questions:
                        key = f"{field}_{current_index}_{participant_key}"
                        # フォーム送信時は session_state["slider_..."] に最新の値が入っている
                        # ただし key名が "slider_{key}" なので注意して取得
                        val = st.session_state.get(f"slider_{key}")
                        
                        if baseline_on_left:
                            responses[field] = val
                        else:
                            idx = local_options.index(val)
                            responses[field] = RATING_SCALE[idx]

                    record = {
                        "timestamp": datetime.now().isoformat(),
                        "user_id": user_id,
                        "item_index": current_index,
                        "source_userid": row["source_userid"],
                        "baseline_on_left": baseline_on_left,
                        **survey_answers,
                        **responses,
                        "comment": comment,
                    }
                    save_response(record)
                    
                    # 次のインデックスへ
                    st.session_state.current_index = get_next_index(
                        all_indices, answered | {current_index}
                    )
                    st.success("評価を保存しました！次のサンプルに進みます。")
                    st.rerun()

                # if st.button("評価を保存して次へ ▶", use_container_width=True):
                #     record = {
                #         "timestamp": datetime.now().isoformat(),
                #         "user_id": user_id,
                #         "item_index": current_index,
                #         "source_userid": row["source_userid"],
                #         "baseline_on_left": baseline_on_left,
                #         **survey_answers,
                #         **responses,
                #         "comment": comment,
                #     }
                #     save_response(record)
                #     st.session_state.current_index = get_next_index(
                #         all_indices, answered | {current_index}
                #     )
                #     st.success("評価を保存しました！次のサンプルに進みます。")
                #     st.rerun()


if __name__ == "__main__":
    main()
