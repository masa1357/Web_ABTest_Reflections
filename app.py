import json
import pandas as pd
import streamlit as st
import hashlib
import random
from datetime import datetime
from pathlib import Path
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from typing import Any, Dict, List, Optional, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore  # Python 3.10 fallback
    except ModuleNotFoundError:
        tomllib = None

# ===== パスと定数 =====
RESULTS_PATH = Path("results/abtest_results.jsonl")
BASELINE_PATH = Path("data/worse2.json")
ADVICE_PATH = Path("data/better_phase2_format.json")

# 保存するデータのカラム順序（スプレッドシートのヘッダーになります）
CSV_COLUMNS = [
    "timestamp",
    "user_id",
    "item_index",
    "source_userid",
    "baseline_on_left",
    "kyushu_student",
    "info_course_taken",
    "info_course_grade",
    "accuracy",
    "readability",
    "persuasiveness",
    "actionability",
    "hallucination",
    "usefulness",
    "overall",
    "comment",
]

# ABテスト対象とする userid のサンプル一覧（順序は後でシャッフルされる）
correct_uid = [
    "C-2022-1_U57"
    "C-2022-1_U73"
    "C-2021-1_U8"
    "C-2021-2_U35"
    "C-2021-2_U161"
    "C-2021-2_U23"
    "C-2021-1_U29"
    "C-2021-1_U66",
    "C-2021-1_U21",
    "C-2021-2_U172",
]

incorrect_uid = [
    "C-2021-2_U115",
    "C-2022-1_U52",
    "C-2021-1_U43",
    "C-2022-1_U6",
    "C-2022-1_U29",
    "C-2022-1_U61",
    "C-2021-1_U97",
    "C-2022-1_U92",
    "C-2021-2_U149",
    "C-2021-2_U1",
]

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


# ===== スプレッドシート接続機能 =====
@st.cache_resource
def load_app_secrets() -> Dict[str, Any]:
    """st.secrets を優先し、無ければ config/secrets.toml を読み込む"""
    secrets: Dict[str, Any] = {}
    try:
        secrets.update(dict(st.secrets))
    except Exception:
        pass

    required_keys = ("gcp_service_account", "spreadsheet_id")
    if all(k in secrets for k in required_keys):
        return secrets

    config_path = Path("config/secrets.toml")
    if config_path.exists():
        if tomllib is None:
            st.error("config/secrets.toml を読むには tomllib/tomli が必要です。")
            st.stop()
        with config_path.open("rb") as f:
            file_secrets = tomllib.load(f)
        # st.secrets を優先し、足りないキーだけ補完
        for key in required_keys:
            if key not in secrets and key in file_secrets:
                secrets[key] = file_secrets[key]

    return secrets


@st.cache_resource
def get_worksheet():
    """Secretsから認証情報を読み込み、スプレッドシートのワークシートオブジェクトを返す"""
    secrets = load_app_secrets()

    # Secretsから認証情報を取得（st.secrets 優先、無ければ config/secrets.toml）
    if "gcp_service_account" not in secrets:
        st.error(
            "Secretsに gcp_service_account が設定されていません。（config/secrets.toml も確認済み）"
        )
        st.stop()

    # 認証スコープ
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    # 辞書オブジェクトからCredentialsを作成
    creds_dict = dict(secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    # スプレッドシートを開く
    spreadsheet_id = secrets["spreadsheet_id"]
    try:
        sh = client.open_by_key(spreadsheet_id)
        # 1枚目のシートを使用（なければ作るなどの処理も可能だが今回はsheet1固定）
        worksheet = sh.sheet1
        return worksheet
    except Exception as e:
        st.error(f"スプレッドシートへの接続に失敗しました: {e}")
        st.stop()


def init_sheet_header(worksheet):
    """シートが空の場合、ヘッダー行を追加する"""
    try:
        existing = worksheet.get_all_values()
        if not existing:
            worksheet.append_row(CSV_COLUMNS)
    except Exception:
        pass


# ===== データロード（変更なし） =====
def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        st.error(f"データファイルが見つかりません: {path}")
        st.stop()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def clean_baseline_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    # 先頭の "assistant" ラベルやボイラープレートを除去
    cleaned = re.sub(r"(?i)\Aassistant\b\s*\n*", "", cleaned).strip()
    cleaned = re.sub(
        r"\A以下は[^\n]*?(生成文|フィードバック|改善アドバイス)[^\n]*\n+",
        "",
        cleaned,
    ).strip()
    cleaned = cleaned.replace(" �善", "改善")
    # 見出しレベルを統一（最初の # は ###、2つ目以降は ####）
    heading_count = 0
    normalized_lines: List[str] = []
    label_headings = [
        "あなたの強み",
        "改善が必要なポイント",
        "これから意識したいこと",
        "まとめ",
    ]
    for line in cleaned.splitlines():
        m = re.match(r"^(\s*)(#+)\s*(.*)$", line)
        if m:
            leading_ws, _hashes, rest = m.groups()
            rest_stripped = rest.strip()
            if rest_stripped.startswith("**") and rest_stripped.endswith("】"):
                rest = rest_stripped[2:-1].strip()
            heading_count += 1
            target = "###" if heading_count == 1 else "####"
            normalized_lines.append(f"{leading_ws}{target} {rest}".rstrip())
        else:
            stripped = line.strip()
            if stripped.startswith("**") and stripped.endswith("】"):
                stripped = stripped[2:-1].strip()
                line = stripped
            matched = next((h for h in label_headings if stripped == h), None)
            if matched:
                line = line.replace(matched, f"{matched}：", 1)
            normalized_lines.append(line)
    cleaned = "\n".join(normalized_lines).strip()
    return cleaned


# ===== 結果の読み書き（スプレッドシート版に変更） =====
# app.py の load_answered_indices 関数をこれに置き換える


def load_answered_indices(user_id: str) -> set[int]:
    """スプレッドシートから回答済みの item_index を取得する（型変換対応版）"""
    worksheet = get_worksheet()

    try:
        # 全データを取得
        records = worksheet.get_all_records()
        if not records:
            return set()

        df = pd.DataFrame(records)

        # カラム名の揺らぎを吸収（user_id か userid か）
        user_col = None
        if "user_id" in df.columns:
            user_col = "user_id"
        elif "userid" in df.columns:
            user_col = "userid"

        if user_col is None:
            return set()

        # 【重要】データフレーム側と入力側の両方を「文字列」に強制変換して比較する
        # これにより 123(int) と "123"(str) の不一致を防ぐ
        df[user_col] = df[user_col].astype(str).str.strip()
        target_user_id = str(user_id).strip()

        user_df = df[df[user_col] == target_user_id]

        # item_index が存在するか確認
        idx_col = None
        if "item_index" in user_df.columns:
            idx_col = "item_index"
        elif "item_index" in df.columns:  # 全体から探す
            idx_col = "item_index"

        if idx_col:
            # 空文字や欠損を除外して int に変換
            return set(
                pd.to_numeric(user_df[idx_col], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )

        return set()

    except Exception as e:
        # デバッグ用にエラーを表示したければコメントアウトを外す
        # st.error(f"読み込みエラー: {e}")
        return set()


def save_response(record: Dict[str, Any]) -> None:
    """回答1件をスプレッドシートに追記する"""
    worksheet = get_worksheet()

    # 初回だけヘッダーチェック
    init_sheet_header(worksheet)

    # 定義したカラム順序に従って値をリスト化する
    row_values = []
    for col in CSV_COLUMNS:
        row_values.append(record.get(col, ""))

    # 追記
    worksheet.append_row(row_values)


def get_next_index(all_indices: List[int], answered: set[int]) -> int | None:
    for idx in all_indices:
        if idx not in answered:
            return idx
    return None


# ...existing code...
def load_items(
    max_items: int = MAX_ITEMS, *, user_id: str | None = None
) -> pd.DataFrame:
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

    def extract_baseline_response(entry: Dict[str, Any], uid: str) -> str:
        if not isinstance(entry, dict):
            return ""
        if isinstance(entry.get("response"), str) and entry["response"]:
            return entry["response"]
        if isinstance(entry.get("text"), str) and entry["text"]:
            return entry["text"]
        if isinstance(entry.get(uid), str) and entry[uid]:
            return entry[uid]
        # worse2.json のように「grade 以外の1キーが本文」の形を想定
        for key, val in entry.items():
            if key in ("grade", "userid", "user_id"):
                continue
            if isinstance(val, str) and val:
                return val
        return ""

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
    # use_uid で何も拾えなかった場合は共通集合を丸ごと使う
    if not prioritized_uids:
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
                "baseline_response": clean_baseline_text(
                    extract_baseline_response(base_entry, uid)
                ),
                "student_advice_title": advice_entry.get("student_advice_title", "")
                or advice_entry.get("title", ""),
                "student_advice_body": advice_entry.get("student_advice_body", "")
                or advice_entry.get("body", ""),
                "student_grade": advice_entry.get("grade", ""),
            }
        )

    df = pd.DataFrame(records).sort_values("item_index").reset_index(drop=True)
    return df


def debug_admin_view(user_id: str) -> bool:
    if user_id != "admin_ms13379":
        return False

    st.title("Debug: baseline data viewer")
    baseline = load_json_dict(BASELINE_PATH)
    advice = load_json_dict(ADVICE_PATH)

    def build_user_map(d: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        m: Dict[str, Dict[str, Any]] = {}
        for top_key, entry in d.items():
            candidates = [top_key]
            if isinstance(entry, dict):
                for fld in ("userid", "user_id"):
                    val = entry.get(fld)
                    if isinstance(val, str) and val:
                        candidates.append(val)
            for uid in candidates:
                if uid not in m:
                    m[uid] = entry
        return m

    base_map = build_user_map(baseline)
    advice_map = build_user_map(advice)
    top_keys_common = set(baseline.keys()) & set(advice.keys())
    map_common = set(base_map.keys()) & set(advice_map.keys())
    st.caption(
        f"baseline: {len(baseline)} | advice: {len(advice)} | "
        f"match(top keys): {len(top_keys_common)} | match(mapped): {len(map_common)}"
    )

    grade_series = (
        pd.DataFrame(
            {"grade": [entry.get("grade", "") for entry in base_map.values()]}
        )
        .fillna("")
        .assign(grade=lambda df: df["grade"].astype(str).str.strip())
    )
    if not grade_series.empty:
        grade_counts = grade_series["grade"].value_counts(dropna=False)
        grade_percent = (grade_counts / len(grade_series) * 100).round(1)
        grade_table = (
            pd.DataFrame({"count": grade_counts, "percent": grade_percent})
            .reset_index()
            .rename(columns={"index": "grade"})
        )
        st.dataframe(grade_table, use_container_width=True)

    rows: List[Dict[str, Any]] = []
    for uid, entry in baseline.items():
        text = ""
        grade = ""
        if isinstance(entry, dict):
            grade = entry.get("grade", "")
            if isinstance(entry.get("response"), str) and entry["response"]:
                text = entry["response"]
            elif isinstance(entry.get("text"), str) and entry["text"]:
                text = entry["text"]
            elif isinstance(entry.get(uid), str) and entry[uid]:
                text = entry[uid]
            else:
                for key, val in entry.items():
                    if key in ("grade", "userid", "user_id"):
                        continue
                    if isinstance(val, str) and val:
                        text = val
                        break
        else:
            text = str(entry)
        rows.append({"userid": uid, "grade": grade, "text": text})

    if "debug_index" not in st.session_state:
        st.session_state.debug_index = 0

    total = len(rows)
    if total:
        nav_cols = st.columns([1, 2, 1])
        if nav_cols[0].button("◀ Prev", disabled=st.session_state.debug_index <= 0):
            st.session_state.debug_index = max(0, st.session_state.debug_index - 1)
        if nav_cols[2].button(
            "Next ▶", disabled=st.session_state.debug_index >= total - 1
        ):
            st.session_state.debug_index = min(total - 1, st.session_state.debug_index + 1)

        current = rows[st.session_state.debug_index]
        st.caption(
            f"{st.session_state.debug_index + 1}/{total} | {current['userid']} | grade: {current['grade']}"
        )
        st.markdown(clean_baseline_text(current["text"]))
        st.divider()

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with st.expander("baseline plain text (no markdown)"):
        for row in rows:
            st.text(f"[{row['userid']}] (grade: {row['grade']})")
            st.text(row["text"])
            st.text("----")
    with st.expander("raw baseline JSON (worse2.json)"):
        st.json(baseline)

    st.stop()
    return True


# ===== ユーザーデータ管理（進捗読み込み＆プロフィール復元） =====
def load_user_data(user_id: str) -> Tuple[set[int], Optional[Dict[str, str]]]:
    """
    指定されたuser_idに関連するデータをスプレッドシートから全検索する。
    戻り値: (回答済みindexの集合, 最後に保存されたプロフィール情報の辞書)
    """
    worksheet = get_worksheet()
    try:
        records = worksheet.get_all_records()
        if not records:
            return set(), None

        target_uid_str = str(user_id).strip()
        answered_indices = set()
        last_profile = None

        # シンプルに全行ループして、user_idが一致するものを探す
        for row in records:
            # カラム名の揺らぎ対応
            row_uid = row.get("user_id") or row.get("userid") or ""

            # 文字列化して空白除去して比較（これが最も確実）
            if str(row_uid).strip() == target_uid_str:

                # 回答済みインデックスを回収
                idx_val = row.get("item_index")
                if idx_val is not None and str(idx_val).strip() != "":
                    try:
                        answered_indices.add(int(idx_val))
                    except ValueError:
                        pass

                # プロフィール情報を保持（上書きしていくので最後に見つかったものが最新になる）
                # 値が空でない場合のみ取得するようにする
                if row.get("kyushu_student"):
                    last_profile = {
                        "kyushu_student": row.get("kyushu_student"),
                        "info_course_taken": row.get("info_course_taken"),
                        "info_course_grade": row.get("info_course_grade"),
                    }

        return answered_indices, last_profile

    except Exception as e:
        print(f"Error loading user data: {e}")
        return set(), None


# ===== Streamlit アプリ本体 =====
def main():
    st.set_page_config(
        page_title="Feedback A/B Test", layout="wide", initial_sidebar_state="collapsed"
    )

    # 最初にスプレッドシート接続テスト＆ヘッダー初期化
    ws = get_worksheet()
    init_sheet_header(ws)

    df_preview = load_items()
    if df_preview.empty:
        st.error(
            "比較対象となるデータが見つかりません。dataフォルダのJSONを確認してください。"
        )
        return

    page = st.sidebar.selectbox("ページ選択", ["説明", "評価画面"])

    if page == "説明":
        st.title("学生フィードバック評価")
        st.markdown(
            f"""
このアプリでは、大学の講義「情報科学」を受講した学生に対するフィードバックを比較評価していただきます。

- 評価するフィードバックは {len(df_preview)} 件あります。
- 所要時間は15~20分程度を想定しています。
- 回答前に簡単なパーソナリティ設問へ回答してもらいます。
- 各サンプルについて5項目（可読性/説得力/行動可能性/ハルシネーション/有用性）で、A・Bいずれが優れているか，5段階で選択いただきます。
- {len(df_preview)} 件すべて回答すると終了です。同じ名前でアクセスすれば途中から再開できます。
回答よろしくお願いいたします！

"""
        )
        st.info(
            "説明は以上です。サイドバーの「ページ選択」から「評価画面」を選択して、評価を開始してください。"
        )
        return

    st.title("フィードバック比較評価")
    st.markdown("同じ名前でアクセスすると、途中から評価を再開できます。")
    st.markdown(
        "意図せず途中から始まる場合、同じ名前が使われている可能性があります。\n別の名前をお試しください。"
    )

    user_id = st.text_input(
        "任意の名前を記入してください...",
        value="",
        placeholder="例: 名前+生年月日 など",
    ).strip()
    # もし途中から始まるなら，別の名前で始めるよう促すテキストを表示
    # 赤文字で，名前を忘れると途中から再開できない旨を表示
    st.markdown(
        "<span style='color:red;'>注意! 名前を忘れると途中から再開できません!</span>",
        unsafe_allow_html=True,
    )
    if not user_id:
        st.warning("名前を入力すると評価を開始できます。")
        st.stop()

    debug_admin_view(user_id)

    # ユーザーIDが入力されたら、スプレッドシートからデータを取得
    # ここで「回答済みセット」と「過去のプロフィール」を一括取得
    answered, prev_profile = load_user_data(user_id)

    # ユーザー切り替え時のセッション初期化
    if st.session_state.get("current_user_id") != user_id:
        st.session_state.current_user_id = user_id
        st.session_state.current_index = None
        st.session_state.survey_answers = None

        # ウィジェットのキーを削除してリセットさせる
        for widget_key in ("kyushu_student", "info_course_taken", "info_grade_text"):
            if widget_key in st.session_state:
                del st.session_state[widget_key]

        # ★ここで過去のプロフィールがあればセッションにセット（自動入力）★
        if prev_profile:
            st.toast("過去のプロフィール情報を復元しました", icon="✅")
            st.session_state["kyushu_student"] = prev_profile.get("kyushu_student")
            st.session_state["info_course_taken"] = prev_profile.get(
                "info_course_taken"
            )
            st.session_state["info_grade_text"] = prev_profile.get("info_course_grade")

    st.subheader("パーソナリティ質問")
    kyushu_options = ["-- 選択してください --", "はい", "いいえ"]

    # keyを指定しているので、session_stateに値が入っていればそれが初期値になる
    kyushu_student = st.selectbox(
        "九州大学の学生ですか？", kyushu_options, key="kyushu_student"
    )

    info_course_options = ["-- 選択してください --", "はい", "いいえ"]
    info_course_taken = st.selectbox(
        "情報科学の講義を受講したことがありますか？",
        info_course_options,
        key="info_course_taken",
    )

    grade_options = ["未回答", "A", "B", "C", "D", "F"]
    info_course_grade = st.selectbox(
        "受講していたときの成績（任意）", options=grade_options, key="info_grade_text"
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

    df_items = load_items(user_id=user_id)
    if df_items.empty:
        st.error("正しくロードできる評価対象がありません。")
        st.stop()

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
        st.markdown(f"#### 成績{row['baseline_grade']}の学生に対するフィードバック")
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
            ("readability", "ステップ1：可読性", "どちらが読みやすいと感じますか？"),
            (
                "persuasiveness",
                "ステップ2：説得力",
                "どちらの根拠が明確だと思いますか？",
            ),
            (
                "actionability",
                "ステップ3：行動可能性",
                "成績向上のためのフィードバックが明確に示されていますか？",
            ),
            (
                "hallucination",
                "ステップ4：ハルシネーション",
                "嘘や誇張と思われる表現を使っていますか？",
            ),
            (
                "usefulness",
                "ステップ5：有用性",
                "あなたが学生だった場合、どちらのフィードバックが役に立つと思いますか？",
            ),
        ]

        participant_key = str(abs(hash(user_id)))
        responses: Dict[str, str] = {}

        # コンテナとフォームの開始
        with st.container(height=800):
            with st.form(key=f"eval_form_{current_index}_{participant_key}"):

                for field, title, description in questions:
                    st.markdown(f"**{title}**")
                    st.caption(description)

                    # ユニークキーの生成
                    key = f"{field}_{current_index}_{participant_key}"

                    # session_state の初期化
                    if key not in st.session_state:
                        st.session_state[key] = local_options[2]

                    # カラム配置
                    cols = st.columns([1.5, 0.8, 0.8, 0.8, 0.8, 0.8, 1.5])
                    cols[0].markdown("**Aが良い**", unsafe_allow_html=True)

                    # スライダー表示（ここで値は取得せず、表示のみ行う）
                    # keyを指定しているので、ユーザーの操作はsession_stateに自動記録されます
                    st.select_slider(
                        "評価スコア",
                        options=local_options,
                        value=st.session_state.get(key, local_options[2]),
                        key=f"slider_{key}",
                        label_visibility="collapsed",
                        format_func=lambda _: "",
                    )

                    cols[6].markdown("**Bが良い**", unsafe_allow_html=True)
                    st.divider()

                # コメント欄
                st.markdown("**ステップ6：コメント（任意）**")
                comment = st.text_area(
                    "各解答について、改善点や感想があればご記入ください。",
                    height=120,
                    key=f"comment_{current_index}_{participant_key}",
                )

                st.write("")
                st.write("")

                # 送信ボタン
                submitted = st.form_submit_button(
                    "評価を保存して次へ ▶", use_container_width=True
                )

                if submitted:
                    # ===== ここで一括して値を回収します =====
                    for field, _, _ in questions:
                        key = f"{field}_{current_index}_{participant_key}"
                        # フォーム送信時の最新の値を取得
                        val = st.session_state.get(f"slider_{key}")

                        if baseline_on_left:
                            responses[field] = val
                        else:
                            # 逆転している場合の正規化
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


if __name__ == "__main__":
    main()
