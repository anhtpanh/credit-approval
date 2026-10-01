import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

NUM = ["A2", "A3", "A8", "A11", "A14", "A15"]
CAT = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
FEATS = [f"A{i}" for i in range(1, 16)]
ALL_COLS = FEATS + ["A16"]


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    with open("metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    return model, metrics


def clean(df):
    """Ép kiểu số, đổi '?' hoặc ô trống thành NaN (giống lúc huấn luyện)."""
    df = df.copy()
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT:
        df[c] = df[c].map(
            lambda v: np.nan if (pd.isna(v) or str(v).strip() in ("?", "")) else str(v).strip()
        )
    return df


def read_upload(f):
    """Đọc CSV/Excel. Nếu file không có dòng tiêu đề (như crx.data) thì tự gán A1..A16."""
    is_excel = f.name.lower().endswith((".xlsx", ".xls"))

    def _read(header):
        f.seek(0)
        if is_excel:
            return pd.read_excel(f, header=header)
        return pd.read_csv(f, header=header, sep=None, engine="python")

    raw = _read(0)
    raw.columns = [str(c).strip() for c in raw.columns]
    if not set(FEATS).issubset(raw.columns):
        raw = _read(None)
        if raw.shape[1] in (15, 16):
            raw.columns = ALL_COLS[: raw.shape[1]]
    return raw


def predict(model, raw, threshold):
    X = clean(raw[FEATS])
    proba = model.predict_proba(X)[:, 1]
    out = raw.copy()
    out["Xác suất duyệt"] = proba.round(3)
    out["Dự báo"] = np.where(proba >= threshold, "+", "-")
    out["Kết luận"] = np.where(proba >= threshold, "✅ Duyệt (+)", "❌ Từ chối (-)")
    return out


def tab_dashboard(M):
    st.subheader("Hiệu năng mô hình trên tập kiểm tra")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mô hình dùng", M["best_model"])
    c2.metric("Accuracy", f"{M['accuracy']:.1%}")
    c3.metric("Precision", f"{M['precision']:.1%}")
    c4.metric("Recall", f"{M['recall']:.1%}")
    c5.metric("F1", f"{M['f1']:.1%}")
    st.caption(f"Huấn luyện trên {M['n_train']} dòng, kiểm tra trên {M['n_test']} dòng.")

    left, right = st.columns(2)
    with left:
        cm = np.array(M["confusion_matrix"])
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=["Dự báo -", "Dự báo +"], y=["Thực tế -", "Thực tế +"],
            title="Ma trận nhầm lẫn",
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        imp = pd.DataFrame(M["importance"]).sort_values("value")
        fig = px.bar(imp, x="value", y="feature", orientation="h",
                     title="Top đặc trưng quan trọng nhất")
        fig.update_layout(xaxis_title="Mức độ quan trọng", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**So sánh các mô hình**")
        st.dataframe(pd.DataFrame(M["comparison"]), hide_index=True, width="stretch")
    with right:
        cc = pd.DataFrame({"Nhãn": list(M["class_counts"].keys()),
                           "Số dòng": list(M["class_counts"].values())})
        fig = px.pie(cc, names="Nhãn", values="Số dòng", title="Phân bố nhãn trong dữ liệu gốc")
        st.plotly_chart(fig, width="stretch")


def tab_predict(model):
    st.subheader("Upload dữ liệu mới và dự báo")
    st.write("File cần có 15 cột **A1 ... A15** (cột A16 là nhãn thật, có hay không đều được). "
             "Hỗ trợ CSV và Excel; giá trị thiếu ghi là `?` hoặc để trống.")

    if os.path.exists("mau_upload.csv"):
        with open("mau_upload.csv", "rb") as f:
            st.download_button("⬇️ Tải file mẫu để thử", f.read(), "mau_upload.csv", "text/csv")

    up = st.file_uploader("Chọn file dữ liệu", type=["csv", "xlsx", "xls"])
    if up is None:
        st.info("Hãy upload một file để bắt đầu.")
        return

    try:
        raw = read_upload(up)
        missing = [c for c in FEATS if c not in raw.columns]
        if missing:
            st.error(f"File thiếu các cột: {', '.join(missing)}")
            return
    except Exception as e:
        st.error(f"Không đọc được file: {e}")
        return

    c1, c2 = st.columns(2)
    n = c1.number_input("Số dòng muốn dự báo (n)", min_value=1, max_value=len(raw),
                        value=min(len(raw), 50), step=1)
    thr = c2.slider("Ngưỡng duyệt (xác suất ≥ ngưỡng thì dự báo +)", 0.05, 0.95, 0.5, 0.05)

    res = predict(model, raw.head(int(n)), thr)

    n_pos = int((res["Dự báo"] == "+").sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Số dòng đã dự báo", len(res))
    k2.metric("Dự báo duyệt (+)", n_pos)
    k3.metric("Dự báo từ chối (-)", len(res) - n_pos)
    k4.metric("Xác suất duyệt trung bình", f"{res['Xác suất duyệt'].mean():.1%}")

    left, right = st.columns(2)
    with left:
        counts = res["Dự báo"].value_counts().rename_axis("Dự báo").reset_index(name="Số dòng")
        st.plotly_chart(px.bar(counts, x="Dự báo", y="Số dòng", color="Dự báo",
                               title="Số hồ sơ theo kết quả dự báo"), width="stretch")
    with right:
        st.plotly_chart(px.histogram(res, x="Xác suất duyệt", nbins=20,
                                     title="Phân bố xác suất duyệt"), width="stretch")

    if "A16" in res.columns:
        truth = res["A16"].astype(str).str.strip()
        valid = truth.isin(["+", "-"])
        if valid.any():
            acc = (truth[valid] == res.loc[valid, "Dự báo"]).mean()
            st.success(f"File có nhãn A16: độ chính xác trên {int(valid.sum())} dòng = {acc:.1%}")

    st.markdown("**Kết quả chi tiết**")
    first = ["Kết luận", "Dự báo", "Xác suất duyệt"]
    st.dataframe(res[first + [c for c in res.columns if c not in first]],
                 width="stretch", hide_index=True)
    st.download_button("⬇️ Tải kết quả (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                       "ket_qua_du_bao.csv", "text/csv")


def main():
    st.set_page_config(page_title="Dự báo duyệt thẻ tín dụng", page_icon="💳", layout="wide")
    st.title("💳 Dự báo duyệt hồ sơ thẻ tín dụng")
    st.caption("Dữ liệu: UCI Credit Approval · Mô hình học máy huấn luyện trên Google Colab, "
               "dữ liệu lưu ở MySQL (Aiven).")
    model, M = load_artifacts()
    t1, t2 = st.tabs(["📊 Dashboard mô hình", "📤 Upload & dự báo"])
    with t1:
        tab_dashboard(M)
    with t2:
        tab_predict(model)


if __name__ == "__main__":
    main()
