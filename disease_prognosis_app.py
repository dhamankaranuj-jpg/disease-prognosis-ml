import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Disease Prognosis System", page_icon="🩺", layout="wide")

# ---------------------------------------------------------
# GLOBAL STYLE
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Hide default streamlit chrome for a cleaner look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #2563eb 0%, #7c3aed 55%, #06b6d4 100%);
    padding: 38px 40px;
    border-radius: 22px;
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.25);
    margin-bottom: 26px;
}
.hero-text .kicker {
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 12px;
    opacity: 0.85;
    font-weight: 600;
}
.hero-text h1 {
    font-size: 40px;
    font-weight: 800;
    margin: 6px 0 8px 0;
    color: white;
}
.hero-text p {
    font-size: 15px;
    opacity: 0.92;
    max-width: 560px;
    margin: 0;
}
.hero-icon svg { width: 110px; height: 110px; }

/* Stat chips row */
.chip-row { display: flex; gap: 14px; margin-bottom: 28px; flex-wrap: wrap; }
.chip {
    flex: 1;
    min-width: 150px;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}
.chip .chip-label { font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.chip .chip-value { font-size: 24px; font-weight: 800; color: #1e293b; margin-top: 4px; }
.chip .chip-emoji { font-size: 22px; }

/* Metric / model cards */
.metric-card {
    padding: 20px 16px;
    border-radius: 16px;
    background: linear-gradient(160deg, #ffffff, #f8fafc);
    border: 1px solid #e2e8f0;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    transition: transform 0.15s ease;
}
.metric-card:hover { transform: translateY(-4px); }
.metric-card .m-name { font-size: 14px; color: #6b7280; font-weight: 600; margin-bottom: 6px; }
.metric-card .m-value { font-size: 30px; font-weight: 800; color: #2563eb; }
.metric-card .m-crown { font-size: 20px; }

/* Result boxes */
.result-box-safe {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #10b981;
    text-align: center;
    box-shadow: 0 10px 26px rgba(16,185,129,0.18);
}
.result-box-danger {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 2px solid #ef4444;
    text-align: center;
    box-shadow: 0 10px 26px rgba(239,68,68,0.18);
}
.result-box-info {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 2px solid #2563eb;
    text-align: center;
    box-shadow: 0 10px 26px rgba(37,99,235,0.18);
}
.result-box-safe h2, .result-box-danger h2, .result-box-info h2 { margin: 0; }

/* Section header helper */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 4px;
}
.section-sub { color: #6b7280; font-size: 14px; margin-bottom: 18px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stFileUploader label div { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HERO HEADER (with inline SVG illustration — no external images needed)
# ---------------------------------------------------------
KIDNEY_SVG = """
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <path d="M75 20 C40 20 20 55 25 95 C29 128 55 150 60 175 C63 188 78 190 82 178
           C88 158 78 140 82 118 C86 96 110 92 118 70 C128 45 118 20 90 20 Z"
        fill="rgba(255,255,255,0.92)"/>
  <path d="M130 25 C165 25 185 60 180 100 C176 133 150 155 145 180 C142 193 127 195 123 183
           C117 163 127 145 123 123 C119 101 95 97 87 75 C77 50 87 25 115 25 Z"
        fill="rgba(255,255,255,0.65)"/>
  <circle cx="60" cy="95" r="6" fill="#2563eb"/>
  <circle cx="140" cy="100" r="6" fill="#2563eb"/>
</svg>
"""

st.markdown(f"""
<div class="hero">
    <div class="hero-text">
        <div class="kicker">AI-Powered Clinical Screening Tool</div>
        <h1>🩺 Disease Prognosis System</h1>
        <p>Predicting disease from patient data with Random Forest, XGBoost, Decision Tree
        &amp; Logistic Regression — trained live on your uploaded dataset, compared side-by-side,
        and ready to score a new patient in seconds.</p>
    </div>
    <div class="hero-icon">{KIDNEY_SVG}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
MAX_ROWS = 5000   # maximum patients (rows) allowed
MAX_COLS = 100    # maximum attributes (columns) allowed
MAX_CLASSES = 20  # result column must have at most this many different values

TARGET_NAMES = ['classification', 'class', 'target', 'label', 'diagnosis', 'outcome', 'result', 'status']
ID_NAMES = {'id', 'patient id', 'patient_id', 'patient no', 'sr no', 's.no', 'sno', 'index'}
MISSING_MARKS = ['?', '', 'nan', 'NaN', 'NAN', 'None', 'NA', 'N/A', 'null']
NEGATIVE_WORDS = ('not', 'non', 'neg', 'normal', 'healthy', 'benign', 'absent')
NEGATIVE_EXACT = {'no', '0', 'false', 'n'}


def clean_name(c):
    """XGBoost does not like [ ] < > in column names."""
    return re.sub(r'[\[\]<>]', '_', str(c).strip())


def is_negative_class(name):
    low = str(name).strip().lower()
    return low in NEGATIVE_EXACT or low.startswith(NEGATIVE_WORDS)


# ---------------------------------------------------------
# CACHED PIPELINE
# ---------------------------------------------------------
def _parse(data, name):
    if name.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(data))
    try:
        return pd.read_csv(io.BytesIO(data), sep=None, engine='python')
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), sep=None, engine='python', encoding='latin-1')


@st.cache_data(show_spinner=False)
def read_file(file_bytes, filename):
    if filename.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            files = [f for f in z.namelist() if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
            if not files:
                raise ValueError("No CSV or Excel file found inside the ZIP.")
            df = _parse(z.read(files[0]), files[0])
    else:
        df = _parse(file_bytes, filename)
    df.columns = [clean_name(c) for c in df.columns]
    return df


def detect_target(columns):
    lower = {c.lower(): c for c in columns}
    for name in TARGET_NAMES:
        if name in lower:
            return lower[name]
    return columns[-1]


@st.cache_data(show_spinner=False)
def prepare(df_raw, target_col):
    df = df_raw.copy()

    # 1. Clean text values (spaces, tabs, '?', 'nan' ...)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().replace(MISSING_MARKS, np.nan)

    # 2. Remove ID-type columns, empty columns, rows without a result
    drop_cols = []
    for c in df.columns:
        if c == target_col:
            continue
        if c.lower() in ID_NAMES or c.lower().startswith('unnamed'):
            drop_cols.append(c)
        elif df[c].isna().all():
            drop_cols.append(c)
        elif not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() == len(df):
            drop_cols.append(c)
    df = df.drop(columns=drop_cols)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Too few rows with a valid result. At least 10 patients are needed.")

    # 3. Result column -> class names
    t = df[target_col]
    if pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_bool_dtype(t):
        if (t % 1 == 0).all():
            t = t.astype(int)
    y_text = t.astype(str).str.strip()
    n_classes = y_text.nunique()
    if n_classes < 2:
        raise ValueError(f"Result column '{target_col}' has only one value. Please choose another column.")
    if n_classes > MAX_CLASSES:
        raise ValueError(f"Result column '{target_col}' has {n_classes} different values. "
                         f"Please choose a column with at most {MAX_CLASSES} classes (like ckd / notckd).")

    # 4. Detect number columns and text columns
    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        raise ValueError("No attribute columns found besides the result column.")

    num_cols, cat_cols = [], []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() >= 0.9 * df[col].notna().sum():
                df[col] = converted
                num_cols.append(col)
            else:
                cat_cols.append(col)

    # very long text columns (like names or addresses) are not useful
    too_many = [c for c in cat_cols if df[c].nunique() > 50]
    if too_many:
        df = df.drop(columns=too_many)
        cat_cols = [c for c in cat_cols if c not in too_many]
        drop_cols += too_many

    # 5. Fill missing values
    meta = {}
    for col in num_cols:
        df[col] = df[col].astype(float)
        df[col] = df[col].fillna(df[col].median())
        s = df[col]
        meta[col] = {"type": "num", "min": float(s.min()), "max": float(s.max()),
                     "median": float(s.median()), "is_int": bool((s % 1 == 0).all())}
    encoders = {}
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0]).astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        meta[col] = {"type": "cat", "options": list(le.classes_)}

    # 6. Final X and y
    target_encoder = LabelEncoder()
    y = pd.Series(target_encoder.fit_transform(y_text), name=target_col)
    X = df[num_cols + cat_cols].astype(float)
    X = X[[c for c in feature_cols if c in X.columns]]
    return X, y, target_encoder, encoders, meta, drop_cols


@st.cache_resource(show_spinner=False)
def train_all_models(X, y):
    # Stratify only when every class has enough patients
    n_test = int(np.ceil(len(y) * 0.2))
    can_stratify = y.value_counts().min() >= 2 and n_test >= y.nunique()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    n_classes = y.nunique()
    average_type = 'binary' if n_classes == 2 else 'weighted'

    results = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            proba = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average=average_type, zero_division=0)
        rec = recall_score(y_test, pred, average=average_type, zero_division=0)
        f1 = f1_score(y_test, pred, average=average_type, zero_division=0)

        # AUC-ROC: binary uses the positive-class probability,
        # multi-class uses a weighted one-vs-rest score
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = roc_auc_score(y_test, proba, multi_class='ovr', average='weighted')
        except ValueError:
            auc = float('nan')

        results[name] = {
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2),
            "AUC-ROC (%)": round(auc * 100, 2) if auc == auc else None,  # auc == auc is False only for NaN
        }

    return models, results, scaler, X_test.index.tolist()


# ---------------------------------------------------------
# SIDEBAR — FILE UPLOAD
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV, Excel or ZIP file", type=["zip", "csv", "xlsx", "xls"])
    st.caption(f"Any dataset up to {MAX_ROWS} patients (rows) and {MAX_COLS} attributes (columns)")
    target_slot = st.container()
    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown(
        "1. **Upload** your dataset\n"
        "2. We **clean & encode** it automatically\n"
        "3. **4 ML models** are trained & compared\n"
        "4. The **best model** powers predictions\n"
    )
    st.markdown("---")
    st.markdown("### 🧬 About")
    st.caption(
        "Screening tools like this help flag at-risk patients for "
        "further clinical evaluation. Works with any disease dataset "
        "that has a result column."
    )
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit, scikit-learn & XGBoost")

# If nothing is uploaded, use the built-in sample dataset (if present in the repo)
from pathlib import Path

SAMPLE_FILE = Path(__file__).parent / "sample_data" / "kidney_disease.csv"

if uploaded_file is not None:
    data_bytes, data_name = uploaded_file.getvalue(), uploaded_file.name
elif SAMPLE_FILE.exists():
    data_bytes, data_name = SAMPLE_FILE.read_bytes(), SAMPLE_FILE.name
else:
    data_bytes, data_name = None, None

if data_bytes is None:
    st.info("👈 Upload your dataset from the sidebar to get started.")
    st.markdown(f"""
    <div class="chip-row">
        <div class="chip"><span class="chip-emoji">🤖</span>
            <div class="chip-label">Models trained</div><div class="chip-value">4</div>
        </div>
        <div class="chip"><span class="chip-emoji">👥</span>
            <div class="chip-label">Max patients</div><div class="chip-value">{MAX_ROWS}</div>
        </div>
        <div class="chip"><span class="chip-emoji">🧪</span>
            <div class="chip-label">Max attributes</div><div class="chip-value">{MAX_COLS}</div>
        </div>
        <div class="chip"><span class="chip-emoji">⚡</span>
            <div class="chip-label">Prediction time</div><div class="chip-value">&lt; 1 sec</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------
# LOAD, CHECK SIZE, PICK TARGET
# ---------------------------------------------------------
try:
    raw = read_file(data_bytes, data_name)
except Exception as e:
    st.error(f"❌ Could not read the file: {e}")
    st.stop()

if raw.shape[0] > MAX_ROWS or raw.shape[1] > MAX_COLS:
    st.error(f"❌ Dataset too large. Allowed: {MAX_ROWS} rows and {MAX_COLS} columns. "
             f"Your file has {raw.shape[0]} rows and {raw.shape[1]} columns.")
    st.stop()

with target_slot:
    if uploaded_file is None:
        st.info("Showing the built-in sample (CKD, 400 patients). Upload your own file above to replace it.")
    st.caption(f"Loaded: {raw.shape[0]} rows × {raw.shape[1]} columns")
    columns = list(raw.columns)
    target_col = st.selectbox("🎯 Result column (what to predict)", columns,
                              index=columns.index(detect_target(columns)))

try:
    X, y, target_encoder, encoders, meta, dropped = prepare(raw, target_col)
except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()

models, results, scaler, test_idx = train_all_models(X, y)

results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
results_df = results_df.sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
class_names = list(target_encoder.classes_)

# ---------------------------------------------------------
# TOP STAT CHIPS
# ---------------------------------------------------------
st.markdown(f"""
<div class="chip-row">
    <div class="chip"><span class="chip-emoji">👥</span>
        <div class="chip-label">Total Patients</div><div class="chip-value">{len(X)}</div>
    </div>
    <div class="chip"><span class="chip-emoji">🧪</span>
        <div class="chip-label">Attributes Used</div><div class="chip-value">{X.shape[1]}</div>
    </div>
    <div class="chip"><span class="chip-emoji">🏷️</span>
        <div class="chip-label">Result Classes</div><div class="chip-value">{len(class_names)}</div>
    </div>
    <div class="chip"><span class="chip-emoji">🏆</span>
        <div class="chip-label">Best Model</div><div class="chip-value" style="font-size:18px;">{best_model_name}</div>
    </div>
</div>
""", unsafe_allow_html=True)

counts = y.value_counts().sort_index()
class_text = " • ".join(f"**{class_names[i]}**: {int(n)}" for i, n in counts.items())
st.caption(f"🎯 Result column: **{target_col}**  |  Classes → {class_text}")
if dropped:
    st.caption(f"🧹 Columns removed automatically (IDs / empty / too many text values): {', '.join(dropped)}")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison", "🧪 Predict a Patient", "📋 Full Patient Results", "🔬 Feature Insights"
])

# --- TAB 1: MODEL COMPARISON ---
with tab1:
    st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All four models are trained on the same 80/20 split and checked on five parameters: Accuracy, Precision, Recall, F1-Score and AUC-ROC.</div>', unsafe_allow_html=True)

    medals = ["🥇", "🥈", "🥉", "🎖️"]
    cols = st.columns(4)
    for i, row in results_df.iterrows():
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="m-crown">{medals[i]}</div>
                <div class="m-name">{row['Model']}</div>
                <div class="m-value">{row['Accuracy (%)']}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown("**All 5 Parameters**")
    metric_cols = ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "AUC-ROC (%)"]
    st.dataframe(results_df[metric_cols], width="stretch", hide_index=True)

    st.write("")
    st.bar_chart(results_df.set_index("Model")[["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "AUC-ROC (%)"]])

    st.success(f"🏆 Best Model: **{best_model_name}** with **{results[best_model_name]['Accuracy (%)']}%** accuracy — this model powers the prediction tab.")

# --- TAB 2: MANUAL PREDICTION (form is built from your dataset's columns) ---
with tab2:
    st.markdown('<div class="section-title">Enter Patient Details</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Prediction powered by <b>{best_model_name}</b>. Fill in the values below — the form is created from your dataset.</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        form_cols = st.columns(3)
        values = {}
        for i, col in enumerate(X.columns):
            m = meta[col]
            with form_cols[i % 3]:
                if m["type"] == "cat":
                    values[col] = st.selectbox(col, m["options"], key=f"in_{col}")
                else:
                    lo, hi, med = m["min"], m["max"], m["median"]
                    pad = (hi - lo) * 0.5 or 1.0
                    if m["is_int"]:
                        lo_i, hi_i = int(np.floor(lo - pad)), int(np.ceil(hi + pad))
                        val = int(min(max(round(med), lo_i), hi_i))
                        values[col] = st.number_input(col, min_value=lo_i, max_value=hi_i,
                                                      value=val, step=1, key=f"in_{col}")
                    else:
                        lo_f, hi_f = float(lo - pad), float(hi + pad)
                        step = float(f"{(hi - lo) / 100:.1g}") if hi > lo else 0.01
                        step = step if step > 0 else 0.01
                        val = float(min(max(med, lo_f), hi_f))
                        values[col] = st.number_input(col, min_value=lo_f, max_value=hi_f,
                                                      value=val, step=step, key=f"in_{col}")
        submitted = st.form_submit_button("🔍 Predict", type="primary", width="stretch")

    if submitted:
        row_vals = {}
        for col in X.columns:
            if meta[col]["type"] == "cat":
                row_vals[col] = float(encoders[col].transform([values[col]])[0])
            else:
                row_vals[col] = float(values[col])
        row = pd.DataFrame([row_vals])[X.columns]

        model_input = scaler.transform(row) if best_model_name == "Logistic Regression" else row
        pred = best_model.predict(model_input)[0]
        proba = best_model.predict_proba(model_input)[0]

        pred_name = target_encoder.inverse_transform([pred])[0]
        confidence = round(float(max(proba)) * 100, 1)

        if is_negative_class(pred_name):
            box, color, dark = "result-box-safe", "#10b981", "#065f46"
            icon, note = "✅", "Routine monitoring is still recommended for at-risk patients."
        elif len(class_names) == 2:
            box, color, dark = "result-box-danger", "#ef4444", "#7f1d1d"
            icon, note = "⚠️", "Recommend follow-up with a doctor for confirmatory testing."
        else:
            box, color, dark = "result-box-info", "#2563eb", "#1e3a8a"
            icon, note = "🩺", "Recommend follow-up with a doctor for confirmatory testing."

        if box == "result-box-safe":
            st.balloons()
        st.markdown(f"""
        <div class="{box}">
            <h2 style="color:{color};">{icon} Predicted: {pred_name}</h2>
            <p style="margin:6px 0 0 0;">Model confidence: <b>{confidence}%</b></p>
            <p style="margin:4px 0 0 0; font-size:13px; color:{dark};">{note}</p>
        </div>
        """, unsafe_allow_html=True)

        proba_df = pd.DataFrame({
            "Class": target_encoder.inverse_transform(best_model.classes_),
            "Probability (%)": np.round(proba * 100, 1),
        })
        st.write("")
        st.dataframe(proba_df, width="stretch", hide_index=True)
        st.caption("⚠️ This tool is for educational/screening purposes only and is not a substitute for professional medical diagnosis.")

# --- TAB 3: FULL RESULTS ---
with tab3:
    st.markdown('<div class="section-title">Prediction for All Patients</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Full dataset scored with the best-performing model.</div>', unsafe_allow_html=True)

    X_full_input = scaler.transform(X) if best_model_name == "Logistic Regression" else X
    full_pred = best_model.predict(X_full_input)

    output = pd.DataFrame({
        "Actual": target_encoder.inverse_transform(y.values),
        "Predicted": target_encoder.inverse_transform(full_pred),
        "Data Used": np.where(X.index.isin(test_idx), "Test", "Train"),
    })
    output.index.name = "Patient No"

    match_rate = round((output["Actual"] == output["Predicted"]).mean() * 100, 2)
    st.caption(f"✅ Overall agreement with actual labels (train + test): **{match_rate}%**")

    st.dataframe(output, width="stretch", height=400)

    csv_data = output.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download Full Results (CSV)",
        csv_data,
        "final_result.csv",
        "text/csv",
        width="stretch",
    )

# --- TAB 4: FEATURE INSIGHTS ---
with tab4:
    st.markdown('<div class="section-title">What Drives the Prediction?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Top attributes the best-performing model relies on.</div>', unsafe_allow_html=True)

    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
    elif hasattr(best_model, "coef_"):
        importances = pd.Series(np.abs(best_model.coef_).mean(axis=0), index=X.columns)

    if importances is not None:
        importances = importances.sort_values(ascending=False).head(12)
        imp_df = importances.reset_index()
        imp_df.columns = ["Feature", "Importance"]
        st.bar_chart(imp_df.set_index("Feature"))
        st.caption("Higher bars indicate attributes the model relies on most when classifying a patient.")

    with st.expander("📖 Columns used"):
        info = []
        for col in X.columns:
            m = meta[col]
            if m["type"] == "num":
                info.append({"Column": col, "Type": "Number", "Range / Options": f"{m['min']:g} – {m['max']:g}"})
            else:
                opts = m["options"]
                txt = ", ".join(map(str, opts[:8])) + (" ..." if len(opts) > 8 else "")
                info.append({"Column": col, "Type": "Category", "Range / Options": txt})
        st.dataframe(pd.DataFrame(info), width="stretch", hide_index=True)
