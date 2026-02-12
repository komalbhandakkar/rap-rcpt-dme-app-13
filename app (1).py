import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pandas as pd
import joblib
import os

@st.cache_data
def load_shap_background(feature_cols_local, ranges_df_local):
    bg_path = pth('shap_background.csv') if 'pth' in globals() else 'shap_background.csv'
    if os.path.exists(bg_path):
        bg_df = pd.read_csv(bg_path)
        bg_df = bg_df[[c for c in feature_cols_local if c in bg_df.columns]].dropna()
        if len(bg_df) > 1000:
            bg_df = bg_df.sample(n=1000, random_state=42)
        return bg_df

    cols = [c for c in feature_cols_local if c in ranges_df_local.index]
    synth = []
    for i in range(300):
        row = {}
        for c in cols:
            mn = float(ranges_df_local.loc[c, 'min'])
            mx = float(ranges_df_local.loc[c, 'max'])
            row[c] = (mn + mx) / 2.0
        synth.append(row)
    return pd.DataFrame(synth)[cols]


@st.cache_resource
def build_explainers(rcpt_model_local, dme_model_local, X_bg_local):
    rcpt_expl = shap.TreeExplainer(rcpt_model_local, data=X_bg_local, feature_names=X_bg_local.columns, feature_perturbation='interventional')
    dme_expl = None
    if dme_model_local is not None:
        dme_expl = shap.TreeExplainer(dme_model_local, data=X_bg_local, feature_names=X_bg_local.columns, feature_perturbation='interventional')
    return rcpt_expl, dme_expl


def render_shap_section(model_title, model_key, explainer, feature_cols_local, input_df_local, X_bg_local):
    st.markdown('### ' + model_title + ' SHAP')
    with st.expander(model_key + ' SHAP explanations (Option B)', expanded=False):
        st.caption('Force plot explains the current input. Beeswarm, decision, and dependence plots use a background sample for global behavior.')

        st.markdown('#### Force plot (current input)')
        try:
            sv_input = explainer(input_df_local, check_additivity=False)
            force = shap.force_plot(explainer.expected_value, sv_input.values[0], input_df_local.iloc[0, :], matplotlib=False)
            components.html(force.html(), height=280, scrolling=True)
        except Exception as e:
            st.error('Force plot failed: ' + str(e))

        st.markdown('#### Summary plot (beeswarm)')
        try:
            sv_bg = explainer(X_bg_local, check_additivity=False)
            plt.figure()
            shap.plots.beeswarm(sv_bg, max_display=15, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as e:
            st.error('Beeswarm failed: ' + str(e))

        st.markdown('#### Decision plot (background)')
        try:
            if 'sv_bg' not in locals():
                sv_bg = explainer(X_bg_local, check_additivity=False)
            plt.figure()
            shap.decision_plot(explainer.expected_value, sv_bg.values, X_bg_local, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as e:
            st.error('Decision plot failed: ' + str(e))

        st.markdown('#### Dependence plot')
        dep_feat = st.selectbox(model_key + ' dependence feature', list(X_bg_local.columns), index=0, key='dep_' + model_key)
        try:
            if 'sv_bg' not in locals():
                sv_bg = explainer(X_bg_local, check_additivity=False)
            plt.figure()
            shap.dependence_plot(dep_feat, sv_bg.values, X_bg_local, feature_names=list(X_bg_local.columns), show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as e:
            st.error('Dependence plot failed: ' + str(e))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def pth(fname: str) -> str:
    return os.path.join(BASE_DIR, fname)

st.set_page_config(page_title='RCPT and DME Predictor for RAP Concrete', layout='wide')

@st.cache_resource
def load_artifacts():
    feature_cols_local = joblib.load(pth('feature_cols.pkl'))
    ranges_df_local = pd.read_csv(pth('input_feature_ranges.csv'), index_col=0)
    rcpt_model_local = joblib.load(pth('xgb_optuna_rcpt.pkl'))
    params_local = joblib.load(pth('xgb_optuna_params.pkl'))

    dme_model_local = None
    dme_path = pth('xgb_optuna_dme.pkl')
    if os.path.exists(dme_path):
        dme_model_local = joblib.load(dme_path)

    return feature_cols_local, ranges_df_local, rcpt_model_local, dme_model_local, params_local

feature_cols, ranges_df, rcpt_model, dme_model, params = load_artifacts()

st.markdown(
    """
    <style>
    :root {
        --bg: #070a12;
        --stroke: rgba(255,255,255,0.14);
        --text: rgba(255,255,255,0.98);
        --muted: rgba(255,255,255,0.78);
        --muted2: rgba(255,255,255,0.62);
        --shadow: 0 12px 34px rgba(0,0,0,0.45);
        --radius: 18px;
    }

    .stApp {
        background: radial-gradient(1200px 650px at 15% 0%, rgba(124,58,237,0.30) 0%, rgba(7,10,18,0) 58%),
                    radial-gradient(1100px 650px at 92% 10%, rgba(37,99,235,0.24) 0%, rgba(7,10,18,0) 60%),
                    var(--bg);
        color: var(--text);
    }

    /* Force Calibri bold + white everywhere */
    html, body, [class*='css'], .stApp, .stMarkdown, .stText, .stCaption, label, p, span, div {
        font-family: Calibri, Arial, sans-serif !important;
        font-weight: 700 !important;
        color: var(--text) !important;
    }

    /* Clean header/toolbar */
    [data-testid='stHeader'], [data-testid='stToolbar'] { background: transparent; }

    /* Cards */
    .j-card {
        padding: 1.15rem 1.25rem;
        border-radius: var(--radius);
        border: 1px solid var(--stroke);
        background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.86) 100%);
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    .j-title { font-size: 1.85rem; font-weight: 800 !important; margin: 0; }
    .j-subtitle { margin-top: 0.35rem; color: var(--muted) !important; }

    .section-title { color: rgba(255,255,255,0.98) !important; font-family: Calibri, Arial, sans-serif !important; font-weight: 800 !important; }

    .metric-wrap {
        border-radius: var(--radius);
        padding: 1rem 1.1rem;
        border: 1px solid var(--stroke);
        background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.86) 100%);
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .rcpt-accent { background: linear-gradient(135deg, rgba(255,77,109,0.26) 0%, rgba(255,154,77,0.12) 55%, rgba(255,255,255,0.04) 100%); }
    .dme-accent  { background: linear-gradient(135deg, rgba(34,197,94,0.22) 0%, rgba(6,182,212,0.12) 55%, rgba(255,255,255,0.04) 100%); }

    .metric-title { font-size: 1.05rem; color: var(--muted) !important; margin: 0; }
    .metric-value { font-size: 2.05rem; font-weight: 800 !important; margin: 0.2rem 0 0 0; }
    .metric-sub { font-size: 0.95rem; color: var(--muted2) !important; margin: 0.25rem 0 0 0; }

    /* Inputs */
    .stNumberInput input {
        color: var(--text) !important;
        font-family: Calibri, Arial, sans-serif !important;
        font-weight: 700 !important;
        background-color: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        border-radius: 12px !important;
    }
    ::placeholder { color: rgba(255,255,255,0.55) !important; }

    /* Buttons */
    .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.18);
        background: linear-gradient(90deg, rgba(124,58,237,0.90) 0%, rgba(37,99,235,0.90) 100%);
        color: white !important;
        font-weight: 800 !important;
        padding: 0.65rem 1.05rem;
        box-shadow: 0 14px 26px rgba(37,99,235,0.18);
    }

    /* Expander */
    [data-testid='stExpander'] {
        border: 1px solid var(--stroke);
        border-radius: var(--radius);
        background: rgba(255,255,255,0.03);
    }

    /* Circular summary visuals */
    .circle-wrap { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.75rem; }
    .circle {
        width: 190px;
        height: 190px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .circle.rcpt {
        background: radial-gradient(circle at 30% 25%, rgba(255,77,109,0.18) 0%, rgba(255,255,255,0.92) 60%, rgba(255,255,255,0.86) 100%);
    }
    .circle.dme {
        background: radial-gradient(circle at 30% 25%, rgba(6,182,212,0.16) 0%, rgba(255,255,255,0.92) 60%, rgba(255,255,255,0.86) 100%);
    }
    .circle .label { font-size: 1.1rem; color: rgba(255,255,255,0.92) !important; margin-bottom: 0.35rem; }
    .circle .value { font-size: 1.95rem; font-weight: 800 !important; line-height: 1.05; }
    .circle .unit { font-size: 0.95rem; color: rgba(255,255,255,0.80) !important; margin-top: 0.35rem; }

    .input-circle { width: 145px; height: 145px; }
    .input-circle.c1 { background: radial-gradient(circle at 30% 25%, rgba(124,58,237,0.10) 0%, rgba(255,255,255,0.92) 62%, rgba(255,255,255,0.86) 100%); }
    .input-circle.c2 { background: radial-gradient(circle at 30% 25%, rgba(37,99,235,0.09) 0%, rgba(255,255,255,0.92) 62%, rgba(255,255,255,0.86) 100%); }
    .input-circle.c3 { background: radial-gradient(circle at 30% 25%, rgba(255,154,77,0.08) 0%, rgba(255,255,255,0.92) 62%, rgba(255,255,255,0.86) 100%); }
    .input-circle.c4 { background: radial-gradient(circle at 30% 25%, rgba(34,197,94,0.08) 0%, rgba(255,255,255,0.92) 62%, rgba(255,255,255,0.86) 100%); }
    .input-circle .label { font-size: 0.95rem; }
    .input-circle .value { font-size: 1.25rem; }
    .input-circle .unit { font-size: 0.85rem; color: rgba(255,255,255,0.78) !important; }
    
    /* White-box (card) text should be dark blue */
    .j-card, .j-card *,
    .metric-wrap, .metric-wrap *,
    .circle, .circle * {
        color: #0b2a5b !important;
    }

    /* Ensure values remain bold and readable inside boxes */
    .metric-value, .circle .value { color: #0b2a5b !important; font-weight: 800 !important; }
    .metric-title, .circle .label, .circle .unit, .metric-sub { color: #0b2a5b !important; }

    
    /* Input parameter boxes: make text dark for contrast on light background */
    .stNumberInput input {
        color: #0b2a5b !important;
        background-color: rgba(255,255,255,0.95) !important;
        border: 1px solid rgba(11,42,91,0.25) !important;
    }
    .stNumberInput label {
        color: rgba(255,255,255,0.98) !important;
    }

    
    /* Results boxes: use a dark navy background with white text for strong contrast */
    .metric-wrap {
        background: linear-gradient(180deg, rgba(7,18,44,0.96) 0%, rgba(10,24,60,0.92) 100%) !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
    }
    .metric-wrap, .metric-wrap * {
        color: rgba(255,255,255,0.98) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='j-card'><div class='j-title'>RCPT and DME Predictor for RAP Concrete</div><div class='j-subtitle'>(XGBoost regressor tuned with Optuna)</div></div>", unsafe_allow_html=True)

# Range getter

def rrow(key, field, default_val):
    if key in ranges_df.index and field in ranges_df.columns:
        try:
            return float(ranges_df.loc[key, field])
        except Exception:
            return float(default_val)
    return float(default_val)

dash_col_left, dash_col_right = st.columns([1.1, 0.9], gap='large')

with dash_col_left:
    st.markdown("<div class='section-title'>Input parameters</div>", unsafe_allow_html=True)

    in_col1, in_col2 = st.columns(2)

    with in_col1:
        c_val = st.number_input('C (kg/m³)', min_value=rrow('C', 'min', 0), max_value=rrow('C', 'max', 2000), value=rrow('C', 'mean', 300), step=1.0)
        fa_val = st.number_input('FA (kg/m³)', min_value=rrow('FA', 'min', 0), max_value=rrow('FA', 'max', 2000), value=rrow('FA', 'mean', 100), step=1.0)
        fa_binder_val = st.number_input('FA/Binder', min_value=rrow('FA/Binder', 'min', 0), max_value=rrow('FA/Binder', 'max', 10), value=rrow('FA/Binder', 'mean', 0.2), step=0.01, format='%.2f')
        a_val = st.number_input('A (kg/m³)', min_value=rrow('A', 'min', 0), max_value=rrow('A', 'max', 2000), value=rrow('A', 'mean', 150), step=1.0)

    with in_col2:
        t_crap_val = st.number_input('T CRAP (kg/m³)', min_value=rrow('T CRAP', 'min', 0), max_value=rrow('T CRAP', 'max', 2000), value=rrow('T CRAP', 'mean', 200), step=1.0)
        ca_val = st.number_input('CA (kg/m³)', min_value=rrow('CA', 'min', 0), max_value=rrow('CA', 'max', 3000), value=rrow('CA', 'mean', 900), step=1.0)
        age_val = st.number_input('Age (days)', min_value=rrow('Age', 'min', 1), max_value=rrow('Age', 'max', 3650), value=rrow('Age', 'mean', 28), step=1.0)

with dash_col_right:
    st.markdown("<div class='section-title'>Results</div>", unsafe_allow_html=True)
    predict_clicked = st.button('Predict RCPT and DME')

    if predict_clicked:
        input_dict = {
            'C': c_val,
            'FA': fa_val,
            'FA/Binder': fa_binder_val,
            'A': a_val,
            'T CRAP': t_crap_val,
            'CA': ca_val,
            'Age': age_val
        }

        X_input = pd.DataFrame([input_dict], columns=feature_cols)

        rcpt_pred = float(rcpt_model.predict(X_input)[0])

        dme_pred = None
        if dme_model is not None:
            dme_pred = float(dme_model.predict(X_input)[0])

        mcol1, mcol2 = st.columns(2)

        with mcol1:
            st.markdown(
                "<div class='metric-wrap rcpt-accent'><p class='metric-title'>RCPT</p><p class='metric-value'>" + format(rcpt_pred, ',.2f') + "</p><p class='metric-sub'>Coulombs</p></div>",
                unsafe_allow_html=True
            )

        with mcol2:
            if dme_pred is None:
                st.markdown(
                    "<div class='metric-wrap dme-accent'><p class='metric-title'>DME</p><p class='metric-value'>N/A</p><p class='metric-sub'>Missing model</p></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='metric-wrap dme-accent'><p class='metric-title'>DME</p><p class='metric-value'>" + format(dme_pred, ',.2f') + "</p><p class='metric-sub'>GPa</p></div>",
                    unsafe_allow_html=True
                )


# --- SHAP Option B section (RCPT + DME) ---
try:
    # Load artifacts from common variable names (fallback to disk)
    _feature_cols = None
    _ranges_df = None
    _rcpt_model = None
    _dme_model = None

    for _cand in ['feature_cols', 'feature_columns', 'FEATURE_COLS']:
        if _cand in globals():
            _feature_cols = globals()[_cand]
            break
    if _feature_cols is None:
        _feature_cols = joblib.load('feature_cols.pkl')

    for _cand in ['ranges_df', 'input_ranges_df', 'feature_ranges_df']:
        if _cand in globals():
            _ranges_df = globals()[_cand]
            break
    if _ranges_df is None:
        _ranges_df = pd.read_csv('input_feature_ranges.csv', index_col=0)

    for _cand in ['rcpt_model', 'model_rcpt', 'rcpt_xgb_model']:
        if _cand in globals():
            _rcpt_model = globals()[_cand]
            break
    if _rcpt_model is None:
        _rcpt_model = joblib.load('xgb_optuna_rcpt.pkl')

    for _cand in ['dme_model', 'model_dme', 'dme_xgb_model']:
        if _cand in globals():
            _dme_model = globals()[_cand]
            break
    if _dme_model is None:
        _dme_model = joblib.load('xgb_optuna_dme.pkl')

    X_bg = load_shap_background(_feature_cols, _ranges_df)
    rcpt_explainer, dme_explainer = build_explainers(_rcpt_model, _dme_model, X_bg)

    input_df_for_shap = None
    for _cand in ['input_df', 'X_input', 'df_input', 'input_data_df', 'X_test_single', 'X_one_row']:
        if _cand in globals():
            input_df_for_shap = globals()[_cand]
            break

    if input_df_for_shap is not None and isinstance(input_df_for_shap, pd.DataFrame) and len(input_df_for_shap) == 1:
        input_df_for_shap = input_df_for_shap[[c for c in _feature_cols if c in input_df_for_shap.columns]]
        render_shap_section('RCPT model', 'RCPT', rcpt_explainer, _feature_cols, input_df_for_shap, X_bg)
        if dme_explainer is not None:
            render_shap_section('DME model', 'DME', dme_explainer, _feature_cols, input_df_for_shap, X_bg)
except Exception as e:
    try:
        st.warning('SHAP section not available: ' + str(e))
    except Exception:
        pass
