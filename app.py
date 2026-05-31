import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict AI | Student Performance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.4);
}
.metric-value { font-size: 2.2rem; font-weight: 800; }
.metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 2px; }

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    box-shadow: 0 20px 60px rgba(102,126,234,0.35);
    text-align: center;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
    margin-top: 8px;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white;
    border-radius: 50px;
    padding: 4px 16px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 14px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Section Headers */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(167,139,250,0.3);
}

/* Prediction Result */
.pred-box-H {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 16px 40px rgba(17,153,142,0.4);
}
.pred-box-M {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 16px 40px rgba(247,151,30,0.4);
}
.pred-box-L {
    background: linear-gradient(135deg, #c94b4b, #4b134f);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 16px 40px rgba(201,75,75,0.4);
}
.pred-grade { font-size: 4rem; font-weight: 900; color: white; }
.pred-label { font-size: 1.3rem; font-weight: 700; color: rgba(255,255,255,0.9); }
.pred-prob  { font-size: 0.95rem; color: rgba(255,255,255,0.75); margin-top: 6px; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: all 0.25s !important;
    box-shadow: 0 6px 24px rgba(102,126,234,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(102,126,234,0.6) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #9ca3af !important;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}

/* Footer */
.footer {
    text-align: center;
    color: rgba(255,255,255,0.4);
    font-size: 0.82rem;
    padding: 20px 0 10px;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 40px;
}
.footer span { color: #a78bfa; font-weight: 600; }

/* Info box */
.tip-box {
    background: rgba(167,139,250,0.12);
    border-left: 4px solid #a78bfa;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 0.88rem;
    color: #c4b5fd;
}

/* Selectbox, Sliders label */
label { color: #d1d5db !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Resources ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("knn_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("xAPI-Edu-Data.csv")

model = load_model()
df = load_data()

# ── Feature Encoding ───────────────────────────────────────────────────────────
FEATURE_NAMES = list(model.feature_names_in_)
CLASS_MAP = {0: "L", 1: "M", 2: "H"}
CLASS_LABELS = {"H": "High Performer", "M": "Mid-Level", "L": "Needs Support"}
CLASS_EMOJI  = {"H": "🏆", "M": "📘", "L": "⚠️"}

def encode_input(data: dict) -> pd.DataFrame:
    row = {f: 0 for f in FEATURE_NAMES}
    row['raisedhands']        = data['raisedhands']
    row['VisITedResources']   = data['VisITedResources']
    row['AnnouncementsView']  = data['AnnouncementsView']
    row['Discussion']         = data['Discussion']
    cats = {
        'gender': data['gender'],
        'NationalITy': data['NationalITy'],
        'PlaceofBirth': data['PlaceofBirth'],
        'StageID': data['StageID'],
        'GradeID': data['GradeID'],
        'SectionID': data['SectionID'],
        'Topic': data['Topic'],
        'Semester': data['Semester'],
        'Relation': data['Relation'],
        'ParentAnsweringSurvey': data['ParentAnsweringSurvey'],
        'ParentschoolSatisfaction': data['ParentschoolSatisfaction'],
        'StudentAbsenceDays': data['StudentAbsenceDays'],
    }
    for col, val in cats.items():
        key = f"{col}_{val}"
        if key in row:
            row[key] = 1
    return pd.DataFrame([row])


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-size:3rem;'>🎓</div>
        <div style='font-size:1.2rem; font-weight:700; color:#a78bfa;'>EduPredict AI</div>
        <div style='font-size:0.75rem; color:#6b7280; margin-top:4px;'>Student Performance System</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Navigation", ["🏠 Home", "🔮 Predict", "📊 Analytics", "📋 Dataset", "ℹ️ About"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1)'>", unsafe_allow_html=True)

    total    = len(df)
    high     = (df['Class'] == 'H').sum()
    mid      = (df['Class'] == 'M').sum()
    low      = (df['Class'] == 'L').sum()

    st.markdown(f"""
    <div style='font-size:0.8rem; color:#9ca3af; font-weight:600; margin-bottom:8px; letter-spacing:1px;'>DATASET SNAPSHOT</div>
    <div class='metric-card' style='margin-bottom:10px;'>
        <div class='metric-value' style='color:#a78bfa;'>{total}</div>
        <div class='metric-label'>Total Students</div>
    </div>
    <div style='display:flex; gap:8px;'>
        <div class='metric-card' style='flex:1; text-align:center;'>
            <div style='font-size:1.3rem; font-weight:800; color:#34d399;'>{high}</div>
            <div style='font-size:0.72rem; color:#9ca3af;'>High</div>
        </div>
        <div class='metric-card' style='flex:1; text-align:center;'>
            <div style='font-size:1.3rem; font-weight:800; color:#fbbf24;'>{mid}</div>
            <div style='font-size:0.72rem; color:#9ca3af;'>Mid</div>
        </div>
        <div class='metric-card' style='flex:1; text-align:center;'>
            <div style='font-size:1.3rem; font-weight:800; color:#f87171;'>{low}</div>
            <div style='font-size:0.72rem; color:#9ca3af;'>Low</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='footer'>
        Made with ❤️ by <span>RAHUL THAKUR</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if "🏠 Home" in menu:
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>🎓 EduPredict AI</div>
        <div class='hero-sub'>Intelligent Student Academic Performance Prediction System</div>
        <div class='hero-badge'>Powered by KNN · xAPI Edu Dataset</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("🧑‍🎓", str(total), "Total Students", "#a78bfa"),
        ("✋", f"{df['raisedhands'].mean():.1f}", "Avg Raised Hands", "#60a5fa"),
        ("💻", f"{df['VisITedResources'].mean():.1f}", "Avg Resources Used", "#34d399"),
        ("📢", f"{df['AnnouncementsView'].mean():.1f}", "Avg Announcements", "#fbbf24"),
    ]
    for col, (icon, val, label, color) in zip([c1,c2,c3,c4], cards):
        col.markdown(f"""
        <div class='metric-card' style='text-align:center;'>
            <div style='font-size:2rem;'>{icon}</div>
            <div class='metric-value' style='color:{color};'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>📊 Class Distribution</div>", unsafe_allow_html=True)
        class_counts = df['Class'].value_counts()
        fig = px.pie(values=class_counts.values, names=class_counts.index,
                     color=class_counts.index,
                     color_discrete_map={"H":"#34d399","M":"#fbbf24","L":"#f87171"},
                     hole=0.55)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#e0e0e0', legend_font_size=13,
                          margin=dict(t=20, b=20))
        fig.update_traces(textfont_size=14, textfont_color='white')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-header'>🔢 Engagement by Class</div>", unsafe_allow_html=True)
        eng = df.groupby('Class')[['raisedhands','VisITedResources','AnnouncementsView','Discussion']].mean().reset_index()
        fig2 = px.bar(eng.melt(id_vars='Class', var_name='Feature', value_name='Average'),
                      x='Feature', y='Average', color='Class', barmode='group',
                      color_discrete_map={"H":"#34d399","M":"#fbbf24","L":"#f87171"})
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#e0e0e0', xaxis_tickangle=-20,
                           legend_title_text='Class', margin=dict(t=20,b=20))
        fig2.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
        fig2.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>📌 How It Works</div>", unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    steps = [
        ("1️⃣", "Input Data", "Fill in student demographic and engagement details"),
        ("2️⃣", "Feature Encoding", "Data is one-hot encoded to match model input"),
        ("3️⃣", "KNN Prediction", "K=7 Nearest Neighbors model classifies the student"),
        ("4️⃣", "View Results", "Get class prediction (H/M/L) with probability scores"),
    ]
    for col, (num, title, desc) in zip([h1,h2,h3,h4], steps):
        col.markdown(f"""
        <div class='metric-card' style='text-align:center; height:140px;'>
            <div style='font-size:1.6rem;'>{num}</div>
            <div style='font-weight:700; color:#a78bfa; margin:6px 0 4px;'>{title}</div>
            <div style='font-size:0.8rem; color:#9ca3af;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif "🔮 Predict" in menu:
    st.markdown("""
    <div class='hero-banner' style='padding:24px 32px;'>
        <div style='font-size:1.8rem; font-weight:800; color:white;'>🔮 Student Performance Predictor</div>
        <div style='color:rgba(255,255,255,0.8); margin-top:6px;'>Fill in the details below to get an AI-powered prediction</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='tip-box'>💡 Fill all sections carefully. Sliders represent engagement counts (0–100).</div>", unsafe_allow_html=True)

    with st.form("predict_form"):
        # ── Section 1: Demographics
        st.markdown("<div class='section-header'>👤 Student Demographics</div>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        gender    = d1.selectbox("Gender", ["M", "F"])
        nationality = d2.selectbox("Nationality", sorted(['KW','Jordan','Palestine','Iraq','Lebanon','Tunis','SaudiArabia','Egypt','Syria','USA','Iran','Lybia','Morocco','venzuela']))
        place     = d3.selectbox("Place of Birth", sorted(['KuwaIT','Jordan','Palestine','Iraq','Lebanon','Tunis','SaudiArabia','Egypt','Syria','USA','Iran','Lybia','Morocco','venzuela']))

        d4, d5, d6 = st.columns(3)
        stage     = d4.selectbox("Stage", ["lowerlevel", "MiddleSchool", "HighSchool"])
        grade     = d5.selectbox("Grade", ['G-02','G-04','G-05','G-06','G-07','G-08','G-09','G-10','G-11','G-12'])
        section   = d6.selectbox("Section", ["A", "B", "C"])

        d7, d8 = st.columns(2)
        topic     = d7.selectbox("Topic / Subject", sorted(['Arabic','Biology','Chemistry','English','French','Geology','History','IT','Math','Quran','Science','Spanish']))
        semester  = d8.selectbox("Semester", ["F", "S"])

        # ── Section 2: Family & Attendance
        st.markdown("<div class='section-header'>👨‍👩‍👧 Family & Attendance</div>", unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        relation  = f1.selectbox("Responsible Parent", ["Father", "Mum"])
        parent_survey = f2.selectbox("Parent Answers Survey", ["Yes", "No"])
        parent_sat    = f3.selectbox("School Satisfaction", ["Good", "Bad"])
        absence   = f4.selectbox("Absence Days", ["Under-7", "Above-7"])

        # ── Section 3: Engagement Metrics
        st.markdown("<div class='section-header'>📈 Engagement Metrics</div>", unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        raised   = e1.slider("✋ Raised Hands", 0, 100, 50)
        visited  = e2.slider("💻 Visited Resources", 0, 100, 50)
        e3, e4 = st.columns(2)
        announce = e3.slider("📢 Announcements Viewed", 0, 100, 50)
        discuss  = e4.slider("💬 Discussion Participation", 0, 100, 30)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Predict Performance")

    if submitted:
        input_data = {
            'gender': gender, 'NationalITy': nationality, 'PlaceofBirth': place,
            'StageID': stage, 'GradeID': grade, 'SectionID': section,
            'Topic': topic, 'Semester': semester, 'Relation': relation,
            'ParentAnsweringSurvey': parent_survey,
            'ParentschoolSatisfaction': parent_sat,
            'StudentAbsenceDays': absence,
            'raisedhands': raised, 'VisITedResources': visited,
            'AnnouncementsView': announce, 'Discussion': discuss,
        }
        X = encode_input(input_data)
        pred_int  = model.predict(X)[0]
        proba     = model.predict_proba(X)[0]
        pred_cls  = CLASS_MAP[pred_int]
        label     = CLASS_LABELS[pred_cls]
        emoji     = CLASS_EMOJI[pred_cls]
        box_cls   = f"pred-box-{pred_cls}"
        conf      = proba[pred_int] * 100

        r1, r2 = st.columns([1, 1.5])
        with r1:
            st.markdown(f"""
            <div class='{box_cls}'>
                <div style='font-size:2.5rem;'>{emoji}</div>
                <div class='pred-grade'>{pred_cls}</div>
                <div class='pred-label'>{label}</div>
                <div class='pred-prob'>Confidence: {conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown("<div class='section-header'>📊 Class Probabilities</div>", unsafe_allow_html=True)
            prob_df = pd.DataFrame({
                'Class': [CLASS_MAP[i] for i in range(3)],
                'Probability': [p * 100 for p in proba],
                'Label': [CLASS_LABELS[CLASS_MAP[i]] for i in range(3)]
            })
            fig = px.bar(prob_df, x='Class', y='Probability',
                         color='Class',
                         color_discrete_map={"H":"#34d399","M":"#fbbf24","L":"#f87171"},
                         text=prob_df['Probability'].apply(lambda x: f"{x:.1f}%"))
            fig.update_traces(textposition='outside', textfont_color='white')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#e0e0e0', showlegend=False,
                              yaxis_title='Probability (%)', margin=dict(t=10,b=10))
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.06)', range=[0, 110])
            st.plotly_chart(fig, use_container_width=True)

        # Engagement radar
        st.markdown("<div class='section-header'>🕸️ Student Engagement Profile</div>", unsafe_allow_html=True)
        avg = df.groupby('Class')[['raisedhands','VisITedResources','AnnouncementsView','Discussion']].mean()
        cats = ['Raised Hands', 'Visited Resources', 'Announcements', 'Discussion']
        student_vals = [raised, visited, announce, discuss]
        fig3 = go.Figure()
        colors_r = {"H":"#34d399","M":"#fbbf24","L":"#f87171"}
        for cls in ['H','M','L']:
            vals = avg.loc[cls].tolist()
            fig3.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                                           name=f"Avg {cls}", line=dict(color=colors_r[cls], width=1.5),
                                           fill='toself', opacity=0.25))
        fig3.add_trace(go.Scatterpolar(r=student_vals+[student_vals[0]], theta=cats+[cats[0]],
                                       name="This Student", line=dict(color="#a78bfa", width=3),
                                       fill='toself', opacity=0.4, marker=dict(size=8)))
        fig3.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100],
                                                       gridcolor='rgba(255,255,255,0.1)',
                                                       linecolor='rgba(255,255,255,0.1)'),
                                      angularaxis=dict(linecolor='rgba(255,255,255,0.1)'),
                                      bgcolor='rgba(0,0,0,0)'),
                           paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0',
                           legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif "📊 Analytics" in menu:
    st.markdown("""
    <div class='hero-banner' style='padding:24px 32px;'>
        <div style='font-size:1.8rem; font-weight:800; color:white;'>📊 Data Analytics Dashboard</div>
        <div style='color:rgba(255,255,255,0.8); margin-top:6px;'>Deep insights from the xAPI Edu Dataset</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🏫 Academics", "👨‍👩‍👧 Family", "🔗 Correlations"])

    with tab1:
        st.markdown("<div class='section-header'>Engagement Score Distributions</div>", unsafe_allow_html=True)
        for metric, label in [('raisedhands','Raised Hands'), ('VisITedResources','Visited Resources'),
                               ('AnnouncementsView','Announcements Viewed'), ('Discussion','Discussion')]:
            fig = px.histogram(df, x=metric, color='Class',
                               color_discrete_map={"H":"#34d399","M":"#fbbf24","L":"#f87171"},
                               title=f"{label} Distribution by Class", nbins=30, barmode='overlay', opacity=0.75)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#e0e0e0', title_font_size=15)
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'>📚 Subject vs Performance</div>", unsafe_allow_html=True)
            topic_cls = df.groupby(['Topic','Class']).size().reset_index(name='Count')
            fig = px.bar(topic_cls, x='Topic', y='Count', color='Class', barmode='stack',
                         color_discrete_map={"H":"#34d399","M":"#fbbf24","L":"#f87171"})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#e0e0e0', xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>🏫 Stage vs Performance</div>", unsafe_allow_html=True)
            stage_cls = df.groupby(['StageID','Class']).size().reset_index(name='Count')
            fig = px.bar(stage_cls, x='StageID', y='Count', color='Class', barmode='group',
                         color_discrete_map={"H":"#34d399","M":"#fbbf24","L":"#f87171"})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("<div class='section-header'>📅 Semester Distribution</div>", unsafe_allow_html=True)
            sem_cls = df.groupby(['Semester','Class']).size().reset_index(name='Count')
            fig = px.pie(sem_cls, values='Count', names='Semester', hole=0.5)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            st.markdown("<div class='section-header'>🎓 Absence vs Class</div>", unsafe_allow_html=True)
            abs_cls = df.groupby(['StudentAbsenceDays','Class']).size().reset_index(name='Count')
            fig = px.bar(abs_cls, x='StudentAbsenceDays', y='Count', color='Class', barmode='group',
                         color_discrete_map={"H":"#34d399","M":"#fbbf24","L":"#f87171"})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'>👨‍👩‍👧 Parent Survey Impact</div>", unsafe_allow_html=True)
            ps = df.groupby(['ParentAnsweringSurvey','Class']).size().reset_index(name='Count')
            fig = px.bar(ps, x='ParentAnsweringSurvey', y='Count', color='Class', barmode='group',
                         color_discrete_map={"H":"#34d399","M":"#fbbf24","L":"#f87171"})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>😊 School Satisfaction</div>", unsafe_allow_html=True)
            sat = df.groupby(['ParentschoolSatisfaction','Class']).size().reset_index(name='Count')
            fig = px.bar(sat, x='ParentschoolSatisfaction', y='Count', color='Class', barmode='group',
                         color_discrete_map={"H":"#34d399","M":"#fbbf24","L":"#f87171"})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("<div class='section-header'>⚧ Gender Distribution</div>", unsafe_allow_html=True)
            gen_cls = df.groupby(['gender','Class']).size().reset_index(name='Count')
            fig = px.bar(gen_cls, x='gender', y='Count', color='Class', barmode='group',
                         color_discrete_map={"H":"#34d399","M":"#fbbf24","L":"#f87171"})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            st.markdown("<div class='section-header'>👨‍👩‍👧 Responsible Parent</div>", unsafe_allow_html=True)
            rel = df.groupby(['Relation','Class']).size().reset_index(name='Count')
            fig = px.pie(rel, values='Count', names='Relation', hole=0.5)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("<div class='section-header'>🔗 Feature Correlation Heatmap</div>", unsafe_allow_html=True)
        num_df = df[['raisedhands','VisITedResources','AnnouncementsView','Discussion']].copy()
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='Purples',
                        aspect='auto', title="Correlation Between Engagement Metrics")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0',
                          title_font_size=15)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>📦 Box Plot: Engagement by Class</div>", unsafe_allow_html=True)
        for col in ['raisedhands','VisITedResources','AnnouncementsView','Discussion']:
            fig = px.box(df, x='Class', y=col, color='Class',
                         color_discrete_map={"H":"#34d399","M":"#fbbf24","L":"#f87171"},
                         title=col)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#e0e0e0', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════
elif "📋 Dataset" in menu:
    st.markdown("""
    <div class='hero-banner' style='padding:24px 32px;'>
        <div style='font-size:1.8rem; font-weight:800; color:white;'>📋 Raw Dataset Explorer</div>
        <div style='color:rgba(255,255,255,0.8); margin-top:6px;'>Browse, filter, and export the xAPI Edu Dataset</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        class_filter = st.multiselect("Filter by Class", ["H", "M", "L"], default=["H","M","L"])
    with col2:
        stage_filter = st.multiselect("Filter by Stage", df['StageID'].unique().tolist(), default=df['StageID'].unique().tolist())
    with col3:
        gender_filter = st.multiselect("Filter by Gender", ["M","F"], default=["M","F"])

    filtered = df[df['Class'].isin(class_filter) & df['StageID'].isin(stage_filter) & df['gender'].isin(gender_filter)]

    st.markdown(f"""
    <div class='tip-box'>Showing <b>{len(filtered)}</b> of <b>{total}</b> students</div>
    """, unsafe_allow_html=True)

    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=420)

    st.markdown("<div class='section-header'>📊 Statistical Summary</div>", unsafe_allow_html=True)
    st.dataframe(filtered.describe().round(2), use_container_width=True)

    csv_export = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered CSV", data=csv_export,
                       file_name="filtered_students.csv", mime="text/csv",
                       use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif "ℹ️ About" in menu:
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>ℹ️ About EduPredict AI</div>
        <div class='hero-sub'>Everything you need to know about this application</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        <div class='section-header'>🎯 Project Overview</div>
        <div class='metric-card' style='margin-bottom:16px;'>
            <p style='color:#d1d5db; line-height:1.8;'>
            <b style='color:#a78bfa;'>EduPredict AI</b> is an intelligent web application that uses
            machine learning to predict student academic performance based on
            behavioral, demographic, and engagement data collected via the <b>xAPI (Experience API)</b> protocol.
            </p>
            <p style='color:#d1d5db; line-height:1.8;'>
            The model classifies students into three performance bands:
            <span style='color:#34d399; font-weight:700;'>High (H)</span>,
            <span style='color:#fbbf24; font-weight:700;'>Medium (M)</span>, and
            <span style='color:#f87171; font-weight:700;'>Low (L)</span>,
            enabling educators to proactively identify students who may need additional support.
            </p>
        </div>

        <div class='section-header'>🤖 Model Details</div>
        <div class='metric-card'>
            <table style='width:100%; color:#d1d5db; border-collapse:collapse;'>
                <tr><td style='padding:8px 0; color:#9ca3af;'>Algorithm</td><td style='font-weight:600; color:#a78bfa;'>K-Nearest Neighbors (KNN)</td></tr>
                <tr><td style='padding:8px 0; color:#9ca3af;'>K Value</td><td style='font-weight:600; color:#a78bfa;'>7</td></tr>
                <tr><td style='padding:8px 0; color:#9ca3af;'>Input Features</td><td style='font-weight:600; color:#a78bfa;'>72 (after one-hot encoding)</td></tr>
                <tr><td style='padding:8px 0; color:#9ca3af;'>Output Classes</td><td style='font-weight:600; color:#a78bfa;'>L (0), M (1), H (2)</td></tr>
                <tr><td style='padding:8px 0; color:#9ca3af;'>Dataset Size</td><td style='font-weight:600; color:#a78bfa;'>480 students</td></tr>
                <tr><td style='padding:8px 0; color:#9ca3af;'>Dataset Source</td><td style='font-weight:600; color:#a78bfa;'>xAPI-Edu-Data (Kaggle)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='section-header'>✨ App Features</div>
        """, unsafe_allow_html=True)
        features = [
            ("🔮", "Real-Time Prediction", "Instant student class prediction with confidence scores"),
            ("📊", "Rich Analytics", "Interactive charts covering all dataset dimensions"),
            ("🕸️", "Radar Profile", "Visual engagement comparison against class averages"),
            ("📋", "Dataset Explorer", "Filter, browse and export the full dataset"),
            ("🌐", "Fully Responsive", "Optimized for all screen sizes"),
            ("⚡", "Cached Resources", "Fast loading with Streamlit caching"),
        ]
        for icon, title, desc in features:
            st.markdown(f"""
            <div class='metric-card' style='display:flex; gap:14px; align-items:flex-start; margin-bottom:10px;'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div>
                    <div style='font-weight:700; color:#a78bfa;'>{title}</div>
                    <div style='font-size:0.82rem; color:#9ca3af; margin-top:2px;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(102,126,234,0.1));
                border: 1px solid rgba(167,139,250,0.3);
                border-radius: 20px;
                padding: 36px;
                text-align: center;'>
        <div style='font-size:3.5rem;'>👨‍💻</div>
        <div style='font-size:1.6rem; font-weight:800; color:#a78bfa; margin-top:10px;'>RAHUL THAKUR</div>
        <div style='color:#9ca3af; margin-top:6px;'>Developer & Data Scientist</div>
        <div style='margin-top:16px; color:#d1d5db; font-size:0.9rem; max-width:500px; margin-left:auto; margin-right:auto;'>
            Built with ❤️ using Python, Streamlit, Scikit-learn, and Plotly.<br>
            Designed to empower educators through data-driven insights.
        </div>
        <div style='margin-top:18px;'>
            <span style='background:rgba(167,139,250,0.2); color:#a78bfa; border-radius:50px;
                         padding:6px 18px; font-size:0.8rem; font-weight:600;'>
                Created by RAHUL THAKUR
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Global Footer ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🎓 EduPredict AI &nbsp;|&nbsp; Created by <span>RAHUL THAKUR</span> &nbsp;|&nbsp; Powered by KNN & xAPI Edu Dataset
</div>
""", unsafe_allow_html=True)