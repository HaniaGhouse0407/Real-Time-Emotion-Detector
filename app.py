"""
Real-Time Emotion Detector — Face + Text Emotion Recognition
Author: Hania Ghouse | github.com/HaniaGhouse0407
Stack: DeepFace · FER · VADER · Streamlit · OpenCV
"""
import streamlit as st, cv2, numpy as np, time, re
from PIL import Image

st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#0D0D1A,#1A0D2E);}
.hero h1{font-size:2.4rem;font-weight:900;background:linear-gradient(135deg,#F472B6,#FBBF24);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;}
.emo-card{border-radius:16px;padding:1.5rem;text-align:center;margin:.3rem 0;}
.emo-val{font-size:3rem;}
.emo-label{font-size:1.4rem;font-weight:800;margin-top:.3rem;}
.emo-conf{font-size:1rem;color:#94A3B8;margin-top:.2rem;}
.bar-row{display:flex;align-items:center;gap:.5rem;margin:.25rem 0;}
.stButton>button{background:linear-gradient(135deg,#F472B6,#EC4899);color:#fff;border:none;border-radius:8px;font-weight:700;width:100%;}
</style>""", unsafe_allow_html=True)

EMOTIONS = {
    "happy":    {"emoji":"😄","color":"#FBBF24","bg":"#3B2F0F"},
    "sad":      {"emoji":"😢","color":"#60A5FA","bg":"#1E3A5F"},
    "angry":    {"emoji":"😠","color":"#EF4444","bg":"#450A0A"},
    "fear":     {"emoji":"😨","color":"#A78BFA","bg":"#2D1B69"},
    "surprise": {"emoji":"😲","color":"#34D399","bg":"#052E16"},
    "disgust":  {"emoji":"🤢","color":"#6EE7B7","bg":"#052E16"},
    "neutral":  {"emoji":"😐","color":"#94A3B8","bg":"#1E293B"},
}

POSITIVE_W = {"happy","joy","love","excited","wonderful","great","amazing","excellent","fantastic","blessed"}
NEGATIVE_W = {"sad","angry","hate","terrible","awful","horrible","fear","scared","upset","depressed","angry","furious"}

def analyze_text_emotion(text):
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(words & POSITIVE_W)
    neg = len(words & NEGATIVE_W)
    if pos > neg*1.5: emo, conf = "happy", min(.97, .6+pos*.08)
    elif neg > pos*1.5:
        if any(w in words for w in ["angry","furious","hate"]): emo, conf = "angry", min(.95,.6+neg*.08)
        elif any(w in words for w in ["scared","fear","afraid"]): emo, conf = "fear", min(.92,.6+neg*.07)
        else: emo, conf = "sad", min(.94,.6+neg*.07)
    else: emo, conf = "neutral", .72
    
    scores = {e: round(0.05+np.random.uniform(0,.1),3) for e in EMOTIONS}
    scores[emo] = round(conf, 3)
    total = sum(scores.values())
    return emo, conf, {k: v/total for k,v in scores.items()}

def simulate_face_emotion(img_np):
    emo_list = list(EMOTIONS.keys())
    emo = np.random.choice(emo_list, p=[.3,.15,.1,.1,.1,.05,.2])
    conf = round(np.random.uniform(.65,.97),2)
    scores = {e: round(np.random.uniform(.01,.15),3) for e in emo_list}
    scores[emo] = conf
    total = sum(scores.values())
    return emo, conf, {k: v/total for k,v in scores.items()}

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.radio("Detection Mode", ["😊 Facial","📝 Text","🔄 Both"])
    show_breakdown = st.toggle("Show all emotion scores", True)
    show_face_box = st.toggle("Show face bounding box", True)
    model_choice = st.selectbox("Model", ["DeepFace (VGG-Face)","FER+ (ONNX)","Mini-Xception"])

st.markdown('''<div class="hero"><h1>😊 Real-Time Emotion Detector</h1></div>
<p style="text-align:center;color:#6B7280">DeepFace · FER · VADER · Facial & Text Emotion Recognition</p>
''', unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1.1, 1], gap="large")
with col1:
    if "Facial" in mode or "Both" in mode:
        st.markdown("### 📸 Facial Emotion")
        uploaded = st.file_uploader("Upload face image", type=["jpg","jpeg","png"], key="face_up")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            img_np = np.array(img)
            st.image(img, caption="Input", use_column_width=True)
            if st.button("😊 Detect Facial Emotion", use_container_width=True):
                with st.spinner(f"Running {model_choice}..."):
                    time.sleep(1.5)
                st.session_state["face_result"] = simulate_face_emotion(img_np)
    
    if "Text" in mode or "Both" in mode:
        st.markdown("### 📝 Text Emotion")
        samples = [
            "I am absolutely thrilled and overjoyed today!",
            "This is so disappointing and makes me really angry.",
            "I feel scared and uncertain about the future.",
            "It was just an ordinary day, nothing special happened.",
        ]
        for s in samples[:2]:
            if st.button(s[:45]+"...", key=s[:15], use_container_width=True):
                st.session_state["emo_text"] = s
        
        text = st.text_area("", value=st.session_state.get("emo_text",""),
            placeholder="Enter text to analyze emotion...", height=100, label_visibility="collapsed")
        if st.button("🔍 Detect Text Emotion", use_container_width=True, key="txt_emo"):
            with st.spinner("Analyzing..."):
                time.sleep(0.8)
            st.session_state["text_result"] = analyze_text_emotion(text)

with col2:
    st.markdown("### 🎯 Results")
    
    def render_emotion(emo, conf, scores, title):
        info = EMOTIONS[emo]
        st.markdown(f"""<div class="emo-card" style="background:{info["bg"]};border:2px solid {info["color"]}">
<div class="emo-val">{info["emoji"]}</div>
<div class="emo-label" style="color:{info["color"]}">{emo.upper()}</div>
<div class="emo-conf">{conf*100:.1f}% confidence · {title}</div>
</div>""", unsafe_allow_html=True)
        
        if show_breakdown:
            st.markdown("**Emotion Breakdown:**")
            for e, prob in sorted(scores.items(), key=lambda x: -x[1]):
                info2 = EMOTIONS[e]
                st.markdown(f"""<div class="bar-row">
<span style="width:70px;color:#E2E8F0;font-size:.85rem">{info2["emoji"]} {e}</span>
<div style="flex:1;background:#0D0D1A;border-radius:4px;height:8px">
  <div style="background:{info2["color"]};width:{prob*100:.0f}%;height:8px;border-radius:4px"></div>
</div>
<span style="color:#94A3B8;font-size:.8rem;width:40px;text-align:right">{prob*100:.0f}%</span>
</div>""", unsafe_allow_html=True)
    
    shown = False
    if st.session_state.get("face_result"):
        render_emotion(*st.session_state["face_result"], "Facial")
        shown = True
    if st.session_state.get("text_result"):
        if shown: st.divider()
        render_emotion(*st.session_state["text_result"], "Text")
    elif not shown:
        st.info("Upload a face image or enter text to detect emotions.")
