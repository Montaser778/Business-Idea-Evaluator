import streamlit as st
import operator
import time
from typing import List, Annotated, Dict
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

# --- إعدادات التصميم ---
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {color: #00ffcc !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase;}
    .sub-title {color: #ffffff !important; font-size: 1.2rem;}
    .dev-credit {color: #00ffcc !important; font-size: 0.9rem; font-style: italic; margin-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI STARTUP ENGINE")
st.markdown("<p class='sub-title'>Advanced Multi-Agent Business Idea Analysis System</p>", unsafe_allow_html=True)
st.markdown("<p class='dev-credit'>Developed by Eng. Montaser</p>", unsafe_allow_html=True)

if "GROQ_API_KEY" not in st.secrets:
    st.error("مفتاح GROQ_API_KEY مفقود!")
    st.stop()

# --- تعريف الحالة ---
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

# دالة إعادة محاولة ذكية (تمنع خطأ 429)
def safe_call(llm, prompt):
    wait_time = 2
    for _ in range(5): # محاولة حتى 5 مرات
        try:
            return llm.invoke([SystemMessage(content=prompt)])
        except Exception as e:
            if "429" in str(e):
                time.sleep(wait_time)
                wait_time *= 2 # زيادة وقت الانتظار تدريجياً
            else: raise e
    return llm.invoke([SystemMessage(content=prompt)])

# --- بناء المحرك ---
@st.cache_resource
def get_graph():
    llm_fast = ChatGroq(model="llama-3.1-8b-instant", api_key=st.secrets["GROQ_API_KEY"])
    llm_deep = ChatGroq(model="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])
    
    def advisor_node(state: State, role: str):
        prompt = f"Act as {role}. Analyze: {state['messages'][-1].content}"
        res = safe_call(llm_fast, prompt)
        return {"advisor_reports": {role: res.content}}

    builder = StateGraph(State)
    # التنفيذ التسلسلي (بدلاً من التوازي)
    builder.add_node("Market", lambda s: advisor_node(s, "Market"))
    builder.add_node("Legal", lambda s: advisor_node(s, "Legal"))
    builder.add_node("Tech", lambda s: advisor_node(s, "Tech"))
    builder.add_node("Strategy", lambda s: advisor_node(s, "Strategy"))
    builder.add_node("report", lambda s: {"final_report": safe_call(llm_deep, f"Summarize: {s['advisor_reports']}").content})
    
    # الربط التسلسلي (أكثر استقراراً)
    builder.set_entry_point("Market")
    builder.add_edge("Market", "Legal")
    builder.add_edge("Legal", "Tech")
    builder.add_edge("Tech", "Strategy")
    builder.add_edge("Strategy", "report")
    builder.add_edge("report", END)
    
    return builder.compile()

graph = get_graph()

# --- التفاعل ---
if prompt := st.chat_input("أدخل فكرتك التجارية..."):
    st.chat_message("user").write(prompt)
    with st.status("🚀 جاري التحليل...", expanded=True) as status:
        st.write("📊 وكيل السوق...")
        result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
        st.write("⚖️ وكيل القانون...")
        st.write("⚙️ وكيل التقنية...")
        st.write("🎯 وكيل الاستراتيجية...")
        status.update(label="✅ التحليل مكتمل.", state="complete")
        
    st.markdown("---")
    st.markdown("### 📑 التقرير النهائي")
    st.info(result["final_report"])
