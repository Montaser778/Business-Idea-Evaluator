import streamlit as st  # استيراد مكتبة ستريم ليت
import operator  # استيراد مكتبة العمليات المنطقية
from typing import List, Annotated, Dict  # استيراد أنواع البيانات
from typing_extensions import TypedDict  # استيراد الأنواع الهيكلية
from langchain_groq import ChatGroq  # استيراد موديلات Groq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage  # استيراد الرسائل
from langgraph.graph.message import add_messages  # استيراد مُقلل الرسائل
from langgraph.graph import StateGraph, END  # استيراد هيكل الجراف

# --- 1. إعدادات الصفحة والتصميم ---
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {color: #00ffcc !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase; margin-bottom: 5px;}
    .sub-title {color: #ffffff !important; font-size: 1.2rem; margin-bottom: 5px;}
    .dev-credit {color: #00ffcc !important; font-size: 0.9rem; font-style: italic; margin-bottom: 2rem;}
    [data-testid="stChatMessage"] {background-color: #151515 !important; border: 1px solid #333 !important;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI STARTUP ENGINE") 
# العنوان المتقدم كما طلبت
st.markdown("<p class='sub-title'>Advanced Multi-Agent Business Idea Analysis System</p>", unsafe_allow_html=True)
# التوقيع في سطر جديد
st.markdown("<p class='dev-credit'>Developed by Eng. Montaser</p>", unsafe_allow_html=True)

if "GROQ_API_KEY" not in st.secrets:
    st.error("مفتاح GROQ_API_KEY مفقود!")
    st.stop()

# --- 2. تعريف الحالة (State) ---
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

# --- 3. بناء المحرك (بدون تأخير وبدون ذاكرة) ---
@st.cache_resource
def get_graph():
    # استخدام موديلات أسرع لتجنب أخطاء الضغط (Rate Limits)
    llm_fast = ChatGroq(model="llama-3.1-8b-instant", api_key=st.secrets["GROQ_API_KEY"])
    llm_deep = ChatGroq(model="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])
    
    def advisor_node(state: State, role: str):
        prompt = f"Act as a senior {role} expert. Analyze: {state['messages'][-1].content}"
        res = llm_fast.invoke([SystemMessage(content=prompt)])
        return {"advisor_reports": {role: res.content}}

    builder = StateGraph(State)
    builder.add_node("fanout", lambda s: {})
    for role in ["Market", "Legal", "Tech", "Strategy"]:
        builder.add_node(role, lambda s, r=role: advisor_node(s, r))
        builder.add_edge("fanout", role)
        builder.add_edge(role, "report")
        
    builder.add_node("report", lambda s: {"final_report": llm_deep.invoke([HumanMessage(content=str(s['advisor_reports']))]).content})
    
    builder.set_entry_point("fanout")
    builder.add_edge("report", END)
    return builder.compile() # بدون Checkpointer (Stateless)

graph = get_graph()

# --- 4. التفاعل السريع ---
if prompt := st.chat_input("أدخل فكرتك التجارية..."):
    st.chat_message("user").write(prompt)
    
    with st.status("🚀 جاري التحليل...", expanded=True) as status:
        st.write("📊 وكيل السوق...")
        st.write("⚖️ وكيل القانون...")
        st.write("⚙️ وكيل التقنية...")
        st.write("🎯 وكيل الاستراتيجية...")
        
        # تنفيذ المحرك مباشرة
        result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
        
        status.update(label="✅ التحليل جاهز.", state="complete")
        
    st.markdown("---")
    st.markdown("### 📑 التقرير النهائي")
    st.info(result["final_report"])
