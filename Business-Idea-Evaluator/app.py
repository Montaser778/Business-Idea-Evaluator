import streamlit as st  # استيراد مكتبة ستريم ليت
import operator # للعمليات المنطقية
import time # للتحكم في سرعة الرد
from typing import List, Annotated, Dict # لتحديد أنواع البيانات
from typing_extensions import TypedDict # للأنواع الهيكلية
from langchain_groq import ChatGroq # للربط مع Groq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage # أنواع الرسائل
from langgraph.graph.message import add_messages # لتجميع الرسائل
from langgraph.graph import StateGraph, END # لبناء الجراف

# --- 1. إعدادات التصميم الاحترافي ---
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {color: #00ffcc !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase;}
    .dev-credit {color: #00ffcc !important; font-size: 0.9rem; font-style: italic; margin-bottom: 2rem;}
    [data-testid="stChatMessage"] {background-color: #151515 !important; border: 1px solid #333 !important;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI STARTUP ENGINE")
st.markdown("<p class='dev-credit'>Developed by Eng. Montaser | Stateless Mode: No Conversation History Saved</p>", unsafe_allow_html=True)

if "GROQ_API_KEY" not in st.secrets:
    st.error("مفتاح GROQ_API_KEY مفقود!")
    st.stop()

# --- 2. هيكل الحالة (بدون حفظ) ---
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

# دالة استدعاء ذكية مع إدارة الضغط (Rate Limit Handling)
def safe_call(llm, prompt):
    for _ in range(3): 
        try:
            return llm.invoke([SystemMessage(content=prompt)])
        except Exception as e:
            if "429" in str(e): time.sleep(3) # انتظار في حال الضغط
            else: raise e
    return llm.invoke([SystemMessage(content=prompt)])

# --- 3. بناء المحرك (بدون MemorySaver) ---
@st.cache_resource
def get_graph():
    llm_fast = ChatGroq(model="llama-3.1-8b-instant", api_key=st.secrets["GROQ_API_KEY"])
    llm_deep = ChatGroq(model="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])
    
    def advisor_node(state: State, role: str):
        # معالجة ذكية بناءً على طول السؤال
        input_text = state['messages'][-1].content
        time.sleep(min(len(input_text) / 500, 2)) 
        prompt = f"Act as {role}. Analyze: {input_text}"
        res = safe_call(llm_fast, prompt)
        return {"advisor_reports": {role: res.content}}

    builder = StateGraph(State)
    builder.add_node("fanout", lambda s: {})
    for role in ["Market", "Legal", "Tech", "Strategy"]:
        builder.add_node(role, lambda s, r=role: advisor_node(s, r))
        builder.add_edge("fanout", role)
        builder.add_edge(role, "report")
        
    builder.add_node("report", lambda s: {"final_report": safe_call(llm_deep, f"Summarize: {s['advisor_reports']}").content})
    
    builder.set_entry_point("fanout")
    builder.add_edge("report", END)
    
    # Compile بدون Checkpointer = لا يوجد حفظ للجلسات
    return builder.compile() 

graph = get_graph()

# --- 4. التفاعل (Stateless Interaction) ---
if prompt := st.chat_input("أدخل فكرتك التجارية (سأحللها الآن)..."):
    st.chat_message("user").write(prompt)
    
    with st.status("🚀 جاري تهيئة العقول الاصطناعية...", expanded=True) as status:
        st.write("📊 تحليل السوق...")
        st.write("⚖️ المراجعة القانونية...")
        st.write("⚙️ الفحص التقني...")
        st.write("🎯 الاستراتيجية...")
        
        # عند كل invoke جديدة، نبدأ بـ state جديد تماماً، مما يمنع الحفظ
        result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
        
        status.update(label="✅ التحليل مكتمل.", state="complete")
        
    # عرض النتائج
    st.markdown("---")
    st.markdown("### 📑 التقرير النهائي")
    st.info(result["final_report"])
