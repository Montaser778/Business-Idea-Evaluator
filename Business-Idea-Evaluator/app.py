import streamlit as st  # استيراد مكتبة ستريم ليت للواجهة
import operator  # استيراد مكتبة العمليات المنطقية
import time  # استيراد مكتبة الوقت للتحكم في فترات الانتظار
from typing import List, Annotated, Dict  # استيراد أدوات تحديد الأنواع
from typing_extensions import TypedDict  # استيراد الأنواع الهيكلية
from langchain_groq import ChatGroq  # استيراد مكتبة الربط مع موديلات Groq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage  # استيراد أنواع الرسائل
from langgraph.graph.message import add_messages  # استيراد مُقلل الرسائل
from langgraph.graph import StateGraph, END  # استيراد هيكل الجراف

# --- 1. إعدادات الصفحة والتصميم ---
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

# تصميم CSS احترافي (Dark UI)
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {color: #00ffcc !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase; margin-bottom: 5px;}
    .sub-title {color: #ffffff !important; font-size: 1.2rem; margin-bottom: 5px;}
    .dev-credit {color: #00ffcc !important; font-size: 0.9rem; font-style: italic; margin-bottom: 2rem;}
    [data-testid="stChatMessage"] {background-color: #151515 !important; border: 1px solid #333 !important;}
    .stInfo {background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #00ffcc !important;}
    </style>
""", unsafe_allow_html=True)

# العنوان والنص الاحترافي
st.title("⚡ AI STARTUP ENGINE") 
st.markdown("<p class='sub-title'>Multi-Agent Business Idea Analysis System</p>", unsafe_allow_html=True)
st.markdown("<p class='dev-credit'>Developed by Eng. Montaser</p>", unsafe_allow_html=True)

if "GROQ_API_KEY" not in st.secrets:
    st.error("مفتاح GROQ_API_KEY مفقود في إعدادات الـ Secrets.")
    st.stop()

# --- 2. هيكل الحالة (State) ---
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

# دالة استدعاء آمنة (تمنع الخطأ 429)
def safe_call(llm, prompt):
    for _ in range(3): # محاولة 3 مرات قبل الاستسلام
        try:
            return llm.invoke([SystemMessage(content=prompt)])
        except Exception as e:
            if "429" in str(e): time.sleep(3) # انتظار في حال الضغط
            else: raise e
    return llm.invoke([SystemMessage(content=prompt)])

# --- 3. بناء المحرك (بدون Checkpointer لتجنب الأخطاء) ---
@st.cache_resource
def get_graph():
    # موديل سريع للمهام، وموديل ذكي للتقرير النهائي
    llm_fast = ChatGroq(model="llama-3.1-8b-instant", api_key=st.secrets["GROQ_API_KEY"])
    llm_deep = ChatGroq(model="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])
    
    def advisor_node(state: State, role: str):
        input_text = state['messages'][-1].content
        # محاكاة ذكية: تعقيد السؤال يحدد وقت التحليل
        time.sleep(min(len(input_text) / 500, 2)) 
        prompt = f"Act as a senior {role} expert. Analyze: {input_text}"
        res = safe_call(llm_fast, prompt)
        return {"advisor_reports": {role: res.content}}

    builder = StateGraph(State)
    builder.add_node("fanout", lambda s: {})
    for role in ["Market", "Legal", "Tech", "Strategy"]:
        builder.add_node(role, lambda s, r=role: advisor_node(s, r))
        builder.add_edge("fanout", role)
        builder.add_edge(role, "report")
        
    builder.add_node("report", lambda s: {"final_report": safe_call(llm_deep, f"Synthesize these deep insights into a professional report: {s['advisor_reports']}").content})
    
    builder.set_entry_point("fanout")
    builder.add_edge("report", END)
    
    # Compile بدون Checkpointer لضمان عدم وجود أخطاء في الـ State
    return builder.compile() 

graph = get_graph()

# --- 4. التفاعل مع المستخدم ---
if prompt := st.chat_input("أدخل فكرتك التجارية (سأحللها بعمق)..."):
    st.chat_message("user").write(prompt)
    
    with st.status("🚀 جاري تهيئة العقول الاصطناعية...", expanded=True) as status:
        st.write("📊 وكيل السوق: دراسة الجدوى والمنافسين...")
        st.write("⚖️ وكيل القانون: مراجعة المخاطر والامتثال...")
        st.write("⚙️ وكيل التقنية: فحص الفجوة التقنية...")
        st.write("🎯 وكيل الاستراتيجية: وضع خطة الانطلاق...")
        
        # تنفيذ المحرك
        result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
        
        status.update(label="✅ التحليل مكتمل - النظام جاهز.", state="complete")
        
    # عرض النتائج
    st.markdown("---")
    st.markdown("### 📑 الترير النهائي")
    st.info(result["final_report"])
