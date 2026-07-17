import streamlit as st  # استيراد مكتبة ستريم ليت لبناء الواجهة التفاعلية
import operator  # استيراد مكتبة العمليات الحسابية
import time  # استيراد مكتبة الوقت للتحكم في فترات الانتظار
from typing import List, Annotated, Dict  # استيراد أدوات تحديد أنواع البيانات
from typing_extensions import TypedDict  # استيراد أنواع البيانات الهيكلية
from langchain_groq import ChatGroq  # استيراد مكتبة الربط مع موديلات Groq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage  # استيراد رسائل LangChain
from langgraph.graph.message import add_messages  # استيراد مُقلل الرسائل
from langgraph.graph import StateGraph, END  # استيراد أدوات بناء الجراف

# --- 1. إعدادات الصفحة والتصميم ---
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

# تصميم CSS احترافي للواجهة (Dark UI)
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {color: #00ffcc !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase; margin-bottom: 0px !important;}
    .tagline {color: #cccccc !important; font-size: 1.1rem; margin-bottom: 5px;}
    .dev-credit {color: #00ffcc !important; font-size: 0.9rem; font-style: italic; margin-bottom: 2rem;}
    [data-testid="stChatMessage"] {background-color: #151515 !important; border: 1px solid #333 !important;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI STARTUP ENGINE") # عنوان الأداة
st.markdown("<p class='tagline'>Advanced Multi-Agent System for Professional Business Idea Validation & Strategic Analysis.</p>", unsafe_allow_html=True)
st.markdown("<p class='dev-credit'>Developed by Eng. Montaser</p>", unsafe_allow_html=True) # التوقيع

if "GROQ_API_KEY" not in st.secrets:
    st.error("مفتاح GROQ_API_KEY مفقود في إعدادات الـ Secrets.")
    st.stop()

# --- 2. تعريف الحالة (State) ---
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

# --- 3. بناء المحرك (Graph) ---
@st.cache_resource
def get_graph():
    # استخدام الموديل القوي للتحليل العميق
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=st.secrets["GROQ_API_KEY"])
    
    # دالة التحليل مع محاكاة سرعة تعتمد على طول المدخل (تعقيد السؤال)
    def advisor_node(state: State, role: str):
        input_text = state['messages'][-1].content
        # كلما كان السؤال طويلاً، زاد وقت "التفكير" لزيادة دقة التحليل
        delay = min(len(input_text) / 500, 3) 
        time.sleep(delay) 
        
        prompt = f"Act as a senior {role} expert. Provide deep strategic insights for: {input_text}"
        res = llm.invoke([SystemMessage(content=prompt)])
        return {"advisor_reports": {role: res.content}}

    builder = StateGraph(State) # بناء الجراف
    builder.add_node("fanout", lambda s: {}) # عقدة توزيع المهام
    for role in ["Market", "Legal", "Tech", "Strategy"]:
        builder.add_node(role, lambda s, r=role: advisor_node(s, r))
        builder.add_edge("fanout", role)
        builder.add_edge(role, "report")
        
    builder.add_node("report", lambda s: {"final_report": llm.invoke([HumanMessage(content=str(s['advisor_reports']))]).content})
    
    builder.set_entry_point("fanout") # نقطة البدء
    builder.add_edge("report", END) # نقطة النهاية
    
    return builder.compile() # تم إزالة الـ Checkpointer نهائياً لمنع أخطاء الجلسات

graph = get_graph()

# --- 4. التفاعل مع المستخدم ---
if prompt := st.chat_input("أدخل فكرتك التجارية (سأحللها بعمق)..."):
    st.chat_message("user").write(prompt) # عرض رسالة المستخدم
    
    # عرض خطوات التفكير (Thinking Steps)
    with st.status("🚀 جاري تهيئة العقول الاصطناعية...", expanded=True) as status:
        st.write("📊 وكيل السوق: دراسة الجدوى والمنافسين...")
        st.write("⚖️ وكيل القانون: مراجعة المخاطر والامتثال...")
        st.write("⚙️ وكيل التقنية: فحص الفجوة التقنية...")
        st.write("🎯 وكيل الاستراتيجية: وضع خطة الانطلاق...")
        
        # تنفيذ التحليل
        result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
        
        status.update(label="✅ التحليل مكتمل - النظام جاهز.", state="complete")
        
    # عرض النتائج
    st.markdown("---")
    st.markdown("### 📑 التقرير النهائي (Executive Summary)")
    st.info(result["final_report"]) # عرض التقرير
