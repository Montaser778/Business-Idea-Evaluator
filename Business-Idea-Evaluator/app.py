import streamlit as st  # استيراد مكتبة ستريم ليت لبناء الواجهة التفاعلية
import operator  # استيراد مكتبة العمليات الحسابية والمنطقية
import time  # استيراد مكتبة الوقت لإدارة فترات الانتظار عند حدوث خطأ
from typing import List, Annotated, Dict  # استيراد أدوات تحديد أنواع البيانات
from typing_extensions import TypedDict  # استيراد أنواع البيانات الهيكلية
from langchain_groq import ChatGroq  # استيراد مكتبة الربط مع موديلات Groq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage  # استيراد أنواع الرسائل
from langgraph.graph.message import add_messages  # استيراد المُقلل الخاص بالرسائل
from langgraph.graph import StateGraph, END  # استيراد بناء الجراف والإنهاء
from langgraph.checkpoint.memory import MemorySaver  # استيراد ذاكرة حفظ الحالة

# إعداد الصفحة وتحديد التخطيط لتكون واجهة احترافية واسعة
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

# تنسيق CSS احترافي للوضع الداكن (Dark Mode)
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {color: #00ffcc !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase;}
    [data-testid="stChatMessage"] {background-color: #1a1a1a !important; border: 1px solid #333 !important;}
    .stStatus {background: #111 !important; color: #00ffcc !important;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI STARTUP ENGINE") # عنوان الأداة الرئيسي

# التحقق من وجود مفتاح الـ API في الإعدادات
if "GROQ_API_KEY" not in st.secrets:
    st.error("مفتاح GROQ_API_KEY مفقود في إعدادات الـ Secrets.")
    st.stop()

# تعريف هيكل الحالة (State) الذي يمر عبر الجراف
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

# دالة ذكية لإعادة المحاولة عند حدوث خطأ 429
def safe_invoke(llm, messages):
    try:
        return llm.invoke(messages) # محاولة تنفيذ الطلب
    except Exception as e:
        if "429" in str(e): # في حال تجاوز حد الطلبات
            st.warning("ضغط عالٍ في الطلبات، جاري الانتظار 5 ثوانٍ...")
            time.sleep(5) # انتظار بسيط قبل الإعادة
            return llm.invoke(messages)
        raise e

# بناء الجراف مع تحسين الموديلات (نموذج سريع للمهام البسيطة، ونموذج قوي للتحليل)
@st.cache_resource
def get_graph():
    llm_fast = ChatGroq(model="llama-3.1-8b-instant", api_key=st.secrets["GROQ_API_KEY"])
    llm_deep = ChatGroq(model="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])
    
    # عقدة القرار (سريعة جداً)
    def decide_node(state: State):
        msg = safe_invoke(llm_fast, [SystemMessage(content="Decide: DONE or ask.")] + state["messages"])
        return {"messages": [msg]}

    # عقدة التحليل (عميقة ومفصلة)
    def advisor_node(state: State, role: str):
        prompt = f"Act as {role}. Analyze in depth: {state['messages'][-1].content}"
        res = safe_invoke(llm_deep, [SystemMessage(content=prompt)])
        return {"advisor_reports": {role: res.content}}

    builder = StateGraph(State) # بناء الهيكل
    builder.add_node("decide", decide_node) # عقدة القرار
    builder.add_node("fanout", lambda s: {}) # عقدة التوزيع
    
    for role in ["Market", "Legal", "Tech", "Strategy"]:
        builder.add_node(role, lambda s, r=role: advisor_node(s, r))
        builder.add_edge("fanout", role)
        builder.add_edge(role, "report")
    
    builder.add_node("report", lambda s: {"final_report": safe_invoke(llm_deep, [HumanMessage(content=str(s['advisor_reports']))]).content})
    
    builder.set_entry_point("decide")
    builder.add_conditional_edges("decide", lambda s: "fanout" if "DONE" in s["messages"][-1].content.upper() else "decide", {"fanout": "fanout", "decide": "decide"})
    builder.add_edge("report", END)
    return builder.compile() # بدون memory للبدء دائماً من الصفر

graph = get_graph() 
config = {"configurable": {"thread_id": time.time()}} # توليد ID جديد في كل مرة لبدء جلسة جديدة

# التفاعل مع المستخدم
if prompt := st.chat_input("أدخل فكرتك التجارية - المحرك جاهز..."):
    st.chat_message("user").write(prompt) # عرض المدخل
    with st.status("🚀 جاري معالجة الفكرة بعمق...", expanded=True) as status:
        graph.invoke({"messages": [HumanMessage(content=prompt)]}, config) # تشغيل الجراف
        status.update(label="✅ تم التحليل بنجاح.", state="complete") # التحديث عند الانتهاء

# عرض النتائج النهائية
state = graph.get_state(config).values
if final := state.get("final_report"):
    st.markdown("---")
    st.markdown("### 📑 التقرير النهائي")
    st.info(final)
