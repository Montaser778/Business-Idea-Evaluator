import streamlit as st  # استيراد مكتبة ستريم ليت للواجهة
import operator  # استيراد مكتبة العمليات المنطقية
from typing import List, Annotated, Dict  # استيراد أنواع البيانات
from typing_extensions import TypedDict  # استيراد الأنواع الهيكلية
from langchain_groq import ChatGroq  # استيراد مكتبة جروك
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage  # استيراد أنواع الرسائل
from langgraph.graph.message import add_messages  # استيراد مُقلل الرسائل
from langgraph.graph import StateGraph, END  # استيراد هيكل الجراف

# إعداد الصفحة وتحديد التخطيط لتكون واجهة احترافية
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

# تصميم CSS لـ Dark Mode احترافي
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {color: #00ffcc !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase;}
    [data-testid="stChatMessage"] {background-color: #151515 !important; border: 1px solid #333 !important; border-radius: 10px;}
    .stInfo {background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #00ffcc !important;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI STARTUP ENGINE") # عنوان الأداة
st.markdown("Developed by Eng. Montaser", unsafe_allow_html=True) # التوقيع

# التحقق من وجود مفتاح الـ API
if "GROQ_API_KEY" not in st.secrets:
    st.error("مفتاح GROQ_API_KEY مفقود!")
    st.stop()

# تعريف هيكل الحالة (State)
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

# بناء المحرك (Graph) بدون checkpointer لتجنب الأخطاء
@st.cache_resource
def get_graph():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=st.secrets["GROQ_API_KEY"])
    
    # عقدة التحليل
    def advisor_node(state: State, role: str):
        prompt = f"Act as {role}. Deeply analyze: {state['messages'][-1].content}"
        res = llm.invoke([SystemMessage(content=prompt)])
        return {"advisor_reports": {role: res.content}}

    builder = StateGraph(State)
    builder.add_node("fanout", lambda s: {})
    for role in ["Market", "Legal", "Tech", "Strategy"]:
        builder.add_node(role, lambda s, r=role: advisor_node(s, r))
        builder.add_edge("fanout", role)
        builder.add_edge(role, "report")
    builder.add_node("report", lambda s: {"final_report": llm.invoke([HumanMessage(content=str(s['advisor_reports']))]).content})
    
    builder.set_entry_point("fanout")
    builder.add_edge("report", END)
    return builder.compile() # تم إزالة الـ checkpointer نهائياً

graph = get_graph()

# التفاعل مع المستخدم
if prompt := st.chat_input("أدخل فكرتك التجارية - المحرك جاهز..."):
    st.chat_message("user").write(prompt) # عرض رسالة المستخدم
    with st.status("🚀 جاري معالجة الفكرة بعمق...", expanded=True) as status:
        # تنفيذ الجراف بدون الاعتماد على ذاكرة الحالة
        result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
        st.chat_message("assistant").write(result["final_report"]) # عرض التقرير النهائي
        status.update(label="✅ تم التحليل بنجاح.", state="complete")
