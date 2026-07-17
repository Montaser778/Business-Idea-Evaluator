import streamlit as st
import operator
from typing import List, Annotated, Dict
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. إعدادات الصفحة والتصميم القوي
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {color: #00ffcc !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase; margin-bottom: 0px !important;}
    .sub-title {color: #ffffff !important; font-size: 1.2rem; margin-top: 5px;}
    .dev-by {color: #555 !important; font-size: 0.9rem; margin-bottom: 2rem;}
    [data-testid="stChatMessage"] {background-color: #151515 !important; border: 1px solid #333 !important; border-radius: 10px;}
    .stStatus {background: #1a1a1a !important; color: #00ffcc !important;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI STARTUP ENGINE")
st.markdown("<p class='sub-title'>Multi-Agent Business Idea Analysis System</p>", unsafe_allow_html=True)
st.markdown("<p class='dev-by'>Developed by Eng. Montaser</p>", unsafe_allow_html=True)

# 2. إعداد الحالة (State)
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    market_report: str
    legal_report: str
    tech_report: str
    strategy_report: str
    final_report: str

# 3. بناء الـ Graph الموسع (إعادة هيكلة الـ 172 سطر)
@st.cache_resource
def get_graph():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=st.secrets["GROQ_API_KEY"])
    
    # تعريف الوكلاء كدوال مستقلة لضمان العمق
    def market_node(state: State):
        res = llm.invoke([SystemMessage(content="Analyze Market potential, competition, and target audience in detail.")] + state['messages'])
        return {"market_report": res.content}

    def legal_node(state: State):
        res = llm.invoke([SystemMessage(content="Analyze legal requirements, risks, and compliance for this business idea.")] + state['messages'])
        return {"legal_report": res.content}

    def tech_node(state: State):
        res = llm.invoke([SystemMessage(content="Analyze technical feasibility, architecture, and required stack.")] + state['messages'])
        return {"tech_report": res.content}

    def strategy_node(state: State):
        res = llm.invoke([SystemMessage(content="Create a go-to-market strategy, growth plan, and timeline.")] + state['messages'])
        return {"strategy_report": res.content}

    def final_report_node(state: State):
        summary = f"Market: {state['market_report']}\nLegal: {state['legal_report']}\nTech: {state['tech_report']}\nStrategy: {state['strategy_report']}"
        res = llm.invoke([SystemMessage(content=f"Synthesize this deep analysis into a final professional startup report: {summary}")] + state['messages'])
        return {"final_report": res.content}

    builder = StateGraph(State)
    builder.add_node("market", market_node)
    builder.add_node("legal", legal_node)
    builder.add_node("tech", tech_node)
    builder.add_node("strategy", strategy_node)
    builder.add_node("report", final_report_node)
    
    # الربط (Parallel Execution)
    builder.set_entry_point("market")
    builder.add_edge("market", "legal")
    builder.add_edge("legal", "tech")
    builder.add_edge("tech", "strategy")
    builder.add_edge("strategy", "report")
    builder.add_edge("report", END)
    
    return builder.compile(checkpointer=MemorySaver())

graph = get_graph()
config = {"configurable": {"thread_id": "montaser_session"}}

# 4. التفاعل (نظام Loading احترافي)
if prompt := st.chat_input("أدخل فكرتك التجارية - المحرك جاهز..."):
    st.chat_message("user").write(prompt)
    with st.status("🚀 جاري التحليل العميق...", expanded=True) as status:
        st.write("📈 تحليل السوق...")
        st.write("⚖️ المراجعة القانونية...")
        st.write("⚙️ الفحص التقني...")
        st.write("🎯 صياغة الاستراتيجية...")
        graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        status.update(label="✅ تم إنجاز التحليل الشامل.", state="complete")

# 5. عرض النتائج النهائية
final = graph.get_state(config).values.get("final_report")
if final:
    st.markdown("---")
    st.markdown("### 📑 التقرير النهائي (Deep Analysis)")
    st.info(final)
