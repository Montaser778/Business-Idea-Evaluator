import streamlit as st
import operator
from typing import List, Annotated, Dict
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- الإعدادات والتصميم (Dark UI) ---
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {color: #00ffcc !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase; margin-bottom: 0px !important;}
    .sub-title {color: #ffffff !important; font-size: 1.2rem; margin-top: 5px; font-weight: bold;}
    .dev-by {color: #555 !important; font-size: 0.9rem; margin-bottom: 2rem;}
    [data-testid="stChatMessage"] {background-color: #151515 !important; border: 1px solid #333 !important; border-radius: 10px; margin-bottom: 15px;}
    .stInfo {background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #00ffcc !important;}
    </style>
""", unsafe_allow_html=True)

# --- الواجهة ---
st.title("⚡ AI STARTUP ENGINE")
st.markdown("<p class='sub-title'>Multi-Agent Business Idea Analysis System</p>", unsafe_allow_html=True)
st.markdown("<p class='dev-by'>Developed by Eng. Montaser</p>", unsafe_allow_html=True)

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY missing in secrets.")
    st.stop()

# --- تعريف الحالة ---
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    market_report: str
    legal_report: str
    tech_report: str
    strategy_report: str
    final_report: str

# --- بناء المحرك (Graph الموسع) ---
@st.cache_resource
def get_graph():
    # استخدام موديل متطور لردود عميقة
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=st.secrets["GROQ_API_KEY"])
    
    def market_node(state: State):
        prompt = "Acting as a Market Expert, analyze the potential, competition, and target audience for: " + state['messages'][-1].content
        res = llm.invoke([SystemMessage(content=prompt)])
        return {"market_report": res.content}

    def legal_node(state: State):
        prompt = "Acting as a Legal Consultant, analyze risks, compliance, and regulatory needs for: " + state['messages'][-1].content
        res = llm.invoke([SystemMessage(content=prompt)])
        return {"legal_report": res.content}

    def tech_node(state: State):
        prompt = "Acting as a CTO, analyze technical feasibility, architecture, and tech stack for: " + state['messages'][-1].content
        res = llm.invoke([SystemMessage(content=prompt)])
        return {"tech_report": res.content}

    def strategy_node(state: State):
        prompt = "Acting as a Business Strategist, create a GTM plan, growth tactics, and milestones for: " + state['messages'][-1].content
        res = llm.invoke([SystemMessage(content=prompt)])
        return {"strategy_report": res.content}

    def final_report_node(state: State):
        full_context = f"Market: {state['market_report']}\nLegal: {state['legal_report']}\nTech: {state['tech_report']}\nStrategy: {state['strategy_report']}"
        res = llm.invoke([SystemMessage(content="Synthesize these insights into a world-class professional startup analysis: " + full_context)])
        return {"final_report": res.content}

    builder = StateGraph(State)
    builder.add_node("market", market_node)
    builder.add_node("legal", legal_node)
    builder.add_node("tech", tech_node)
    builder.add_node("strategy", strategy_node)
    builder.add_node("report", final_report_node)
    
    # تنفيذ متسلسل لضمان التغطية الكاملة
    builder.set_entry_point("market")
    builder.add_edge("market", "legal")
    builder.add_edge("legal", "tech")
    builder.add_edge("tech", "strategy")
    builder.add_edge("strategy", "report")
    builder.add_edge("report", END)
    
    return builder.compile(checkpointer=MemorySaver())

graph = get_graph()
config = {"configurable": {"thread_id": "montaser_session"}}

# --- التفاعل ---
if prompt := st.chat_input("أدخل فكرتك التجارية (سأقوم بتحليلها بعمق)..."):
    st.chat_message("user").write(prompt)
    with st.status("🚀 جاري المعالجة (محرك الذكاء الاصطناعي)...", expanded=True) as status:
        st.write("📊 تحليل السوق والمنافسين...")
        st.write("⚖️ المراجعة القانونية والمخاطر...")
        st.write("💻 البنية التقنية والتنفيذ...")
        st.write("🎯 الاستراتيجية وخطة النمو...")
        graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        status.update(label="✅ تم التحليل بنجاح.", state="complete")

# --- عرض النتائج ---
state = graph.get_state(config).values
if final := state.get("final_report"):
    st.markdown("---")
    st.markdown("### 📑 التقرير النهائي (Deep Executive Analysis)")
    st.info(final)

# --- إغلاق الحلقة ---
# (ملاحظة: هذا الكود الهيكلي يضمن أداءً مستقراً وتوسعاً مستقبلياً)
