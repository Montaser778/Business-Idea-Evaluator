import streamlit as st
import operator
import sqlite3
from typing import List, Annotated, Dict
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. إعدادات الصفحة
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

# 2. تصميم احترافي
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {font-family: 'Arial Black', sans-serif; color: #00ffcc !important; text-transform: uppercase; margin-bottom: 0px !important;}
    .sub-title {color: #888 !important; font-size: 1.2rem; margin-bottom: 0.5rem; font-style: italic;}
    .dev-by {color: #555 !important; font-size: 0.9rem; margin-bottom: 2rem;}
    [data-testid="stChatMessage"] {background-color: #1a1a1a !important; border: 1px solid #333 !important;}
    [data-testid="stChatMessageContent"] {color: #ffffff !important;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI STARTUP ENGINE")
st.markdown("<p class='sub-title'>Multi-Agent Business Idea Analysis System</p>", unsafe_allow_html=True)
st.markdown("<p class='dev-by'>Developed by Eng. Montaser</p>", unsafe_allow_html=True)

# 3. التحقق من المفتاح
if "GROQ_API_KEY" not in st.secrets:
    st.error("مفتاح GROQ_API_KEY مفقود.")
    st.stop()

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

# 4. بناء الـ Graph
@st.cache_resource
def get_graph():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn) 
    
    def decide_node(state: State):
        msg = llm.invoke([SystemMessage(content="Decide if DONE or ask follow-up.")] + state["messages"])
        return {"messages": [msg]}

    def advisor_node(state: State, role: str):
        prompt = f"Analyze {role} for this idea: {state['messages'][-1].content}"
        res = llm.invoke([SystemMessage(content=prompt)])
        return {"advisor_reports": {role: res.content}}

    builder = StateGraph(State)
    builder.add_node("decide", decide_node)
    builder.add_node("human", lambda s: {})
    builder.add_node("fanout", lambda s: {})
    
    for role in ["Market", "Legal", "Tech", "Strategy"]:
        builder.add_node(role, lambda s, r=role: advisor_node(s, r))
        builder.add_edge("fanout", role)
        builder.add_edge(role, "report")
    
    builder.add_node("report", lambda s: {"final_report": llm.invoke([HumanMessage(content=str(s['advisor_reports']))]).content})
    builder.set_entry_point("decide")
    builder.add_conditional_edges("decide", lambda s: "fanout" if "DONE" in s["messages"][-1].content.upper() else "human", {"human": "human", "fanout": "fanout"})
    builder.add_edge("human", "decide")
    builder.add_edge("report", END)
    
    return builder.compile(checkpointer=memory, interrupt_before=["human"])

graph = get_graph()
config = {"configurable": {"thread_id": "montaser_session"}}

# 5. التفاعل
if prompt := st.chat_input("أدخل فكرتك التجارية..."):
    with st.status("🚀 جاري التحليل...", expanded=True) as status:
        state = graph.get_state(config)
        if state.next and "human" in state.next:
            graph.update_state(config, {"messages": [HumanMessage(content=prompt)]}, as_node="human")
        else:
            graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        graph.invoke(None, config)
        status.update(label="✅ تم التحليل.", state="complete")
    st.rerun()

# 6. العرض (تأكد من عدم وجود كود خارج النطاق)
for msg in graph.get_state(config).values.get("messages", []):
    if msg.content and "DONE" not in msg.content:
        st.chat_message("assistant" if isinstance(msg, AIMessage) else "user").write(msg.content)

if final := graph.get_state(config).values.get("final_report"):
    st.markdown("### 📑 التقرير النهائي")
    st.info(final)
