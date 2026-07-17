import streamlit as st
import operator
from typing import List, Annotated, Dict
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. إعدادات الصفحة
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

# 2. تصميم احترافي (Dark UI)
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

st.title("⚡ AI STARTUP ENGINE")
st.markdown("<p class='sub-title'>Multi-Agent Business Idea Analysis System</p>", unsafe_allow_html=True)
st.markdown("<p class='dev-by'>Developed by Eng. Montaser</p>", unsafe_allow_html=True)

# 3. التحقق من المفتاح
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY missing in secrets.")
    st.stop()

# 4. تعريف الحالة (State)
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

# 5. بناء المحرك (Graph)
@st.cache_resource
def get_graph():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    
    # تعريف الوكلاء كما في الكود الخاص بك
    def decide_node(state: State):
        base_system_msg = SystemMessage(content="You are a helpful tool. Decide if you have enough info about the startup idea. If not, ask ONE precise question. If yes, say: DONE")
        ai_reply = llm.invoke([base_system_msg] + state["messages"])
        return {"messages": [ai_reply]}

    def advisor_node(state: State, role: str):
        prompt = f"You are a senior {role}. Evaluate: {state['messages'][-1].content}"
        report = llm.invoke([SystemMessage(content=prompt)])
        return {"advisor_reports": {role: report.content}}

    builder = StateGraph(State)
    builder.add_node("decide_node", decide_node)
    builder.add_node("fanout", lambda s: {})
    for role in ["Market Analyst", "Legal Advisor", "Technical Expert", "Strategist"]:
        builder.add_node(role, lambda s, r=role: advisor_node(s, r))
        builder.add_edge("fanout", role)
        builder.add_edge(role, "collect_and_report")
    
    builder.add_node("collect_and_report", lambda s: {"final_report": llm.invoke([SystemMessage(content=f"Summarize these reports: {s['advisor_reports']}")]).content})
    
    builder.set_entry_point("decide_node")
    builder.add_conditional_edges("decide_node", lambda s: "fanout" if "DONE" in s["messages"][-1].content.upper() else "decide_node", {"fanout": "fanout", "decide_node": "decide_node"})
    builder.add_edge("collect_and_report", END)
    
    return builder.compile(checkpointer=MemorySaver())

graph = get_graph()
config = {"configurable": {"thread_id": "montaser_session"}}

# 6. التفاعل مع المستخدم
if prompt := st.chat_input("أدخل فكرتك التجارية هنا..."):
    st.chat_message("user").write(prompt)
    with st.status("🚀 جاري معالجة الفكرة...", expanded=True) as status:
        graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        status.update(label="✅ تم التحليل بنجاح.", state="complete")

# 7. عرض الرسائل
state = graph.get_state(config).values
for msg in state.get("messages", []):
    if msg.content and "DONE" not in msg.content and not isinstance(msg, SystemMessage):
        st.chat_message("assistant" if isinstance(msg, AIMessage) else "user").write(msg.content)

if final := state.get("final_report"):
    st.markdown("---")
    st.markdown("### 📑 التقرير النهائي")
    st.info(final)
