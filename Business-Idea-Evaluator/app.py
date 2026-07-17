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

# 2. تصميم CSS احترافي (الوضع المظلم مع وضوح النصوص)
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {font-family: 'Arial Black', sans-serif; color: #00ffcc !important; text-transform: uppercase; text-shadow: 0px 0px 10px rgba(0, 255, 204, 0.5);}
    [data-testid="stChatMessage"] {background-color: #1a1a1a !important; border: 1px solid #333 !important;}
    [data-testid="stChatMessageContent"] {color: #ffffff !important; font-size: 1.1rem !important;}
    p, div {color: #ffffff !important;}
    .stInfo {background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #333 !important;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI STARTUP ENGINE")

# 3. التحقق من المفتاح
if "GROQ_API_KEY" not in st.secrets:
    st.error("مفتاح GROQ_API_KEY مفقود في إعدادات الـ Secrets.")
    st.stop()

# 4. تعريف الحالة
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

# 5. بناء الـ Graph
@st.cache_resource
def get_graph():
    # استخدام الموديل المحدث والمستقر
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    
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
    
    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["human"])

graph = get_graph()
config = {"configurable": {"thread_id": "montaser_session"}}

# 6. التفاعل مع المستخدم
if prompt := st.chat_input("أدخل فكرتك التجارية - المحرك جاهز..."):
    st.chat_message("user", avatar="👤").write(prompt)
    
    with st.status("🚀 جاري تهيئة المحركات...", expanded=True) as status:
        st.write("🔍 فحص الفكرة...")
        state = graph.get_state(config)
        if state.next and "human" in state.next:
            graph.update_state(config, {"messages": [HumanMessage(content=prompt)]}, as_node="human")
        else:
            graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        
        st.write("🧠 الوكلاء في مرحلة المعالجة العميقة...")
        graph.invoke(None, config)
        
        status.update(label="✅ تم إنجاز التحليل بنجاح.", state="complete")

# 7. عرض النتائج
for msg in graph.get_state(config).values.get("messages", []):
    if isinstance(msg, (HumanMessage, AIMessage)) and "DONE" not in msg.content:
        st.chat_message("assistant" if isinstance(msg, AIMessage) else "user").write(msg.content)

if final := graph.get_state(config).values.get("final_report"):
    st.markdown("---")
    st.markdown("### 📑 التقرير النهائي (Engine Output)")
    st.info(final)
