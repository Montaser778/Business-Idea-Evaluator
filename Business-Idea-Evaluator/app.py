import streamlit as st
import operator
from typing import List, Annotated, Dict
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# تنسيق احترافي ونظيف
st.set_page_config(page_title="AI Startup Advisor", layout="centered")

st.markdown("""
    <style>
    .main {background-color: #f5f7f9;}
    .stChatFloatingInputContainer {background-color: #ffffff; border-radius: 10px; padding: 10px;}
    h1 {color: #1f2937; text-align: center; margin-bottom: 30px;}
    </style>
""", unsafe_allow_html=True)

st.title("🚀 AI Startup Idea Advisor")

# حل الإيرور: استخدام الموديل المحدث
if "GROQ_API_KEY" not in st.secrets:
    st.error("مفتاح API مفقود في إعدادات الـ Secrets.")
    st.stop()

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

@st.cache_resource
def get_graph():
    # الموديل المعتمد حالياً من Groq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    
    def decide_node(state: State):
        msg = llm.invoke([SystemMessage(content="Decide: DONE or ask follow-up.")] + state["messages"])
        return {"messages": [msg]}

    builder = StateGraph(State)
    builder.add_node("decide", decide_node)
    builder.add_node("human", lambda s: {})
    builder.add_node("fanout", lambda s: {})
    # إضافة الوكلاء بشكل مبسط
    for role in ["Market", "Legal", "Tech", "Strategy"]:
        builder.add_node(role, lambda s, r=role: {"advisor_reports": {r: llm.invoke([SystemMessage(content=f"Analyze {r} for idea")]+s['messages']).content}})
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

# التفاعل
if prompt := st.chat_input("أدخل فكرتك التجارية..."):
    state = graph.get_state(config)
    if state.next and "human" in state.next:
        graph.update_state(config, {"messages": [HumanMessage(content=prompt)]}, as_node="human")
        graph.invoke(None, config)
    else:
        graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)

# العرض
for msg in graph.get_state(config).values.get("messages", []):
    if isinstance(msg, (HumanMessage, AIMessage)) and "DONE" not in msg.content:
        st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

if final := graph.get_state(config).values.get("final_report"):
    with st.expander("📑 التقرير النهائي (اضغط للعرض)"):
        st.write(final)
