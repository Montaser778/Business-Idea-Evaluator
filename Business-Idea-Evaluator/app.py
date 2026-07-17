import streamlit as st
import operator
from typing import List, Annotated, Dict
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

# تصميم Dark UI
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {color: #00ffcc !important; text-transform: uppercase;}
    [data-testid="stChatMessage"] {background-color: #1a1a1a !important; border: 1px solid #333 !important;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI STARTUP ENGINE")
st.markdown("<p style='color:#888'>Multi-Agent Business Idea Analysis System | Dev by Eng. Montaser</p>", unsafe_allow_html=True)

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY missing.")
    st.stop()

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

@st.cache_resource
def get_graph():
    # زيادة العمق بالتحليل عبر تقليل القيود وزيادة التفاصيل
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=st.secrets["GROQ_API_KEY"])
    
    def advisor_node(state: State, role: str):
        # جعل البرومبت أكثر عمقاً وتفصيلاً
        prompt = f"Act as a senior {role} expert. Provide a deep, critical, and comprehensive analysis for: {state['messages'][-1].content}. Use bullet points and strategic insights."
        res = llm.invoke([SystemMessage(content=prompt)])
        return {"advisor_reports": {role: res.content}}

    builder = StateGraph(State)
    builder.add_node("decide", lambda s: {"messages": [llm.invoke([SystemMessage(content="Are you done?")] + s["messages"])]})
    builder.add_node("fanout", lambda s: {})
    for role in ["Market", "Legal", "Tech", "Strategy"]:
        builder.add_node(role, lambda s, r=role: advisor_node(s, r))
        builder.add_edge("fanout", role)
        builder.add_edge(role, "report")
    builder.add_node("report", lambda s: {"final_report": llm.invoke([HumanMessage(content=f"Synthesize this deep analysis: {s['advisor_reports']}")])})
    
    builder.set_entry_point("decide")
    builder.add_conditional_edges("decide", lambda s: "fanout" if "DONE" in s["messages"][-1].content.upper() else "decide", {"fanout": "fanout", "decide": "decide"})
    builder.add_edge("report", END)
    return builder.compile(checkpointer=MemorySaver())

graph = get_graph()
config = {"configurable": {"thread_id": "montaser_session"}}

# العرض التفاعلي
if prompt := st.chat_input("أدخل فكرتك التجارية..."):
    st.chat_message("user").write(prompt)
    
    with st.status("🚀 جاري معالجة الفكرة بعمق...", expanded=True) as status:
        st.write("🔍 فحص وتحليل الفكرة...")
        graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        st.write("🧠 تنفيذ التحليلات الاستراتيجية (Market, Legal, Tech)...")
        status.update(label="✅ تم التحليل بنجاح.", state="complete")

# عرض النتائج بدون ريرن
for msg in graph.get_state(config).values.get("messages", []):
    if msg.content and "DONE" not in msg.content:
        st.chat_message("assistant").write(msg.content)

if final := graph.get_state(config).values.get("final_report"):
    st.info(final.content)
