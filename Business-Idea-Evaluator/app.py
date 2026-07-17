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
st.set_page_config(page_title="AI Startup Advisor", page_icon="🚀", layout="centered")

# 2. كود الخلفية المتحركة (CSS)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
h1, h2, h3, p {color: white !important; text-shadow: 1px 1px 2px black;}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# 3. عرض العنوان الاحترافي
st.markdown("<h1 style='text-align: center;'>🚀 AI Startup Idea Advisor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>نظام وكلاء ذكاء اصطناعي متطور لتحليل الأفكار التجارية</p>", unsafe_allow_html=True)

# 4. جلب المفتاح بأمان من Streamlit Secrets
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("⚠️ يرجى إضافة GROQ_API_KEY في إعدادات Secrets الخاصة بالتطبيق.")
    st.stop()

# 5. هيكلية LangGraph
class State(TypedDict):
    idea: str
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

@st.cache_resource
def build_graph(api_key: str):
    llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=api_key)
    
    base_system_msg = SystemMessage(
        content="""You are a helpful tool. Decide whether you have enough info about the idea. 
        If not, ask ONE precise follow-up question. If yes, say: DONE"""
    )

    def decide_node(state: State):
        ai_reply = llm.invoke([base_system_msg] + state["messages"])
        return {"messages": [ai_reply]} 

    def route(state: State):
        return "fanout" if state["messages"][-1].content.strip().upper().startswith("DONE") else "human_node"

    # تعريف الوكلاء (التحليل، القانوني، التقني، الاستراتيجي)
    def advisor_node(state: State, role: str, prompt: str):
        report = llm.invoke([SystemMessage(content=prompt + str(state['messages']))])
        return {"advisor_reports": {role: report.content}}

    builder = StateGraph(State)
    builder.add_node("decide_node", decide_node)
    builder.add_node("human_node", lambda s: {})
    builder.add_node("fanout", lambda s: {}) 
    builder.add_node("market_analyst", lambda s: advisor_node(s, "Market", "Analyze market potential for:"))
    builder.add_node("legal", lambda s: advisor_node(s, "Legal", "Analyze legal issues for:"))
    builder.add_node("technical", lambda s: advisor_node(s, "Tech", "Analyze technical feasibility for:"))
    builder.add_node("strategist", lambda s: advisor_node(s, "Strategy", "Define launch strategy for:"))
    builder.add_node("report", lambda s: {"final_report": llm.invoke(f"Summarize these: {s['advisor_reports']}").content})

    builder.set_entry_point("decide_node")
    builder.add_conditional_edges("decide_node", route, {"human_node": "human_node", "fanout": "fanout"})
    builder.add_edge("human_node", "decide_node")
    for node in ["market_analyst", "legal", "technical", "strategist"]:
        builder.add_edge("fanout", node)
        builder.add_edge(node, "report")
    builder.add_edge("report", END)

    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["human_node"])

# 6. التشغيل
graph = build_graph(api_key)
config = {"configurable": {"thread_id": "startup_thread_1"}}
user_input = st.chat_input("أدخل فكرتك التجارية هنا...")

if user_input:
    state = graph.get_state(config)
    if state.next and "human_node" in state.next:
        graph.update_state(config, {"messages": [HumanMessage(content=user_input)]}, as_node="human_node")
        graph.invoke(None, config)
    else:
        graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)

# عرض النتائج
for msg in graph.get_state(config).values.get("messages", []):
    if isinstance(msg, HumanMessage): st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and not msg.content.upper().startswith("DONE"): st.chat_message("assistant").write(msg.content)

final = graph.get_state(config).values.get("final_report")
if final: st.markdown(f"### 📑 التقرير النهائي\n{final}")
