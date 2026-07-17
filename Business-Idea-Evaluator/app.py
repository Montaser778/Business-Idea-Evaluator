import streamlit as st
import streamlit.components.v1 as components
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

# 2. حقن الجافا سكريبت للتفاعل (تأثير الظهور التدريجي)
js_interaction = """
<script>
    const observer = new MutationObserver(() => {
        const elements = window.parent.document.querySelectorAll('div[data-testid="stChatMessage"]');
        elements.forEach(el => {
            el.style.opacity = '1';
            el.style.transition = 'opacity 0.8s ease-in-out';
        });
    });
    observer.observe(window.parent.document.body, { childList: true, subtree: true });
</script>
"""
components.html(js_interaction, height=0)

# 3. عرض العنوان
st.title("🚀 AI Startup Idea Advisor")
st.markdown("نظام وكلاء ذكاء اصطناعي متطور لتحليل الأفكار التجارية.")

# 4. جلب المفتاح بأمان
if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ يرجى إضافة GROQ_API_KEY في إعدادات Secrets.")
    st.stop()

# 5. هيكلية LangGraph
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

@st.cache_resource
def build_graph():
    # استخدام النموذج الجديد لحل إيرور Decommissioned
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    
    def decide_node(state: State):
        sys_msg = SystemMessage(content="Decide if you have info. If not, ask ONE question. If yes, say: DONE")
        ai_reply = llm.invoke([sys_msg] + state["messages"])
        return {"messages": [ai_reply]} 

    def advisor_node(state: State, role: str, prompt: str):
        report = llm.invoke([SystemMessage(content=prompt + str(state['messages']))])
        return {"advisor_reports": {role: report.content}}

    builder = StateGraph(State)
    builder.add_node("decide_node", decide_node)
    builder.add_node("human_node", lambda s: {})
    builder.add_node("fanout", lambda s: {}) 
    builder.add_node("market", lambda s: advisor_node(s, "Market", "Analyze market potential:"))
    builder.add_node("legal", lambda s: advisor_node(s, "Legal", "Analyze legal issues:"))
    builder.add_node("tech", lambda s: advisor_node(s, "Tech", "Analyze technical feasibility:"))
    builder.add_node("strategy", lambda s: advisor_node(s, "Strategy", "Define launch strategy:"))
    builder.add_node("report", lambda s: {"final_report": llm.invoke(f"Summarize: {s['advisor_reports']}").content})

    builder.set_entry_point("decide_node")
    builder.add_conditional_edges("decide_node", lambda s: "fanout" if s["messages"][-1].content.upper().startswith("DONE") else "human_node", {"human_node": "human_node", "fanout": "fanout"})
    builder.add_edge("human_node", "decide_node")
    for node in ["market", "legal", "tech", "strategy"]:
        builder.add_edge("fanout", node)
        builder.add_edge(node, "report")
    builder.add_edge("report", END)

    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["human_node"])

# 6. التشغيل
graph = build_graph()
config = {"configurable": {"thread_id": "startup_thread_1"}}
user_input = st.chat_input("أدخل فكرتك التجارية هنا...")

if user_input:
    state = graph.get_state(config)
    if state.next and "human_node" in state.next:
        graph.update_state(config, {"messages": [HumanMessage(content=user_input)]}, as_node="human_node")
        graph.invoke(None, config)
    else:
        graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)

# عرض المحادثة
for msg in graph.get_state(config).values.get("messages", []):
    if isinstance(msg, HumanMessage): st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and not msg.content.upper().startswith("DONE"): st.chat_message("assistant").write(msg.content)

final = graph.get_state(config).values.get("final_report")
if final: st.markdown(f"### 📑 التقرير النهائي\n{final}")
