import streamlit as st
import os, operator
from typing import List, Annotated, Dict
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. إعدادات واجهة Streamlit
# ==========================================
st.set_page_config(page_title="AI Startup Advisor", page_icon="🚀", layout="wide")
st.title("🚀 AI Startup Idea Advisor")
st.markdown("نظام وكلاء متعدد (Multi-Agent System) لتحليل الأفكار التجارية باستخدام **LangGraph** و **Groq**.")

# إعداد القائمة الجانبية لإدخال مفتاح API
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Groq API Key", type="password")
    st.markdown("---")
    st.markdown("Developed by **Eng. Montaser**")

# ==========================================
# 2. بناء هياكل LangGraph والوكلاء
# ==========================================
class State(TypedDict):
    idea: str
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str

@st.cache_resource
def build_graph(api_key: str):
    """بناء الـ StateGraph وتجميعه مع خاصية التوقف لانتظار المستخدم"""
    llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=api_key)
    
    base_system_msg = SystemMessage(
        content="""You are a helpful tool. 
        Your job: decide whether you have enough information about the start-up idea.
        If not, ask ONE precise follow-up question.
        If yes, say: DONE"""
    )

    def decide_node(state: State):
        conversation = [base_system_msg] + state["messages"]
        ai_reply = llm.invoke(conversation)
        return {"messages": [ai_reply]} 

    def route(state: State):
        last_ai_msg = state["messages"][-1]
        if last_ai_msg.content.strip().upper().startswith("DONE"):
            return "fanout"
        return "human_node"

    def human_node(state: State):
        # عقدة وهمية (Placeholder)
        # Streamlit سيقوم بتحديث الحالة (State) بإجابة المستخدم قبل استئناف هذه العقدة
        return {}

    def market_analyst_advisor(state: State):
        prompt = f"Evaluate the market potential, competition, target demographics, and trends for this Idea:\n{state['messages']}"
        report = llm.invoke([SystemMessage(content=prompt)])
        return {"advisor_reports": {"Market Analyst": report.content}}

    def legal_advisor(state: State):
        prompt = f"Identify IP, licensing, GDPR, and compliance issues for this Idea:\n{state['messages']}"
        report = llm.invoke([SystemMessage(content=prompt)])
        return {"advisor_reports": {"Legal Advisor": report.content}}

    def technical_advisor(state: State):
        prompt = f"Estimate development complexity, tech stacks, and infrastructure risks for this Idea:\n{state['messages']}"
        report = llm.invoke([SystemMessage(content=prompt)])
        return {"advisor_reports": {"Technical Advisor": report.content}}

    def strategist_advisor(state: State):
        prompt = f"Define launch milestones, distribution channels, and early traction tactics for this Idea:\n{state['messages']}"
        report = llm.invoke([SystemMessage(content=prompt)])
        return {"advisor_reports": {"Strategist Advisor": report.content}}

    def collect_and_report(state: State):
        if len(state.get("advisor_reports", {})) < 4:
            return {}
        report_prompt = f"You are a senior consultant. Combine the four advisor notes below into one clear, structured evaluation report.\n{state['advisor_reports']}"
        report = llm.invoke(report_prompt).content
        return {"final_report": report}

    # تركيب مسارات Graph
    builder = StateGraph(State)
    builder.add_node("decide_node", decide_node)
    builder.add_node("human_node", human_node)
    builder.add_node("fanout", lambda state: {}) 
    builder.add_node("market_analyst_advisor", market_analyst_advisor)
    builder.add_node("legal_advisor", legal_advisor)
    builder.add_node("technical_advisor", technical_advisor)
    builder.add_node("strategist_advisor", strategist_advisor)
    builder.add_node("collect_and_report", collect_and_report)

    builder.set_entry_point("decide_node")
    builder.add_conditional_edges("decide_node", route, {"human_node": "human_node", "fanout": "fanout"})
    builder.add_edge("human_node", "decide_node")  # العودة للتقييم بعد إجابة المستخدم
    
    # تفريع المهام للوكلاء الأربعة
    for advisor in ["market_analyst_advisor", "legal_advisor", "technical_advisor", "strategist_advisor"]:
        builder.add_edge("fanout", advisor)
        builder.add_edge(advisor, "collect_and_report")
        
    builder.add_edge("collect_and_report", END)

    # استخدام MemorySaver لحفظ الجلسة وإيقاف الجراف قبل عقدة human_node
    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["human_node"])

# ==========================================
# 3. إدارة جلسة التشغيل والتفاعل (Session State)
# ==========================================
if api_key:
    graph = build_graph(api_key)
    
    # تعريف رقم الجلسة لضمان استمرار الذاكرة
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "startup_thread_1"
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # جلب مدخلات المستخدم
    user_input = st.chat_input("Enter your business idea or answer...")

    if user_input:
        current_state = graph.get_state(config)
        
        # التحقق: هل الجراف متوقف وينتظر إجابة المستخدم؟ (Human-in-the-loop)
        if current_state.next and "human_node" in current_state.next:
            # تمرير الإجابة للجراف وتحديث الحالة
            graph.update_state(config, {"messages": [HumanMessage(content=user_input)]}, as_node="human_node")
            with st.spinner("Analyzing your response and generating reports..."):
                graph.invoke(None, config) # استئناف التشغيل
        else:
            # بدء تشغيل جديد لفكرة جديدة
            init_state = {
                "idea": user_input,
                "messages": [HumanMessage(content=user_input)],
                "advisor_reports": {},
                "final_report": ""
            }
            with st.spinner("Evaluating your startup idea..."):
                graph.invoke(init_state, config)

    # ==========================================
    # 4. عرض المحادثة والنتيجة النهائية
    # ==========================================
    current_state = graph.get_state(config)
    
    if current_state.values:
        # طباعة سجل الدردشة بين المستخدم و decide_node
        for msg in current_state.values.get("messages", []):
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, AIMessage) and not msg.content.strip().upper().startswith("DONE"):
                st.chat_message("assistant").write(msg.content)
        
        # عرض التقرير النهائي إذا اكتمل
        if current_state.values.get("final_report"):
            st.divider()
            st.subheader("📑 Final Evaluation Report")
            st.info("تم إنشاء هذا التقرير عبر 4 وكلاء ذكاء اصطناعي (محلل سوق، مستشار قانوني، خبير تقني، ومخطط استراتيجي).")
            st.markdown(current_state.values["final_report"])

else:
    st.warning("👈 Please enter your Groq API Key in the sidebar to start.")