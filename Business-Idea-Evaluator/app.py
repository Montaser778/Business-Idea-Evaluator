import streamlit as st  # استيراد مكتبة ستريم ليت للواجهة
from langchain_groq import ChatGroq  # استيراد مكتبة الربط مع Groq
from langchain_core.messages import HumanMessage, SystemMessage  # استيراد رسائل LangChain

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="AI Startup Engine", layout="wide", page_icon="⚡")

# تصميم CSS احترافي (Dark UI)
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a !important;}
    h1 {color: #00ffcc !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase; margin-bottom: 5px;}
    .sub-title {color: #ffffff !important; font-size: 1.2rem; margin-bottom: 5px;}
    .dev-credit {color: #00ffcc !important; font-size: 0.9rem; font-style: italic; margin-bottom: 2rem;}
    [data-testid="stChatMessage"] {background-color: #151515 !important; border: 1px solid #333 !important; border-radius: 10px;}
    .stInfo {background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #00ffcc !important;}
    </style>
""", unsafe_allow_html=True)

# العنوان
st.title("⚡ AI STARTUP ENGINE") 
st.markdown("<p class='sub-title'>Advanced Multi-Agent Business Idea Analysis System</p>", unsafe_allow_html=True)
st.markdown("<p class='dev-credit'>Developed by Eng. Montaser</p>", unsafe_allow_html=True)

# التحقق من مفتاح الـ API
if "GROQ_API_KEY" not in st.secrets:
    st.error("مفتاح GROQ_API_KEY مفقود في إعدادات الـ Secrets.")
    st.stop()

# --- 2. بناء المحرك (وكيل واحد ذكي ومستقر) ---
# نستخدم الموديل القوي 70B للتحليل العميق في طلب واحد لضمان الاستقرار
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=st.secrets["GROQ_API_KEY"])

# --- 3. التفاعل مع المستخدم ---
if prompt := st.chat_input("أدخل فكرتك التجارية - سأقوم بتحليلها فوراً..."):
    st.chat_message("user").write(prompt)
    
    with st.status("🚀 جاري التحليل الاحترافي...", expanded=True) as status:
        st.write("🔍 جاري معالجة الفكرة بواسطة المحرك الذكي...")
        
        # برومبت شامل يقوم بكل خطوات التحليل في استدعاء واحد (سريع ومستقر)
        full_analysis_prompt = f"""
        Act as an expert consulting team (Market, Legal, Tech, Strategy).
        Analyze this business idea: {prompt}
        Provide a structured, deep, and professional report covering:
        1. Market Analysis & Competition.
        2. Legal Risks & Regulatory Compliance.
        3. Technical Feasibility & Architecture.
        4. Go-to-Market Strategy & Growth Plan.
        """
        
        try:
            # استدعاء الموديل
            response = llm.invoke([SystemMessage(content="You are a senior startup consultant."), HumanMessage(content=full_analysis_prompt)])
            
            # عرض النتيجة
            st.chat_message("assistant").write(response.content)
            status.update(label="✅ تم التحليل بنجاح.", state="complete")
        except Exception as e:
            st.error(f"خطأ في الاتصال: {str(e)}")
            status.update(label="❌ فشل التحليل.", state="error")
