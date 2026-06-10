"""
AI 旅行规划师 - Streamlit 网页应用
支持行程生成、RAG 知识库、带记忆的行程对话
"""
import streamlit as st
import os
from ai_travel_planner import TravelPlannerAI

# ---------- 页面基本设置 ----------
st.set_page_config(page_title="AI 旅行规划师", layout="wide")
st.title("🌍 AI 旅行规划师")
st.markdown("基于 LangChain 的智能行程生成与旅行问答助手")

# ---------- 侧边栏配置 ----------
st.sidebar.header("⚙️ 配置")

# ----- API Key 获取逻辑（不显示输入框）-----
def get_api_key():
    try:
        key = st.secrets["DEEPSEEK_API_KEY"]
        if key:
            return key
    except (KeyError, AttributeError):
        pass
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if key:
        return key
    return None

api_key = get_api_key()

if api_key:
    st.sidebar.success("✅ API Key 已配置（从 Secrets 或环境变量加载）")
else:
    st.sidebar.error("❌ 未找到 API Key，请在 Streamlit Secrets 中设置 DEEPSEEK_API_KEY")

knowledge_dir = st.sidebar.text_input("知识库文件夹路径", value="travel_knowledge")

# ---------- 初始化 AI 引擎 ----------
if "ai_engine" not in st.session_state:
    st.session_state.ai_engine = None

if api_key:
    if st.session_state.ai_engine is None or st.session_state.ai_engine.api_key != api_key:
        with st.spinner("初始化 AI 引擎（加载知识库、嵌入模型...）"):
            try:
                st.session_state.ai_engine = TravelPlannerAI(api_key, knowledge_dir)
                st.sidebar.success("AI 引擎已就绪")
            except Exception as e:
                st.sidebar.error(f"初始化失败: {e}")
else:
    st.sidebar.warning("无法初始化 AI 引擎：缺少 API Key")

# ---------- 主界面：行程生成区域 ----------
st.header("📅 生成旅行计划")

col1, col2, col3, col4 = st.columns(4)
with col1:
    destination = st.text_input("目的地", placeholder="例如：京都")
with col2:
    days = st.number_input("天数", min_value=1, max_value=14, value=3)
with col3:
    budget = st.selectbox("预算", ["经济", "舒适", "豪华"])
with col4:
    interests = st.text_input("兴趣标签", placeholder="美食、历史、购物")

if st.button("✨ 生成行程", type="primary"):
    if not api_key:
        st.error("未配置 API Key，请在 Streamlit Secrets 中添加 DEEPSEEK_API_KEY")
    elif not destination:
        st.error("请输入目的地")
    else:
        ai = st.session_state.ai_engine
        if ai is None:
            st.error("AI 引擎未初始化")
        else:
            with st.spinner("正在结合本地攻略为您规划行程..."):
                try:
                    plan = ai.generate_plan(destination, days, budget, interests)
                    st.session_state.current_plan = plan
                    st.success("行程生成完毕！")
                except Exception as e:
                    st.error(f"生成失败: {e}")

# ---------- 显示行程 ----------
if "current_plan" in st.session_state:
    plan = st.session_state.current_plan
    st.subheader(f"📍 {plan.destination} {len(plan.days)}天详细行程")

    for day in plan.days:
        with st.expander(f"📌 Day {day.day} - {day.hotel or '未指定住宿'}", expanded=(day.day == 1)):
            if day.activities:
                for act in day.activities:
                    st.markdown(f"- **{act.time}**  {act.description}")
            if day.meals:
                st.markdown(f"🍽️ 餐饮推荐：{' / '.join(day.meals)}")
            st.caption(f"🏨 住宿：{day.hotel or '未指定'}")

    st.download_button(
        label="📥 下载行程 JSON",
        data=plan.model_dump_json(indent=2),
        file_name=f"{plan.destination}_travel_plan.json",
        mime="application/json"
    )

# ---------- 对话区域 ----------
st.header("💬 行程对话")
st.markdown("对已有行程进行追问、修改，或查询距离、汇率等")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("在这里输入你的问题...")
if user_input:
    ai = st.session_state.ai_engine
    if ai is None:
        st.error("AI 引擎未就绪，请检查 API Key")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("思考中..."):
            try:
                if "current_plan" not in st.session_state:
                    reply = "无法找到行程，请先生成行程。"
                else:
                    history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.chat_history[:-1]
                    ]
                    reply = ai.chat(user_input, history)
            except Exception as e:
                reply = f"出错啦：{e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.rerun()

if st.button("清空对话历史"):
    st.session_state.chat_history = []
    st.rerun()