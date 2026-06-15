import os
import requests
from typing import List, Optional
from pydantic import BaseModel, Field
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 导入 LangChain 和 Agent 核心组件
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent # 使用 v1.0 标准接口
from langchain.tools import tool

# RAG 相关导入
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- 数据结构 ----------
class Activity(BaseModel):
    time: str = Field(description="活动时间，例如 09:00")
    description: str = Field(description="活动内容描述")

class DayPlan(BaseModel):
    day: int = Field(description="第几天")
    activities: List[Activity] = Field(description="当天活动列表")
    meals: List[str] = Field(description="推荐餐饮")
    hotel: Optional[str] = Field(description="推荐住宿")

class TravelPlan(BaseModel):
    destination: str = Field(description="目的地")
    days: List[DayPlan] = Field(description="每日行程")

travel_parser = PydanticOutputParser(pydantic_object=TravelPlan)

# ---------- 提示词 ----------
travel_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个资深的旅行规划师。请根据用户提供的旅行信息和下面给出的参考资料，生成一份详细的 {days} 天行程计划。

目的地：{destination}
预算：{budget}
兴趣标签：{interests}

【参考资料】（来自本地知识库，请优先参考）
{context}

输出格式要求：
{format_instructions}

注意：如果参考资料中有与目的地相关的具体景点、美食、交通、住宿等信息，请尽量采纳。"""),
    ("human", "请为我规划行程")
])

# ---------- LLM ----------
def get_llm(api_key: str):
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.7
    )

# ---------- 工具定义 ----------
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """汇率转换，例如 convert_currency(100, 'USD', 'CNY')。使用实时汇率。"""
    try:
        url = f"https://api.frankfurter.app/latest?from={from_currency.upper()}&to={to_currency.upper()}"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        rate = data['rates'][to_currency.upper()]
        converted = amount * rate
        return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()} (汇率: 1 {from_currency.upper()} = {rate} {to_currency.upper()})"
    except Exception as e:
        return f"汇率转换失败：{str(e)}"

@tool
def estimate_distance(origin: str, destination: str) -> str:
    """估算两个地点之间的距离（公里）。输入地名，例如 estimate_distance('东京站', '浅草寺')。"""
    try:
        headers = {'User-Agent': 'TravelPlannerAI/1.0'}
        geocode1 = requests.get(f"https://nominatim.openstreetmap.org/search?q={origin}&format=json&limit=1", headers=headers, timeout=5).json()
        geocode2 = requests.get(f"https://nominatim.openstreetmap.org/search?q={destination}&format=json&limit=1", headers=headers, timeout=5).json()
        if not geocode1 or not geocode2:
            return "无法获取其中一个地点的坐标"
        lat1, lon1 = float(geocode1[0]['lat']), float(geocode1[0]['lon'])
        lat2, lon2 = float(geocode2[0]['lat']), float(geocode2[0]['lon'])
        from math import radians, sin, cos, sqrt, atan2
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        return f"{origin} 到 {destination} 的直线距离约为 {distance:.1f} 公里"
    except Exception as e:
        return f"距离估算失败：{str(e)}"

# 工具列表
tools = [convert_currency, estimate_distance]

# ---------- RAG 向量库构建 ----------
def build_vectorstore(knowledge_dir: str):
    """构建 RAG 向量库"""
    # 检查知识库目录是否存在
    if not os.path.exists(knowledge_dir):
        print(f"知识库目录不存在: {knowledge_dir}")
        return None
    loader = DirectoryLoader(knowledge_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, recursive=True)
    md_loader = DirectoryLoader(knowledge_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, recursive=True)
    docs = loader.load() + md_loader.load()
    if not docs:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = os.path.join(knowledge_dir, "chroma_db")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    vectorstore.persist()
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------- 主类 ----------
class TravelPlannerAI:
    def __init__(self, api_key: str, knowledge_dir: str = "travel_knowledge"):
        self.api_key = api_key
        self.knowledge_dir = knowledge_dir
        self.llm = get_llm(api_key)
        self.current_plan = None
        self.plan_text = ""

        # RAG 初始化
        self.retriever = None
        if knowledge_dir and os.path.exists(knowledge_dir):
            try:
                self.retriever = build_vectorstore(knowledge_dir)
                print("RAG 检索器已加载")
            except Exception as e:
                print(f"RAG 初始化失败: {e}")

        # 构建 Agent
        self.agent = self._build_agent()

    def _build_agent(self):
        system_prompt = """你是一个智能旅行助手。你可以调用以下工具来回答用户的问题：
- convert_currency: 实时汇率转换
- estimate_distance: 估算两地距离

如果用户询问这类实时信息，你应该调用相应的工具。对于其他旅行相关问题，请基于已有行程计划回答。
请始终使用中文回答。"""
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
        )
        return agent

    def _retrieve_context(self, destination: str, interests: str) -> str:
    # 如果检索器未初始化，直接返回空字符串
        if not self.retriever:
            return ""
    # 构建查询语句，包含目的地、兴趣点和通用旅游关键词
        query = f"{destination} {interests} 旅游 景点 美食 交通 住宿"
        try:
        # 使用检索器执行查询
            docs = self.retriever.invoke(query)
        # 提取文档内容并去除空白字符，过滤掉空内容
            contexts = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
        # 如果没有检索到任何内容，返回空字符串
            if not contexts:
                return ""
        # 将所有检索到的上下文内容合并，并用分隔符分隔
            combined = "\n\n---\n\n".join(contexts)
        # 如果合并后的内容超过3000字符，进行截断处理
            if len(combined) > 3000:
                combined = combined[:3000] + "..."
            return combined
        except Exception as e:
            print(f"检索出错: {e}")
            return ""

    def generate_plan(self, destination: str, days: int, budget: str, interests: str) -> TravelPlan:
    # 检索与目的地和兴趣相关的上下文信息
        context = self._retrieve_context(destination, interests)
    # 设置旅行计划生成的处理链：提示词 -> 大语言模型 -> 结果解析器
        travel_chain = travel_prompt | self.llm | travel_parser
    # 准备输入参数字典
        inputs = {
            "destination": destination,      # 目的地
            "days": days,                    # 天数
            "budget": budget,                # 预算
            "interests": interests,          # 兴趣
            "context": context,              # 上下文信息
            "format_instructions": travel_parser.get_format_instructions()  # 格式说明
        }
    # 执行旅行计划生成链
        plan = travel_chain.invoke(inputs)
    # 保存生成的计划
        self.current_plan = plan
        self.plan_text = plan.model_dump_json(indent=2)  # 以格式化的JSON形式保存
        return plan  # 返回生成的旅行计划对象

    def chat(self, user_input: str, history: list) -> str:
        # 检查是否存在当前旅行计划，如果没有则提示用户先生成计划
        if not self.current_plan:
            return "请先生成旅行计划，然后再进行对话。"

        # 初始化列表用于存储格式化的对话历史
        lc_history = []
        # 遍历历史消息，将其转换为HumanMessage和AIMessage对象
        for msg in history:
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_history.append(AIMessage(content=msg["content"]))

        # 构建完整的输入内容，包含当前旅行计划和用户问题
        full_input = f"当前旅行计划：\n{self.plan_text}\n\n用户问题：{user_input}"

        try:
            # 调用智能助手处理输入，并获取回复
            response = self.agent.invoke({
                "messages": lc_history + [HumanMessage(content=full_input)]
            })
            # 获取最终消息内容
            final_message = response["messages"][-1]
            # 返回消息内容，如果对象有content属性则返回content，否则返回字符串形式
            return final_message.content if hasattr(final_message, 'content') else str(final_message)
        except Exception as e:
            return f"哎呀，智能助手出了点状况：{e}。请稍后再试。"