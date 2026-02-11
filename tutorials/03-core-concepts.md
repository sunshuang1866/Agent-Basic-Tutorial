# 核心概念

深入理解 Agent 的核心概念和关键组件，掌握 Agent 系统的工作原理。

## Agent 架构

### 基本架构图

```
┌─────────────────────────────────────────────┐
│              用户交互层                      │
│         (User Interface)                    │
└───────────────┬─────────────────────────────┘
                │
┌───────────────▼─────────────────────────────┐
│              Agent 核心                      │
│  ┌──────────────────────────────────────┐  │
│  │  大语言模型 (LLM)                     │  │
│  │  - 理解与推理                         │  │
│  │  - 决策与规划                         │  │
│  └──────────────────────────────────────┘  │
└───┬────────────┬────────────┬──────────────┘
    │            │            │
┌───▼────┐  ┌───▼────┐  ┌───▼────┐
│ 工具层  │  │ 记忆层  │  │ 规划层  │
│ Tools  │  │ Memory │  │Planner │
└────────┘  └────────┘  └────────┘
```

## 1. 大语言模型（LLM）

LLM 是 Agent 的核心"大脑"，负责理解、推理和决策。

### 关键特性

- **自然语言理解**：理解用户意图
- **推理能力**：逻辑推理和问题分析
- **知识储备**：丰富的世界知识
- **生成能力**：生成文本和代码

### 选择 LLM 的考虑因素

| 因素 | 说明 | 推荐 |
|------|------|------|
| 性能 | 响应速度和准确度 | GPT-4 > GPT-3.5 |
| 成本 | API 调用费用 | GPT-3.5 更经济 |
| 上下文长度 | 支持的 token 数量 | GPT-4-32k 最大 |
| 特定能力 | 代码、推理等 | 根据需求选择 |

### 示例：配置不同的 LLM

```python
from langchain.chat_models import ChatOpenAI

# GPT-3.5 - 快速且经济
llm_fast = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# GPT-4 - 更强大但较慢
llm_powerful = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

# 自定义参数
llm_custom = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,      # 控制创造性
    max_tokens=1000,      # 限制输出长度
    presence_penalty=0.5  # 避免重复
)
```

## 2. 工具系统（Tools）

工具是 Agent 与外部世界交互的接口，扩展了 Agent 的能力。

### 工具的组成

1. **名称（Name）**：工具的唯一标识
2. **描述（Description）**：告诉 Agent 工具的功能
3. **函数（Function）**：实际执行的代码
4. **参数（Arguments）**：工具需要的输入

### 创建自定义工具

```python
from langchain.tools import Tool
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# 方法 1: 使用 Tool 类
def search_database(query: str) -> str:
    """在数据库中搜索信息"""
    # 实现搜索逻辑
    return f"搜索结果：{query}"

search_tool = Tool(
    name="DatabaseSearch",
    func=search_database,
    description="在数据库中搜索信息。输入应该是搜索查询。"
)

# 方法 2: 继承 BaseTool
class CustomSearchTool(BaseTool):
    name = "CustomSearch"
    description = "自定义搜索工具"
    
    def _run(self, query: str) -> str:
        """同步执行"""
        return f"搜索：{query}"
    
    async def _arun(self, query: str) -> str:
        """异步执行"""
        return f"异步搜索：{query}"

# 方法 3: 使用 @tool 装饰器
from langchain.tools import tool

@tool
def calculate_sum(numbers: str) -> str:
    """计算数字的和。输入应该是逗号分隔的数字。"""
    nums = [float(n.strip()) for n in numbers.split(',')]
    return str(sum(nums))
```

### 内置工具示例

```python
from langchain.agents import load_tools

# 加载内置工具
tools = load_tools(
    [
        "python_repl",        # Python 执行
        "requests_get",       # HTTP GET 请求
        "terminal",           # 终端命令
        "wikipedia",          # 维基百科查询
        "google-search",      # Google 搜索
    ],
    llm=llm
)
```

### 工具设计最佳实践

1. **清晰的描述**：让 Agent 准确理解工具用途
2. **合理的粒度**：不要太复杂也不要太简单
3. **错误处理**：优雅地处理异常情况
4. **性能考虑**：避免长时间运行的操作

## 3. 记忆系统（Memory）

记忆系统使 Agent 能够记住历史信息，实现连贯的对话。

### 记忆类型

#### 3.1 对话缓冲记忆（ConversationBufferMemory）

保存所有历史对话：

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

#### 3.2 对话摘要记忆（ConversationSummaryMemory）

对历史对话进行摘要：

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history"
)
```

#### 3.3 对话窗口记忆（ConversationBufferWindowMemory）

只保留最近 K 轮对话：

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=5,  # 保留最近 5 轮对话
    memory_key="chat_history"
)
```

#### 3.4 实体记忆（ConversationEntityMemory）

记住对话中的关键实体：

```python
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(
    llm=llm,
    memory_key="entity_memory"
)
```

### 使用记忆的 Agent

```python
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,  # 添加记忆
    verbose=True
)

# 第一次对话
response1 = agent.run("我叫张三")
# 第二次对话 - Agent 会记住之前的对话
response2 = agent.run("我叫什么名字？")
```

## 4. 规划系统（Planning）

规划系统负责将复杂任务分解为可执行的步骤。

### ReAct 模式

ReAct (Reasoning + Acting) 是最常用的规划模式：

```
Thought: 我需要做什么？
Action: 选择要使用的工具
Action Input: 工具的输入
Observation: 工具的输出
... (重复上述过程)
Thought: 我现在知道答案了
Final Answer: 最终答案
```

### 示例：ReAct 过程

```
问题：今天北京的天气如何？明天适合户外活动吗？

Thought: 我需要先查询今天北京的天气
Action: WeatherQuery
Action Input: 北京，今天
Observation: 北京今天晴，气温 20-28°C

Thought: 现在我需要查询明天的天气
Action: WeatherQuery
Action Input: 北京，明天
Observation: 北京明天多云，气温 18-25°C

Thought: 根据天气情况，我可以给出建议了
Final Answer: 今天北京天气晴朗，气温 20-28°C。
明天多云，气温 18-25°C，天气适宜户外活动。
```

### Plan-and-Execute 模式

适合更复杂的任务：

```python
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner
)

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(
    planner=planner,
    executor=executor,
    verbose=True
)
```

## 5. 提示工程（Prompt Engineering）

### System Prompt 设计

```python
system_prompt = """
你是一个专业的数据分析助手。

你的职责是：
1. 帮助用户分析数据
2. 生成可视化图表
3. 提供数据洞察

工作原则：
- 始终基于数据事实
- 提供清晰的解释
- 避免过度推测

可用工具：
- DataQuery: 查询数据
- PlotChart: 生成图表
- Statistics: 计算统计指标
"""
```

### Few-Shot Learning

通过示例提升性能：

```python
examples = """
示例 1:
用户: 帮我计算平均值
思考: 我需要使用 Statistics 工具
动作: Statistics
输入: average

示例 2:
用户: 显示销售趋势
思考: 我需要先查询数据，然后绘制图表
动作: DataQuery
输入: sales_data
"""
```

## 6. 错误处理与重试

### 实现重试机制

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,        # 最大迭代次数
    max_execution_time=60,   # 最大执行时间（秒）
    early_stopping_method="generate",  # 提前停止策略
    handle_parsing_errors=True,        # 处理解析错误
)
```

### 自定义错误处理

```python
def handle_tool_error(error: Exception) -> str:
    """自定义错误处理"""
    return f"工具执行出错：{str(error)}。请尝试其他方法。"

tool = Tool(
    name="MyTool",
    func=my_function,
    description="工具描述",
    handle_tool_error=handle_tool_error
)
```

## 总结

本章介绍了 Agent 的核心概念：

- ✅ LLM 作为 Agent 的大脑
- ✅ 工具系统扩展 Agent 能力
- ✅ 记忆系统保持对话连贯性
- ✅ 规划系统实现任务分解
- ✅ 提示工程优化 Agent 行为
- ✅ 错误处理保证系统稳定性

## 下一步

- 查看 [实践案例](04-practical-examples.md) 学习实际应用
- 阅读 [最佳实践](05-best-practices.md) 提升开发技能
- 探索 [高级主题](06-advanced-topics.md) 掌握进阶技术
