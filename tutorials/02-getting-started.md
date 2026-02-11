# 快速开始

本教程将指导您创建第一个 Agent 程序，了解 Agent 开发的基本流程。

## 环境准备

### 1. 安装 Python

确保您的系统已安装 Python 3.8 或更高版本：

```bash
python --version
```

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv agent-env

# 激活虚拟环境
# Windows
agent-env\Scripts\activate
# Linux/Mac
source agent-env/bin/activate
```

### 3. 安装依赖包

我们使用 LangChain 作为 Agent 开发框架：

```bash
pip install langchain openai python-dotenv
```

### 4. 配置 API 密钥

创建 `.env` 文件，添加您的 OpenAI API 密钥：

```bash
OPENAI_API_KEY=your-api-key-here
```

## 第一个 Agent 程序

### 示例 1：简单的对话 Agent

创建 `simple_agent.py` 文件：

```python
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# 定义一个简单的工具
def get_current_time():
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 创建工具列表
tools = [
    Tool(
        name="GetTime",
        func=get_current_time,
        description="获取当前系统时间"
    )
]

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 运行 Agent
if __name__ == "__main__":
    response = agent.run("现在几点了？")
    print(f"\n回答：{response}")
```

运行程序：

```bash
python simple_agent.py
```

### 示例 2：带计算功能的 Agent

```python
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化 LLM
llm = ChatOpenAI(temperature=0)

# 加载内置工具
tools = load_tools(["python_repl"], llm=llm)

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 测试计算功能
if __name__ == "__main__":
    # 简单计算
    result = agent.run("请计算 123 * 456 等于多少")
    print(f"结果：{result}")
    
    # 复杂计算
    result = agent.run("请帮我计算斐波那契数列的第10项")
    print(f"结果：{result}")
```

## 理解 Agent 的工作过程

当您运行上面的示例时，您会看到类似的输出：

```
> Entering new AgentExecutor chain...
我需要使用 GetTime 工具来获取当前时间

Action: GetTime
Action Input: 无需输入

Observation: 2024-01-15 14:30:25
Thought: 我现在知道当前时间了

Final Answer: 现在是 2024年1月15日 14:30:25

> Finished chain.
```

这个过程展示了 Agent 的核心工作机制：

1. **Thought（思考）**：Agent 分析问题，决定需要采取什么行动
2. **Action（行动）**：选择合适的工具
3. **Action Input（输入）**：为工具提供必要的输入
4. **Observation（观察）**：获取工具执行的结果
5. **Final Answer（最终答案）**：基于观察结果给出回答

## Agent 类型说明

LangChain 支持多种 Agent 类型：

### 1. ZERO_SHOT_REACT_DESCRIPTION
- 最常用的 Agent 类型
- 基于工具描述选择工具
- 适合大多数场景

### 2. CONVERSATIONAL_REACT_DESCRIPTION
- 支持对话上下文
- 能记住之前的对话内容
- 适合需要多轮对话的场景

### 3. STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
- 支持结构化输入
- 适合复杂工具调用
- 可以处理多参数工具

### 4. SELF_ASK_WITH_SEARCH
- 自我提问式 Agent
- 擅长分解复杂问题
- 适合需要逐步推理的场景

## 常见问题

### Q1: API 调用失败怎么办？

确认：
- API 密钥是否正确
- 网络连接是否正常
- API 额度是否充足

### Q2: Agent 响应很慢？

可以：
- 使用更快的模型（如 gpt-3.5-turbo）
- 减少工具数量
- 优化工具描述

### Q3: Agent 没有调用工具？

检查：
- 工具描述是否清晰
- 问题是否真的需要该工具
- 尝试使用 `verbose=True` 查看详细日志

## 调试技巧

### 1. 启用详细日志

```python
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # 显示详细执行过程
)
```

### 2. 设置最大迭代次数

```python
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=5  # 限制最大迭代次数
)
```

### 3. 添加错误处理

```python
try:
    response = agent.run("你的问题")
except Exception as e:
    print(f"Error: {e}")
```

## 实践练习

尝试完成以下练习：

1. **练习 1**：创建一个能够查询天气的 Agent
2. **练习 2**：实现一个简单的计算器 Agent
3. **练习 3**：开发一个文件操作 Agent

提示：查看 [实践案例](04-practical-examples.md) 获取更多示例。

## 下一步

- 学习 [核心概念](03-core-concepts.md) 深入理解 Agent 原理
- 查看 [实践案例](04-practical-examples.md) 了解更多应用场景
- 阅读 [最佳实践](05-best-practices.md) 提高开发技能

## 资源链接

- [LangChain 快速开始](https://python.langchain.com/docs/get_started/quickstart)
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [示例代码仓库](../examples/)
