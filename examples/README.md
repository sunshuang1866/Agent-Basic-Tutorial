# Agent 示例代码

本目录包含各种 Agent 应用的示例代码。

## 📁 目录结构

```
examples/
├── README.md                   # 本文件
├── 01_simple_agent.py         # 简单 Agent 示例
├── 02_chat_agent.py           # 对话 Agent
├── 03_calculator_agent.py     # 计算器 Agent
├── 04_file_manager_agent.py   # 文件管理 Agent
├── 05_data_analysis_agent.py  # 数据分析 Agent
└── requirements.txt           # 依赖包列表
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd examples
pip install -r requirements.txt
```

### 2. 配置 API 密钥

创建 `.env` 文件：

```bash
OPENAI_API_KEY=your-api-key-here
```

### 3. 运行示例

```bash
# 简单 Agent
python 01_simple_agent.py

# 对话 Agent
python 02_chat_agent.py

# 其他示例...
```

## 📝 示例说明

### 01_simple_agent.py
**功能：** 基础 Agent 实现  
**学习点：**
- Agent 初始化
- 工具定义
- 基本交互

### 02_chat_agent.py
**功能：** 支持多轮对话的 Agent  
**学习点：**
- 记忆系统
- 对话上下文
- 连贯交互

### 03_calculator_agent.py
**功能：** 数学计算 Agent  
**学习点：**
- 工具集成
- 错误处理
- 结果验证

### 04_file_manager_agent.py
**功能：** 文件操作 Agent  
**学习点：**
- 文件系统操作
- 安全性考虑
- 权限控制

### 05_data_analysis_agent.py
**功能：** 数据分析 Agent  
**学习点：**
- 数据处理
- 可视化
- 报告生成

## 💡 使用建议

1. **从简单开始**：先运行 `01_simple_agent.py` 了解基础
2. **逐步深入**：按顺序学习各个示例
3. **动手修改**：尝试修改代码，观察效果
4. **解决问题**：遇到问题参考 [故障排除](../tutorials/07-troubleshooting.md)

## 🔧 自定义示例

可以基于这些示例创建自己的 Agent：

```python
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

# 1. 初始化 LLM
llm = ChatOpenAI(temperature=0)

# 2. 定义工具
def my_custom_tool(input_str: str) -> str:
    # 实现你的逻辑
    return f"处理结果：{input_str}"

tools = [
    Tool(
        name="MyTool",
        func=my_custom_tool,
        description="工具描述"
    )
]

# 3. 创建 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 4. 运行
response = agent.run("你的问题")
print(response)
```

## 📚 相关资源

- [教程文档](../tutorials/)
- [最佳实践](../tutorials/05-best-practices.md)
- [常见问题](../tutorials/07-troubleshooting.md)

## 🤝 贡献

欢迎贡献新的示例代码！请确保：

- 代码清晰易懂
- 包含注释说明
- 提供使用说明
- 测试通过

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../LICENSE) 文件。
