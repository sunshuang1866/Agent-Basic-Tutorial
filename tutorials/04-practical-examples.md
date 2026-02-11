# 实践案例

通过实际案例学习 Agent 的应用，从简单到复杂逐步提升。

## 案例 1：智能对话助手

创建一个能够回答问题并执行简单任务的对话助手。

### 代码实现

```python
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from datetime import datetime
import json

# 初始化 LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# 定义工具函数
def get_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算出错：{str(e)}"

def save_note(content: str) -> str:
    """保存笔记到文件"""
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {content}\n")
    return "笔记已保存"

# 创建工具列表
tools = [
    Tool(name="GetTime", func=get_time, 
         description="获取当前日期和时间"),
    Tool(name="Calculate", func=calculate, 
         description="执行数学计算。输入应该是一个数学表达式，如 '2+3' 或 '10*5'"),
    Tool(name="SaveNote", func=save_note, 
         description="保存笔记。输入应该是要保存的笔记内容"),
]

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 测试对话
if __name__ == "__main__":
    queries = [
        "现在几点了？",
        "帮我计算 123 * 456",
        "请帮我记录：今天学习了 Agent 开发",
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"问题：{query}")
        print(f"{'='*50}")
        response = agent.run(query)
        print(f"回答：{response}\n")
```

### 运行效果

```
==================================================
问题：现在几点了？
==================================================
> Entering new AgentExecutor chain...
Action: GetTime
Observation: 2024-01-15 14:30:25
Final Answer: 现在是 2024年1月15日 14:30:25
回答：现在是 2024年1月15日 14:30:25
```

## 案例 2：数据分析 Agent

创建一个能够分析数据、生成统计报告的 Agent。

### 代码实现

```python
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
import pandas as pd
import matplotlib.pyplot as plt
import json

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# 模拟数据
sales_data = pd.DataFrame({
    '日期': pd.date_range('2024-01-01', periods=30),
    '销售额': [1000 + i * 50 + (i % 7) * 100 for i in range(30)],
    '订单数': [20 + i + (i % 5) * 3 for i in range(30)]
})

def query_data(query_type: str) -> str:
    """查询数据"""
    if query_type == "summary":
        return sales_data.describe().to_json()
    elif query_type == "total":
        return f"总销售额: {sales_data['销售额'].sum()}, 总订单数: {sales_data['订单数'].sum()}"
    elif query_type == "average":
        return f"平均销售额: {sales_data['销售额'].mean():.2f}, 平均订单数: {sales_data['订单数'].mean():.2f}"
    return "请指定 summary, total, 或 average"

def calculate_growth(metric: str) -> str:
    """计算增长率"""
    if metric == "sales":
        first_week = sales_data['销售额'][:7].mean()
        last_week = sales_data['销售额'][-7:].mean()
        growth = ((last_week - first_week) / first_week) * 100
        return f"销售额增长率: {growth:.2f}%"
    elif metric == "orders":
        first_week = sales_data['订单数'][:7].mean()
        last_week = sales_data['订单数'][-7:].mean()
        growth = ((last_week - first_week) / first_week) * 100
        return f"订单数增长率: {growth:.2f}%"
    return "请指定 sales 或 orders"

def generate_chart(chart_type: str) -> str:
    """生成图表"""
    plt.figure(figsize=(10, 6))
    if chart_type == "sales":
        plt.plot(sales_data['日期'], sales_data['销售额'])
        plt.title('销售额趋势')
        plt.ylabel('销售额')
    elif chart_type == "orders":
        plt.plot(sales_data['日期'], sales_data['订单数'])
        plt.title('订单数趋势')
        plt.ylabel('订单数')
    else:
        return "请指定 sales 或 orders"
    
    plt.xlabel('日期')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{chart_type}_chart.png')
    plt.close()
    return f"图表已保存为 {chart_type}_chart.png"

tools = [
    Tool(name="QueryData", func=query_data,
         description="查询数据。输入 'summary' 获取统计摘要，'total' 获取总计，'average' 获取平均值"),
    Tool(name="CalculateGrowth", func=calculate_growth,
         description="计算增长率。输入 'sales' 计算销售额增长率，'orders' 计算订单数增长率"),
    Tool(name="GenerateChart", func=generate_chart,
         description="生成趋势图表。输入 'sales' 生成销售额图表，'orders' 生成订单数图表"),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    # 数据分析任务
    tasks = [
        "给我一个数据摘要",
        "销售额的增长情况如何？",
        "生成销售额趋势图",
    ]
    
    for task in tasks:
        print(f"\n任务：{task}")
        response = agent.run(task)
        print(f"结果：{response}\n")
```

## 案例 3：文件管理 Agent

创建一个能够管理文件和目录的 Agent。

### 代码实现

```python
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
import os
import shutil
from pathlib import Path

llm = ChatOpenAI(temperature=0)

# 工作目录
WORK_DIR = "./agent_workspace"
os.makedirs(WORK_DIR, exist_ok=True)

def list_files(directory: str = ".") -> str:
    """列出目录中的文件"""
    path = Path(WORK_DIR) / directory
    if not path.exists():
        return f"目录不存在: {directory}"
    
    files = []
    for item in path.iterdir():
        item_type = "目录" if item.is_dir() else "文件"
        files.append(f"{item_type}: {item.name}")
    
    return "\n".join(files) if files else "目录为空"

def create_file(filename: str, content: str = "") -> str:
    """创建文件"""
    path = Path(WORK_DIR) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"文件已创建: {filename}"

def read_file(filename: str) -> str:
    """读取文件内容"""
    path = Path(WORK_DIR) / filename
    if not path.exists():
        return f"文件不存在: {filename}"
    return path.read_text(encoding="utf-8")

def delete_file(filename: str) -> str:
    """删除文件"""
    path = Path(WORK_DIR) / filename
    if not path.exists():
        return f"文件不存在: {filename}"
    path.unlink()
    return f"文件已删除: {filename}"

def create_directory(dirname: str) -> str:
    """创建目录"""
    path = Path(WORK_DIR) / dirname
    path.mkdir(parents=True, exist_ok=True)
    return f"目录已创建: {dirname}"

tools = [
    Tool(name="ListFiles", func=list_files,
         description="列出目录中的文件和子目录。输入目录路径，默认为当前目录"),
    Tool(name="CreateFile", func=lambda x: create_file(*x.split('|', 1)),
         description="创建文件。输入格式：文件名|文件内容"),
    Tool(name="ReadFile", func=read_file,
         description="读取文件内容。输入文件名"),
    Tool(name="DeleteFile", func=delete_file,
         description="删除文件。输入文件名"),
    Tool(name="CreateDirectory", func=create_directory,
         description="创建目录。输入目录名"),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    commands = [
        "创建一个名为 test.txt 的文件，内容是 'Hello, Agent!'",
        "列出所有文件",
        "读取 test.txt 的内容",
        "创建一个名为 docs 的目录",
    ]
    
    for cmd in commands:
        print(f"\n命令：{cmd}")
        response = agent.run(cmd)
        print(f"结果：{response}")
```

## 案例 4：代码生成 Agent

创建一个能够生成和执行代码的 Agent。

### 代码实现

```python
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
import subprocess

llm = ChatOpenAI(temperature=0, model="gpt-4")

def generate_code(description: str) -> str:
    """根据描述生成代码"""
    prompt = f"请生成Python代码实现以下功能：{description}\n只返回代码，不要解释。"
    response = llm.predict(prompt)
    return response

def execute_code(code: str) -> str:
    """执行Python代码"""
    try:
        # 保存代码到临时文件
        with open("temp_code.py", "w", encoding="utf-8") as f:
            f.write(code)
        
        # 执行代码
        result = subprocess.run(
            ["python", "temp_code.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return f"执行成功:\n{result.stdout}"
        else:
            return f"执行失败:\n{result.stderr}"
    except Exception as e:
        return f"执行错误: {str(e)}"

def save_code(filename: str, code: str) -> str:
    """保存代码到文件"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
    return f"代码已保存到: {filename}"

tools = [
    Tool(name="GenerateCode", func=generate_code,
         description="根据功能描述生成Python代码"),
    Tool(name="ExecuteCode", func=execute_code,
         description="执行Python代码。输入应该是完整的Python代码"),
    Tool(name="SaveCode", func=lambda x: save_code(*x.split('|', 1)),
         description="保存代码到文件。输入格式：文件名|代码内容"),
]

# 加载 Python REPL 工具
python_repl = load_tools(["python_repl"], llm=llm)
tools.extend(python_repl)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    tasks = [
        "生成一个计算斐波那契数列的函数",
        "生成一个读取CSV文件并计算平均值的脚本",
    ]
    
    for task in tasks:
        print(f"\n任务：{task}")
        response = agent.run(task)
        print(f"结果：{response}")
```

## 案例 5：多功能助手（综合案例）

结合多种功能的综合助手。

### 代码实现

```python
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
import requests
from datetime import datetime

llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 定义各种工具
def get_weather(city: str) -> str:
    """获取天气信息（模拟）"""
    # 实际应用中应调用天气 API
    return f"{city}的天气：晴，温度20-28°C"

def search_web(query: str) -> str:
    """网络搜索（模拟）"""
    return f"搜索'{query}'的结果：[相关信息]"

def set_reminder(reminder: str) -> str:
    """设置提醒"""
    with open("reminders.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {reminder}\n")
    return "提醒已设置"

def translate_text(text: str) -> str:
    """翻译文本（模拟）"""
    # 实际应用中应调用翻译 API
    return f"翻译结果：{text}"

tools = [
    Tool(name="GetWeather", func=get_weather,
         description="获取指定城市的天气信息"),
    Tool(name="SearchWeb", func=search_web,
         description="在网络上搜索信息"),
    Tool(name="SetReminder", func=set_reminder,
         description="设置提醒事项"),
    Tool(name="TranslateText", func=translate_text,
         description="翻译文本"),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

if __name__ == "__main__":
    # 交互式对话
    print("多功能助手已启动！输入 'quit' 退出。\n")
    
    while True:
        user_input = input("您: ")
        if user_input.lower() == 'quit':
            break
        
        response = agent.run(user_input)
        print(f"助手: {response}\n")
```

## 实践建议

### 1. 从简单开始
- 先实现基本功能
- 逐步添加新工具
- 测试每个组件

### 2. 工具设计
- 保持工具功能单一
- 提供清晰的描述
- 处理边界情况

### 3. 错误处理
- 添加异常捕获
- 提供有用的错误信息
- 实现重试机制

### 4. 性能优化
- 缓存常用结果
- 异步执行耗时操作
- 限制迭代次数

## 下一步

- 学习 [最佳实践](05-best-practices.md) 优化您的 Agent
- 探索 [高级主题](06-advanced-topics.md) 深入研究
- 查看更多示例代码在 [examples](../examples/) 目录

## 练习题

1. 实现一个待办事项管理 Agent
2. 创建一个代码审查 Agent
3. 开发一个数据可视化 Agent
4. 构建一个文档生成 Agent

试着结合多个工具，创造您自己的 Agent 应用！
