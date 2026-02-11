# 常见问题与故障排除

解决 Agent 开发和使用过程中遇到的常见问题。

## 1. 安装和配置问题

### Q1: 安装 LangChain 失败

**问题描述：**
```bash
pip install langchain
ERROR: Could not find a version that satisfies the requirement...
```

**解决方案：**

1. 升级 pip：
```bash
python -m pip install --upgrade pip
```

2. 使用国内镜像源：
```bash
pip install langchain -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. 检查 Python 版本（需要 3.8+）：
```bash
python --version
```

### Q2: API 密钥配置不生效

**问题描述：**
设置了环境变量但 Agent 仍然报告找不到 API 密钥。

**解决方案：**

1. 确认 `.env` 文件在正确位置：
```python
# 确保在代码开始处加载
from dotenv import load_dotenv
load_dotenv()  # 默认加载当前目录的 .env

# 或指定路径
load_dotenv('.env')
load_dotenv('/path/to/.env')
```

2. 检查环境变量是否加载：
```python
import os
print(os.getenv('OPENAI_API_KEY'))
```

3. 直接在代码中设置（仅用于测试）：
```python
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'
```

### Q3: 导入错误

**问题描述：**
```python
ImportError: cannot import name 'initialize_agent' from 'langchain.agents'
```

**解决方案：**

1. 更新 LangChain 到最新版本：
```bash
pip install --upgrade langchain
```

2. 检查导入路径：
```python
# 旧版本
from langchain.agents import initialize_agent

# 新版本可能变更了路径
from langchain.agents import AgentExecutor
```

3. 查看官方文档确认最新 API。

## 2. Agent 行为问题

### Q4: Agent 不调用工具

**问题描述：**
Agent 直接回答问题而不使用提供的工具。

**可能原因和解决方案：**

1. **工具描述不够清晰**

```python
# ❌ 描述模糊
Tool(
    name="Tool1",
    func=my_func,
    description="一个工具"
)

# ✅ 描述清晰
Tool(
    name="GetWeather",
    func=get_weather,
    description="获取指定城市的天气信息。输入应该是城市名称，例如'北京'或'上海'。"
)
```

2. **问题不需要工具**

```python
# LLM 可以直接回答的问题不会调用工具
# 确保问题确实需要工具才能回答
query = "现在北京的天气如何？"  # 需要工具
# 而不是：
query = "什么是天气？"  # 不需要工具
```

3. **工具太多导致选择困难**

```python
# 限制工具数量（建议少于 10 个）
tools = tools[:5]  # 只使用前 5 个工具
```

4. **使用 verbose 模式调试**

```python
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # 查看 Agent 的思考过程
)
```

### Q5: Agent 陷入无限循环

**问题描述：**
Agent 重复执行相同的操作，无法完成任务。

**解决方案：**

1. **限制最大迭代次数**

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # 限制迭代次数
    verbose=True
)
```

2. **设置超时时间**

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_execution_time=30,  # 30秒超时
    verbose=True
)
```

3. **检查工具的返回值**

```python
def my_tool(input_str):
    """确保返回明确的结果"""
    result = process(input_str)
    
    # ❌ 返回模糊
    return "处理完成"
    
    # ✅ 返回具体信息
    return f"处理完成，结果：{result}"
```

### Q6: Agent 输出格式错误

**问题描述：**
Agent 的输出格式不符合预期。

**解决方案：**

1. **在提示中明确要求格式**

```python
format_prompt = """
请按照以下格式回答：

标题：xxx
内容：xxx
结论：xxx
"""

agent.run(f"{format_prompt}\n\n问题：{query}")
```

2. **使用输出解析器**

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="answer", description="答案"),
    ResponseSchema(name="confidence", description="置信度"),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# 在提示中包含格式说明
prompt = f"{query}\n\n{format_instructions}"
```

3. **后处理输出**

```python
def clean_output(output):
    """清理和格式化输出"""
    # 移除多余空白
    output = output.strip()
    
    # 统一换行符
    output = output.replace('\r\n', '\n')
    
    # 其他清理逻辑
    return output

result = agent.run(query)
cleaned_result = clean_output(result)
```

## 3. 性能问题

### Q7: Agent 响应太慢

**问题描述：**
Agent 执行任务耗时很长。

**优化方案：**

1. **使用更快的模型**

```python
# 使用 GPT-3.5 而不是 GPT-4
llm = ChatOpenAI(model="gpt-3.5-turbo")  # 更快
# llm = ChatOpenAI(model="gpt-4")  # 更准确但更慢
```

2. **减少上下文长度**

```python
from langchain.memory import ConversationBufferWindowMemory

# 只保留最近几轮对话
memory = ConversationBufferWindowMemory(k=3)
```

3. **优化工具执行**

```python
import functools

@functools.lru_cache(maxsize=100)
def cached_tool(input_str):
    """缓存工具结果"""
    return expensive_operation(input_str)
```

4. **并行执行**

```python
import asyncio

async def parallel_agent_calls():
    tasks = [
        agent1.arun(query1),
        agent2.arun(query2),
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### Q8: 内存占用过高

**问题描述：**
Agent 运行时内存占用不断增长。

**解决方案：**

1. **限制记忆大小**

```python
from langchain.memory import ConversationSummaryMemory

# 使用摘要记忆而不是完整记忆
memory = ConversationSummaryMemory(llm=llm)
```

2. **定期清理**

```python
# 每 N 次交互后清理记忆
interaction_count = 0

def agent_run_with_cleanup(query):
    global interaction_count
    result = agent.run(query)
    
    interaction_count += 1
    if interaction_count > 10:
        memory.clear()
        interaction_count = 0
    
    return result
```

3. **使用流式处理**

```python
from langchain.callbacks import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

## 4. API 相关问题

### Q9: API 调用超时

**问题描述：**
```
timeout: The read operation timed out
```

**解决方案：**

1. **增加超时时间**

```python
llm = ChatOpenAI(
    request_timeout=60  # 60秒超时
)
```

2. **实现重试机制**

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_agent_with_retry(query):
    return agent.run(query)
```

3. **检查网络连接**

```python
import requests

def check_api_connectivity():
    try:
        response = requests.get('https://api.openai.com', timeout=5)
        return True
    except:
        return False

if check_api_connectivity():
    result = agent.run(query)
else:
    print("API 无法访问")
```

### Q10: API 配额超限

**问题描述：**
```
RateLimitError: Rate limit reached
```

**解决方案：**

1. **实现速率限制**

```python
import time
from functools import wraps

def rate_limit(max_per_minute):
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

@rate_limit(max_per_minute=20)
def call_agent(query):
    return agent.run(query)
```

2. **使用队列管理请求**

```python
from queue import Queue
import threading

class RequestQueue:
    def __init__(self, max_per_minute):
        self.queue = Queue()
        self.max_per_minute = max_per_minute
        self.start_worker()
    
    def start_worker(self):
        def worker():
            while True:
                query, callback = self.queue.get()
                result = agent.run(query)
                callback(result)
                time.sleep(60 / self.max_per_minute)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def submit(self, query, callback):
        self.queue.put((query, callback))
```

## 5. 工具相关问题

### Q11: 自定义工具不工作

**问题描述：**
自定义的工具无法被 Agent 正确调用。

**检查清单：**

1. **确保工具描述清晰**

```python
Tool(
    name="MyTool",
    func=my_function,
    description="""
    清晰描述工具的功能。
    
    输入格式：详细说明输入应该是什么
    输出格式：说明会返回什么
    使用场景：什么情况下应该使用这个工具
    """
)
```

2. **检查函数签名**

```python
# ✅ 正确：接受字符串参数，返回字符串
def my_tool(input_str: str) -> str:
    return f"处理结果：{input_str}"

# ❌ 错误：参数类型不匹配
def my_tool(input_dict: dict) -> dict:
    return {"result": "data"}
```

3. **添加错误处理**

```python
def robust_tool(input_str: str) -> str:
    try:
        result = process(input_str)
        return f"成功：{result}"
    except Exception as e:
        return f"错误：{str(e)}"
```

### Q12: 工具返回结果但 Agent 无法理解

**解决方案：**

1. **返回结构化信息**

```python
def better_tool(query):
    result = search(query)
    
    # ❌ 返回原始数据
    return result
    
    # ✅ 返回易于理解的文本
    return f"找到 {len(result)} 个结果。最相关的是：{result[0]}"
```

2. **添加上下文信息**

```python
def contextual_tool(query):
    result = process(query)
    
    return f"""
    查询：{query}
    结果：{result}
    状态：成功
    建议：基于此结果，您可以...
    """
```

## 6. 调试技巧

### 技巧 1: 使用详细日志

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 在关键位置添加日志
logger.debug(f"输入：{user_input}")
logger.debug(f"工具选择：{selected_tool}")
logger.debug(f"工具输出：{tool_output}")
```

### 技巧 2: 逐步执行

```python
# 分步测试
print("步骤 1: 测试工具")
tool_result = tool.run("test input")
print(f"工具结果：{tool_result}")

print("步骤 2: 测试 Agent")
agent_result = agent.run("test query")
print(f"Agent 结果：{agent_result}")
```

### 技巧 3: 使用断点调试

```python
import pdb

def debug_tool(input_str):
    # 设置断点
    pdb.set_trace()
    
    result = process(input_str)
    return result
```

### 技巧 4: 记录中间结果

```python
class DebugAgent:
    def __init__(self, agent):
        self.agent = agent
        self.history = []
    
    def run(self, query):
        self.history.append({
            'query': query,
            'timestamp': datetime.now()
        })
        
        result = self.agent.run(query)
        
        self.history[-1]['result'] = result
        return result
    
    def print_history(self):
        for item in self.history:
            print(f"Query: {item['query']}")
            print(f"Result: {item['result']}")
            print("---")
```

## 常见错误速查表

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `API key not found` | 未设置 API 密钥 | 检查环境变量配置 |
| `Rate limit exceeded` | API 调用过于频繁 | 实现速率限制 |
| `Timeout` | 请求超时 | 增加超时时间或优化操作 |
| `Parsing error` | 输出格式错误 | 优化提示词或添加解析器 |
| `Tool not found` | 工具名称错误 | 检查工具名称拼写 |
| `Max iterations` | 迭代次数超限 | 检查任务逻辑或增加限制 |

## 获取帮助

如果问题仍未解决：

1. **查看官方文档**
   - [LangChain 文档](https://python.langchain.com/)
   - [OpenAI 文档](https://platform.openai.com/docs)

2. **搜索 GitHub Issues**
   - [LangChain Issues](https://github.com/langchain-ai/langchain/issues)

3. **社区支持**
   - Stack Overflow
   - Discord 社区
   - GitHub Discussions

4. **提交 Issue**
   - 提供详细的错误信息
   - 包含最小可复现示例
   - 说明环境信息

## 下一步

- 查看 [资源列表](08-resources.md) 获取更多学习资料
- 回顾 [最佳实践](05-best-practices.md) 避免常见问题
- 参考 [实践案例](04-practical-examples.md) 学习正确用法
