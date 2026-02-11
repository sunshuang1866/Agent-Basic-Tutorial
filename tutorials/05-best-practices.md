# 最佳实践

学习 Agent 开发的最佳实践，提升代码质量和系统性能。

## 1. Prompt 工程技巧

### 1.1 清晰的系统提示

好的系统提示能显著提升 Agent 性能：

```python
system_prompt = """
你是一个专业的 {role}。

职责：
- {responsibility_1}
- {responsibility_2}
- {responsibility_3}

工作原则：
- {principle_1}
- {principle_2}

注意事项：
- {note_1}
- {note_2}
"""
```

**示例：**

```python
data_analyst_prompt = """
你是一个专业的数据分析师。

职责：
- 分析数据并发现模式
- 生成可视化图表
- 提供数据驱动的洞察

工作原则：
- 始终基于实际数据
- 保持客观中立
- 提供可操作的建议

注意事项：
- 避免过度推测
- 说明分析的局限性
- 使用清晰的语言解释技术概念
"""
```

### 1.2 结构化输出

要求 Agent 使用特定格式输出：

```python
format_instructions = """
请按以下格式回答：

分析结果：
- 关键发现1
- 关键发现2

建议：
- 建议1
- 建议2

置信度：[高/中/低]
"""
```

### 1.3 Few-Shot 示例

提供示例帮助 Agent 理解任务：

```python
few_shot_examples = """
示例 1:
输入：分析销售数据
输出：
1. 数据概览：总销售额 100万，订单数 5000
2. 趋势分析：呈上升趋势，增长率 15%
3. 建议：继续当前策略，关注高价值客户

示例 2:
输入：查找异常值
输出：
1. 检测到 3 个异常值
2. 异常日期：2024-01-05, 2024-01-12, 2024-01-20
3. 可能原因：促销活动、系统错误
"""
```

## 2. Agent 设计原则

### 2.1 单一职责原则

每个工具应该只做一件事：

```python
# ❌ 不好的设计 - 工具功能过于复杂
def process_data_and_create_report(data, report_type, format, email):
    # 处理数据
    # 创建报告
    # 发送邮件
    pass

# ✅ 好的设计 - 功能分离
def process_data(data):
    """只负责处理数据"""
    pass

def create_report(processed_data, report_type):
    """只负责创建报告"""
    pass

def send_email(report, recipient):
    """只负责发送邮件"""
    pass
```

### 2.2 清晰的工具描述

工具描述直接影响 Agent 的选择：

```python
# ❌ 描述不清晰
Tool(
    name="DataTool",
    func=process,
    description="处理数据"
)

# ✅ 描述清晰具体
Tool(
    name="CalculateAverage",
    func=calculate_average,
    description="""
    计算数值列表的平均值。
    
    输入：逗号分隔的数字字符串，例如 "1,2,3,4,5"
    输出：平均值，保留两位小数
    
    使用场景：
    - 需要计算平均值时
    - 输入是数字列表时
    """
)
```

### 2.3 适当的工具粒度

```python
# ❌ 粒度过细
Tool(name="Add", func=add, description="两数相加")
Tool(name="Subtract", func=subtract, description="两数相减")
Tool(name="Multiply", func=multiply, description="两数相乘")
Tool(name="Divide", func=divide, description="两数相除")

# ✅ 合理的粒度
Tool(
    name="Calculate",
    func=calculate,
    description="""
    执行数学计算。
    支持加减乘除和基本数学函数。
    输入：数学表达式，如 "2 + 3 * 4" 或 "sqrt(16)"
    """
)
```

### 2.4 错误处理和验证

```python
from pydantic import BaseModel, Field, validator

class CalculatorInput(BaseModel):
    """计算器输入验证"""
    expression: str = Field(description="数学表达式")
    
    @validator('expression')
    def validate_expression(cls, v):
        """验证表达式安全性"""
        # 只允许特定字符
        allowed = set('0123456789+-*/()., ')
        if not all(c in allowed for c in v):
            raise ValueError("表达式包含不允许的字符")
        return v

def safe_calculate(expression: str) -> str:
    """安全的计算函数"""
    try:
        # 验证输入
        input_data = CalculatorInput(expression=expression)
        
        # 执行计算
        result = eval(input_data.expression)
        return f"结果：{result}"
    
    except ValueError as e:
        return f"输入错误：{str(e)}"
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误：{str(e)}"
```

## 3. 性能优化

### 3.1 缓存机制

```python
from functools import lru_cache
from datetime import datetime, timedelta

class CachedAgent:
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
    
    def get_with_cache(self, key, fetch_func):
        """带缓存的数据获取"""
        now = datetime.now()
        
        # 检查缓存
        if key in self.cache:
            data, timestamp = self.cache[key]
            if now - timestamp < self.cache_duration:
                return data
        
        # 获取新数据
        data = fetch_func()
        self.cache[key] = (data, now)
        return data

# 使用装饰器缓存
@lru_cache(maxsize=100)
def expensive_operation(param):
    """耗时操作"""
    # 执行复杂计算
    return result
```

### 3.2 异步执行

```python
import asyncio
from langchain.tools import BaseTool

class AsyncSearchTool(BaseTool):
    name = "AsyncSearch"
    description = "异步搜索工具"
    
    async def _arun(self, query: str) -> str:
        """异步执行"""
        # 并发执行多个搜索
        tasks = [
            self.search_source1(query),
            self.search_source2(query),
            self.search_source3(query),
        ]
        results = await asyncio.gather(*tasks)
        return self.merge_results(results)
    
    def _run(self, query: str) -> str:
        """同步执行（回退）"""
        return asyncio.run(self._arun(query))
```

### 3.3 限制迭代次数

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,           # 最大迭代次数
    max_execution_time=60,       # 最大执行时间（秒）
    early_stopping_method="force"  # 强制停止
)
```

### 3.4 选择合适的模型

```python
# 根据任务复杂度选择模型
simple_llm = ChatOpenAI(model="gpt-3.5-turbo")  # 简单任务
complex_llm = ChatOpenAI(model="gpt-4")          # 复杂任务

# 使用不同模型处理不同任务
if task_complexity == "simple":
    agent = initialize_agent(tools, simple_llm, ...)
else:
    agent = initialize_agent(tools, complex_llm, ...)
```

## 4. 安全性考虑

### 4.1 输入验证

```python
def validate_input(user_input: str) -> bool:
    """验证用户输入"""
    # 长度检查
    if len(user_input) > 1000:
        return False
    
    # 危险字符检查
    dangerous_patterns = ['rm -rf', 'DROP TABLE', 'eval(', 'exec(']
    if any(pattern in user_input for pattern in dangerous_patterns):
        return False
    
    return True

# 在 Agent 中使用
if not validate_input(user_input):
    return "输入包含不安全内容"
```

### 4.2 工具权限控制

```python
class RestrictedTool(BaseTool):
    """受限工具"""
    name = "RestrictedOperation"
    allowed_users = ["admin", "power_user"]
    
    def _run(self, query: str, user: str) -> str:
        # 检查权限
        if user not in self.allowed_users:
            return "权限不足"
        
        # 执行操作
        return self.execute(query)
```

### 4.3 敏感信息处理

```python
def sanitize_output(text: str) -> str:
    """清理输出中的敏感信息"""
    import re
    
    # 隐藏邮箱
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '***@***.***', text)
    
    # 隐藏手机号
    text = re.sub(r'\b1[3-9]\d{9}\b', '***********', text)
    
    # 隐藏身份证号
    text = re.sub(r'\b\d{17}[\dXx]\b', '******************', text)
    
    return text
```

## 5. 测试策略

### 5.1 单元测试

```python
import unittest
from unittest.mock import Mock, patch

class TestAgentTools(unittest.TestCase):
    def test_calculate_tool(self):
        """测试计算工具"""
        result = calculate("2 + 3")
        self.assertEqual(result, "5")
    
    def test_calculate_error(self):
        """测试错误处理"""
        result = calculate("invalid")
        self.assertIn("错误", result)
    
    @patch('requests.get')
    def test_api_call(self, mock_get):
        """测试 API 调用"""
        mock_get.return_value.json.return_value = {"data": "test"}
        result = fetch_data("query")
        self.assertEqual(result, {"data": "test"})
```

### 5.2 集成测试

```python
def test_agent_flow():
    """测试完整 Agent 流程"""
    agent = create_test_agent()
    
    # 测试场景1：简单查询
    response = agent.run("现在几点？")
    assert "时间" in response or "点" in response
    
    # 测试场景2：多步骤任务
    response = agent.run("查询天气并保存结果")
    assert "已保存" in response
    
    # 测试场景3：错误处理
    response = agent.run("执行不可能的任务")
    assert "无法" in response or "错误" in response
```

### 5.3 性能测试

```python
import time

def benchmark_agent(agent, test_queries, iterations=10):
    """性能基准测试"""
    times = []
    
    for _ in range(iterations):
        start = time.time()
        for query in test_queries:
            agent.run(query)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"平均执行时间：{avg_time:.2f}秒")
    print(f"最快：{min(times):.2f}秒")
    print(f"最慢：{max(times):.2f}秒")
```

## 6. 监控和日志

### 6.1 结构化日志

```python
import logging
import json
from datetime import datetime

class AgentLogger:
    def __init__(self):
        self.logger = logging.getLogger('agent')
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        handler = logging.FileHandler('agent.log')
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        self.logger.addHandler(handler)
    
    def log_interaction(self, user_input, agent_output, tools_used):
        """记录交互"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': user_input,
            'output': agent_output,
            'tools': tools_used,
        }
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
```

### 6.2 性能监控

```python
from functools import wraps
import time

def monitor_performance(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        # 记录性能数据
        print(f"{func.__name__} 执行时间：{duration:.2f}秒")
        
        # 如果执行时间过长，发出警告
        if duration > 10:
            print(f"警告：{func.__name__} 执行时间过长")
        
        return result
    return wrapper

@monitor_performance
def agent_run(query):
    return agent.run(query)
```

## 7. 文档和维护

### 7.1 工具文档

```python
def comprehensive_tool_example(param1: str, param2: int = 10) -> str:
    """
    工具功能的简短描述
    
    详细说明：
    这个工具用于...
    
    参数：
        param1 (str): 第一个参数的说明
        param2 (int, optional): 第二个参数的说明，默认为 10
    
    返回：
        str: 返回值的说明
    
    示例：
        >>> comprehensive_tool_example("test", 20)
        "处理结果"
    
    注意事项：
        - 注意事项1
        - 注意事项2
    """
    pass
```

### 7.2 版本控制

```python
# 工具版本管理
TOOL_VERSION = "1.0.0"

class VersionedTool(BaseTool):
    name = "VersionedTool"
    version = TOOL_VERSION
    
    def _run(self, query: str) -> str:
        # 根据版本处理不同逻辑
        if self.version >= "2.0.0":
            return self.new_implementation(query)
        else:
            return self.old_implementation(query)
```

## 最佳实践清单

✅ Prompt 工程
- [ ] 使用清晰的系统提示
- [ ] 提供结构化输出格式
- [ ] 添加 Few-Shot 示例

✅ 设计原则
- [ ] 工具功能单一
- [ ] 描述清晰具体
- [ ] 合理的粒度

✅ 性能优化
- [ ] 实现缓存机制
- [ ] 使用异步执行
- [ ] 限制迭代次数
- [ ] 选择合适模型

✅ 安全性
- [ ] 验证用户输入
- [ ] 控制工具权限
- [ ] 处理敏感信息

✅ 测试
- [ ] 编写单元测试
- [ ] 进行集成测试
- [ ] 执行性能测试

✅ 监控
- [ ] 添加结构化日志
- [ ] 监控性能指标
- [ ] 设置告警机制

✅ 文档
- [ ] 编写详细文档
- [ ] 添加使用示例
- [ ] 维护版本信息

## 下一步

- 探索 [高级主题](06-advanced-topics.md) 学习更多技术
- 查看 [常见问题](07-troubleshooting.md) 解决问题
- 参考 [资源列表](08-resources.md) 深入学习
