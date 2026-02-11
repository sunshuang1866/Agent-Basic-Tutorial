# 高级主题

深入探索 Agent 开发的高级技术和复杂应用场景。

## 1. 多 Agent 系统

多个 Agent 协作可以处理更复杂的任务。

### 1.1 基本多 Agent 架构

```python
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

class MultiAgentSystem:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        
        # 创建专门的 Agent
        self.researcher = self.create_researcher()
        self.analyst = self.create_analyst()
        self.writer = self.create_writer()
    
    def create_researcher(self):
        """研究员 Agent"""
        tools = [...]  # 搜索工具
        return initialize_agent(
            tools, self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
    
    def create_analyst(self):
        """分析师 Agent"""
        tools = [...]  # 分析工具
        return initialize_agent(
            tools, self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
    
    def create_writer(self):
        """写作者 Agent"""
        tools = [...]  # 写作工具
        return initialize_agent(
            tools, self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
    
    def process_task(self, task):
        """处理任务的流程"""
        # 1. 研究员收集信息
        research_result = self.researcher.run(
            f"收集关于 {task} 的信息"
        )
        
        # 2. 分析师分析数据
        analysis_result = self.analyst.run(
            f"分析以下信息：{research_result}"
        )
        
        # 3. 写作者生成报告
        final_report = self.writer.run(
            f"基于以下分析写一份报告：{analysis_result}"
        )
        
        return final_report

# 使用示例
system = MultiAgentSystem()
result = system.process_task("人工智能的最新发展")
```

### 1.2 Agent 通信机制

```python
from typing import List, Dict
import json

class AgentCommunicationHub:
    """Agent 通信中心"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.message_queue = []
    
    def register_agent(self, name: str, agent):
        """注册 Agent"""
        self.agents[name] = agent
    
    def send_message(self, from_agent: str, to_agent: str, message: str):
        """发送消息"""
        self.message_queue.append({
            'from': from_agent,
            'to': to_agent,
            'message': message,
            'timestamp': datetime.now()
        })
    
    def get_messages(self, agent_name: str) -> List[Dict]:
        """获取发给特定 Agent 的消息"""
        return [
            msg for msg in self.message_queue
            if msg['to'] == agent_name
        ]
    
    def broadcast(self, from_agent: str, message: str):
        """广播消息给所有 Agent"""
        for agent_name in self.agents:
            if agent_name != from_agent:
                self.send_message(from_agent, agent_name, message)

# 使用示例
hub = AgentCommunicationHub()
hub.register_agent("researcher", researcher_agent)
hub.register_agent("analyst", analyst_agent)

# 发送消息
hub.send_message("researcher", "analyst", "这是收集到的数据")
```

### 1.3 协作模式

#### 顺序协作

```python
def sequential_collaboration(task):
    """顺序协作模式"""
    # Agent A -> Agent B -> Agent C
    result_a = agent_a.run(task)
    result_b = agent_b.run(f"基于 {result_a} 继续处理")
    result_c = agent_c.run(f"最终处理 {result_b}")
    return result_c
```

#### 并行协作

```python
import asyncio

async def parallel_collaboration(task):
    """并行协作模式"""
    # 多个 Agent 同时工作
    tasks = [
        agent_a.arun(task),
        agent_b.arun(task),
        agent_c.arun(task),
    ]
    results = await asyncio.gather(*tasks)
    
    # 合并结果
    final_result = merge_results(results)
    return final_result
```

#### 层级协作

```python
class HierarchicalAgentSystem:
    """层级 Agent 系统"""
    
    def __init__(self):
        self.manager = ManagerAgent()
        self.workers = [
            WorkerAgent("worker1"),
            WorkerAgent("worker2"),
            WorkerAgent("worker3"),
        ]
    
    def process(self, task):
        # 管理者分配任务
        subtasks = self.manager.decompose_task(task)
        
        # 工作者执行子任务
        results = []
        for subtask, worker in zip(subtasks, self.workers):
            result = worker.run(subtask)
            results.append(result)
        
        # 管理者整合结果
        final_result = self.manager.aggregate(results)
        return final_result
```

## 2. 长期记忆管理

### 2.1 向量数据库集成

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory

# 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings
)

# 创建记忆系统
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# 保存记忆
memory.save_context(
    {"input": "我喜欢吃苹果"},
    {"output": "好的，我记住了"}
)

# 检索记忆
relevant_memories = memory.load_memory_variables(
    {"input": "我喜欢什么水果？"}
)
```

### 2.2 知识图谱

```python
import networkx as nx

class KnowledgeGraph:
    """知识图谱"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_entity(self, entity, entity_type, properties=None):
        """添加实体"""
        self.graph.add_node(
            entity,
            type=entity_type,
            properties=properties or {}
        )
    
    def add_relation(self, entity1, entity2, relation):
        """添加关系"""
        self.graph.add_edge(entity1, entity2, relation=relation)
    
    def query(self, entity):
        """查询实体的关系"""
        if entity not in self.graph:
            return None
        
        # 获取相关信息
        neighbors = list(self.graph.neighbors(entity))
        properties = self.graph.nodes[entity].get('properties', {})
        
        return {
            'entity': entity,
            'properties': properties,
            'relations': neighbors
        }
    
    def find_path(self, start, end):
        """查找两个实体之间的路径"""
        try:
            path = nx.shortest_path(self.graph, start, end)
            return path
        except nx.NetworkXNoPath:
            return None

# 使用示例
kg = KnowledgeGraph()
kg.add_entity("张三", "Person", {"age": 30, "job": "工程师"})
kg.add_entity("ABC公司", "Company", {"industry": "科技"})
kg.add_relation("张三", "ABC公司", "works_at")

# 查询
info = kg.query("张三")
```

### 2.3 分层记忆系统

```python
class HierarchicalMemory:
    """分层记忆系统"""
    
    def __init__(self):
        # 短期记忆（最近的对话）
        self.short_term = []
        self.short_term_limit = 10
        
        # 工作记忆（当前任务相关）
        self.working_memory = {}
        
        # 长期记忆（重要信息）
        self.long_term = []
    
    def add_to_short_term(self, item):
        """添加到短期记忆"""
        self.short_term.append(item)
        if len(self.short_term) > self.short_term_limit:
            # 转移到长期记忆
            old_item = self.short_term.pop(0)
            if self.is_important(old_item):
                self.add_to_long_term(old_item)
    
    def add_to_working_memory(self, key, value):
        """添加到工作记忆"""
        self.working_memory[key] = value
    
    def add_to_long_term(self, item):
        """添加到长期记忆"""
        self.long_term.append(item)
    
    def is_important(self, item):
        """判断信息是否重要"""
        # 实现重要性判断逻辑
        return True
    
    def retrieve(self, query):
        """检索相关记忆"""
        results = []
        
        # 从各层记忆中检索
        results.extend(self.search_short_term(query))
        results.extend(self.search_working_memory(query))
        results.extend(self.search_long_term(query))
        
        return results
```

## 3. Agent 安全性

### 3.1 输入过滤

```python
import re
from typing import List

class SecurityFilter:
    """安全过滤器"""
    
    def __init__(self):
        # 危险模式列表
        self.dangerous_patterns = [
            r'rm\s+-rf',
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
        ]
    
    def is_safe(self, input_text: str) -> bool:
        """检查输入是否安全"""
        # 检查危险模式
        for pattern in self.dangerous_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                return False
        
        # 检查长度
        if len(input_text) > 10000:
            return False
        
        return True
    
    def sanitize(self, input_text: str) -> str:
        """清理输入"""
        # 移除特殊字符
        cleaned = re.sub(r'[<>\"\'\\]', '', input_text)
        
        # 限制长度
        cleaned = cleaned[:10000]
        
        return cleaned

# 使用示例
security_filter = SecurityFilter()

def safe_agent_run(user_input):
    if not security_filter.is_safe(user_input):
        return "输入包含不安全内容"
    
    cleaned_input = security_filter.sanitize(user_input)
    return agent.run(cleaned_input)
```

### 3.2 权限管理

```python
from enum import Enum
from typing import Set

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"

class PermissionManager:
    """权限管理器"""
    
    def __init__(self):
        self.user_permissions: Dict[str, Set[Permission]] = {}
        self.tool_requirements: Dict[str, Set[Permission]] = {}
    
    def grant_permission(self, user: str, permission: Permission):
        """授予权限"""
        if user not in self.user_permissions:
            self.user_permissions[user] = set()
        self.user_permissions[user].add(permission)
    
    def set_tool_requirement(self, tool: str, required: Set[Permission]):
        """设置工具所需权限"""
        self.tool_requirements[tool] = required
    
    def can_use_tool(self, user: str, tool: str) -> bool:
        """检查用户是否可以使用工具"""
        user_perms = self.user_permissions.get(user, set())
        required_perms = self.tool_requirements.get(tool, set())
        
        return required_perms.issubset(user_perms)

# 使用示例
pm = PermissionManager()
pm.grant_permission("user1", Permission.READ)
pm.grant_permission("admin", Permission.ADMIN)
pm.set_tool_requirement("DeleteFile", {Permission.WRITE, Permission.ADMIN})

if pm.can_use_tool("user1", "DeleteFile"):
    # 执行操作
    pass
```

### 3.3 审计日志

```python
import logging
from datetime import datetime
import json

class AuditLogger:
    """审计日志"""
    
    def __init__(self, log_file="audit.log"):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        self.logger.addHandler(handler)
    
    def log_action(self, user, action, details, status):
        """记录操作"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'action': action,
            'details': details,
            'status': status
        }
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def log_tool_use(self, user, tool, input_data, output_data):
        """记录工具使用"""
        self.log_action(
            user=user,
            action=f"use_tool:{tool}",
            details={
                'input': input_data,
                'output': output_data
            },
            status='success'
        )

# 使用示例
audit = AuditLogger()
audit.log_tool_use("user1", "FileDelete", "file.txt", "deleted")
```

## 4. 评估与测试

### 4.1 自动化评估

```python
class AgentEvaluator:
    """Agent 评估器"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def evaluate_accuracy(self, test_cases):
        """评估准确性"""
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            input_text = case['input']
            expected = case['expected']
            
            result = self.agent.run(input_text)
            if self.compare_results(result, expected):
                correct += 1
        
        accuracy = correct / total
        return accuracy
    
    def evaluate_latency(self, queries, iterations=10):
        """评估延迟"""
        import time
        
        times = []
        for _ in range(iterations):
            for query in queries:
                start = time.time()
                self.agent.run(query)
                end = time.time()
                times.append(end - start)
        
        return {
            'average': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'p95': sorted(times)[int(len(times) * 0.95)]
        }
    
    def evaluate_tool_selection(self, test_cases):
        """评估工具选择"""
        correct_selections = 0
        
        for case in test_cases:
            query = case['query']
            expected_tool = case['expected_tool']
            
            # 获取 Agent 使用的工具
            used_tools = self.get_used_tools(query)
            
            if expected_tool in used_tools:
                correct_selections += 1
        
        accuracy = correct_selections / len(test_cases)
        return accuracy
    
    def compare_results(self, result, expected):
        """比较结果"""
        # 实现比较逻辑
        return result.strip().lower() == expected.strip().lower()
    
    def get_used_tools(self, query):
        """获取使用的工具"""
        # 实现获取工具逻辑
        return []

# 使用示例
evaluator = AgentEvaluator(agent)

test_cases = [
    {'input': '现在几点？', 'expected': '包含时间信息'},
    {'input': '2+3等于多少？', 'expected': '5'},
]

accuracy = evaluator.evaluate_accuracy(test_cases)
print(f"准确率：{accuracy * 100:.2f}%")
```

### 4.2 A/B 测试

```python
class ABTester:
    """A/B 测试"""
    
    def __init__(self, agent_a, agent_b):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.results_a = []
        self.results_b = []
    
    def run_test(self, queries):
        """运行 A/B 测试"""
        for query in queries:
            # 测试 Agent A
            result_a = self.agent_a.run(query)
            self.results_a.append({
                'query': query,
                'result': result_a,
                'timestamp': datetime.now()
            })
            
            # 测试 Agent B
            result_b = self.agent_b.run(query)
            self.results_b.append({
                'query': query,
                'result': result_b,
                'timestamp': datetime.now()
            })
    
    def compare_performance(self):
        """比较性能"""
        # 实现性能比较逻辑
        return {
            'agent_a_wins': 0,
            'agent_b_wins': 0,
            'ties': 0
        }
```

## 5. 高级工具集成

### 5.1 API 集成

```python
import requests

class APITool(BaseTool):
    """API 工具基类"""
    
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
    
    def call_api(self, endpoint, params=None):
        """调用 API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f"{self.api_url}/{endpoint}",
            json=params,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API 调用失败：{response.status_code}")
```

### 5.2 数据库集成

```python
import sqlite3

class DatabaseTool(BaseTool):
    """数据库工具"""
    
    def __init__(self, db_path):
        self.db_path = db_path
    
    def query(self, sql, params=None):
        """执行查询"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql, params or [])
            results = cursor.fetchall()
            return results
        finally:
            conn.close()
    
    def execute(self, sql, params=None):
        """执行更新"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql, params or [])
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
```

## 总结

本章介绍了 Agent 开发的高级主题：

- ✅ 多 Agent 系统设计与协作
- ✅ 长期记忆和知识管理
- ✅ 安全性和权限控制
- ✅ 评估与测试方法
- ✅ 高级工具集成

## 下一步

- 查看 [常见问题](07-troubleshooting.md) 解决开发中的问题
- 参考 [资源列表](08-resources.md) 继续深入学习
- 实践项目，应用所学知识
