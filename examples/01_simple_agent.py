"""
简单 Agent 示例

这个示例展示了如何创建一个基本的 Agent，包含一个简单的时间查询工具。

学习要点：
1. Agent 的基本结构
2. 如何定义工具
3. 如何与 Agent 交互
"""

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from datetime import datetime
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 检查 API 密钥
if not os.getenv('OPENAI_API_KEY'):
    print("错误：未找到 OPENAI_API_KEY 环境变量")
    print("请在 .env 文件中设置：OPENAI_API_KEY=your-api-key")
    exit(1)


def get_current_time() -> str:
    """
    获取当前时间
    
    Returns:
        str: 格式化的当前时间字符串
    """
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


def get_current_date() -> str:
    """
    获取当前日期
    
    Returns:
        str: 格式化的当前日期字符串
    """
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 星期%w")


def main():
    """主函数"""
    
    print("=" * 50)
    print("简单 Agent 示例")
    print("=" * 50)
    print()
    
    # 1. 初始化大语言模型
    print("正在初始化 Agent...")
    llm = ChatOpenAI(
        temperature=0,  # 温度为0，使输出更确定
        model="gpt-3.5-turbo"
    )
    
    # 2. 定义工具列表
    tools = [
        Tool(
            name="GetCurrentTime",
            func=get_current_time,
            description="获取当前的日期和时间。当用户询问现在几点或当前时间时使用此工具。"
        ),
        Tool(
            name="GetCurrentDate",
            func=get_current_date,
            description="获取当前的日期和星期。当用户询问今天是几号或星期几时使用此工具。"
        ),
    ]
    
    # 3. 初始化 Agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # 显示详细执行过程
        handle_parsing_errors=True  # 处理解析错误
    )
    
    # 4. 测试 Agent
    test_queries = [
        "现在几点了？",
        "今天是星期几？",
        "告诉我现在的完整日期和时间",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"测试 {i}: {query}")
        print(f"{'='*50}\n")
        
        try:
            response = agent.run(query)
            print(f"\n✅ 回答：{response}\n")
        except Exception as e:
            print(f"\n❌ 错误：{str(e)}\n")
    
    # 5. 交互模式
    print("\n" + "=" * 50)
    print("进入交互模式（输入 'quit' 或 'exit' 退出）")
    print("=" * 50 + "\n")
    
    while True:
        user_input = input("您的问题：").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出', 'q']:
            print("\n再见！👋")
            break
        
        if not user_input:
            continue
        
        try:
            response = agent.run(user_input)
            print(f"\n回答：{response}\n")
        except Exception as e:
            print(f"\n错误：{str(e)}\n")


if __name__ == "__main__":
    main()
