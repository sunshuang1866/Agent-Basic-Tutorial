"""
对话 Agent 示例

这个示例展示了如何创建一个支持多轮对话的 Agent，能够记住之前的对话内容。

学习要点：
1. 如何使用记忆系统
2. 多轮对话的实现
3. 对话上下文管理
"""

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def save_user_info(info: str) -> str:
    """
    保存用户信息（模拟）
    
    Args:
        info: 用户信息
        
    Returns:
        str: 确认消息
    """
    # 实际应用中可以保存到数据库
    return f"已记录信息：{info}"


def calculate(expression: str) -> str:
    """
    执行数学计算
    
    Args:
        expression: 数学表达式
        
    Returns:
        str: 计算结果
    """
    try:
        # 安全性注意：实际应用中应该使用更安全的计算方法
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算出错：{str(e)}"


def main():
    """主函数"""
    
    print("=" * 50)
    print("对话 Agent 示例")
    print("=" * 50)
    print("这个 Agent 能够记住之前的对话内容")
    print()
    
    # 1. 初始化 LLM
    llm = ChatOpenAI(
        temperature=0.7,  # 稍高的温度使对话更自然
        model="gpt-3.5-turbo"
    )
    
    # 2. 创建记忆系统
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # 3. 定义工具
    tools = [
        Tool(
            name="SaveUserInfo",
            func=save_user_info,
            description="保存用户的个人信息，如姓名、爱好等。输入应该是要保存的信息文本。"
        ),
        Tool(
            name="Calculate",
            func=calculate,
            description="执行数学计算。输入应该是数学表达式，如 '2+3' 或 '10*5'。"
        ),
    ]
    
    # 4. 初始化 Agent（使用支持对话的类型）
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # 5. 示例对话流程
    print("演示多轮对话能力：\n")
    
    conversations = [
        "你好！我叫张三。",
        "我喜欢编程和阅读。",
        "请帮我记住这些信息。",
        "我叫什么名字？",
        "我有什么爱好？",
    ]
    
    for i, message in enumerate(conversations, 1):
        print(f"\n{'='*50}")
        print(f"对话 {i}")
        print(f"{'='*50}")
        print(f"用户：{message}\n")
        
        try:
            response = agent.run(message)
            print(f"Agent：{response}\n")
        except Exception as e:
            print(f"错误：{str(e)}\n")
    
    # 6. 交互模式
    print("\n" + "=" * 50)
    print("进入交互模式")
    print("提示：Agent 会记住你告诉它的信息")
    print("输入 'quit' 退出")
    print("=" * 50 + "\n")
    
    while True:
        user_input = input("您：").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出', 'q']:
            print("\n再见！👋")
            break
        
        if not user_input:
            continue
        
        try:
            response = agent.run(user_input)
            print(f"Agent：{response}\n")
        except Exception as e:
            print(f"错误：{str(e)}\n")
    
    # 7. 显示对话历史
    print("\n" + "=" * 50)
    print("对话历史：")
    print("=" * 50)
    
    # 获取记忆内容
    memory_vars = memory.load_memory_variables({})
    if 'chat_history' in memory_vars:
        for msg in memory_vars['chat_history']:
            role = "用户" if hasattr(msg, 'type') and msg.type == 'human' else "Agent"
            print(f"{role}：{msg.content}")


if __name__ == "__main__":
    main()
