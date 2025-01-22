import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_core.tools import FunctionTool
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.conditions import TextMentionTermination

from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()



async def main() -> None:
    tool = PythonCodeExecutionTool(DockerCommandLineCodeExecutor(work_dir="coding").restart())
    #PythonCodeExecutionTool(DockerCommandLineCodeExecutor(work_dir="coding"))
    # DockerCommandLineCodeExecutor
    # LocalCommandLineCodeExecutor
    agent = AssistantAgent(
        "assistant", 
        OpenAIChatCompletionClient(model="gpt-4o-mini"), 
        tools=[tool], 
        reflect_on_tool_use=True, 
        system_message=" generate one code block for the task and execute it."
    )
    

    result = await Console(
        agent.run_stream(
            task="Analyze American Airlines (AAL) stock, include last 2 years"
            # "Create a plot of MSFT stock prices in 2024 and save it to a file. Use yfinance and matplotlib."
        )
    )
    print(result.messages[-1].content)

asyncio.run(main())


