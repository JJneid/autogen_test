import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool
from autogen_core.code_executor import CodeBlock
from autogen_agentchat.conditions import TextMentionTermination
import asyncio
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from pathlib import Path

from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

async def main() -> None:
    # Use async with context manager for JupyterCodeExecutor
    async with JupyterCodeExecutor(
        kernel_name="python3",
        timeout=120,  # Increased timeout for data analysis
        output_dir=Path("coding")  # Specify output directory
    ) as executor:
        try:
            # Create the Python execution tool with Jupyter executor
            tool = PythonCodeExecutionTool(executor)
            
            agent = AssistantAgent(
                "assistant", 
                OpenAIChatCompletionClient(model="gpt-4o-mini"), 
                tools=[tool], 
                reflect_on_tool_use=True, 
                # system_message="""
                # generate one code block for the task and execute it. 
                # install dependencies within the code, use the following format to handle each package dependency:
                # import subprocess
                # subprocess.check_call(['pip', 'install', 'package_name'])
                # """
            )

            result = await Console(
                agent.run_stream(
                    task="Analyze American Airlines (AAL) stock, include last 2 years, use scikit learn"
                )
            )
            print(result.messages[-1].content)
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            raise

if __name__ == "__main__":
    asyncio.run(main())