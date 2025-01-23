import asyncio
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock  # Added this import
from dotenv import load_dotenv
import os


# ask nalyze American Airlines (AAL) stock, include last 2 years, use scikit learn
# ask Create a prediction for next month using the existing model
load_dotenv()

async def main() -> None:
    # Create a persistent Jupyter executor
    async with JupyterCodeExecutor(
        kernel_name="python3",
        timeout=120,
        output_dir=Path("coding")
    ) as executor:
        try:
            tool = PythonCodeExecutionTool(executor)
            agent = AssistantAgent(
                "assistant", 
                OpenAIChatCompletionClient(model="gpt-4o-mini"), 
                tools=[tool], 
                reflect_on_tool_use=True,
                system_message="""
                You are a data analysis assistant. Maintain state between queries by:
                1. First check if required variables exist before recreating them
                2. Reference previously created variables and dataframes when relevant
                3. Let the user know what data is currently available in the session
                4. When creating new analysis, build upon existing data when possible
                """
            )

            # Interactive loop for multiple queries
            while True:
                # Get user input
                user_query = input("\nEnter your query (or 'exit' to quit): ")
                
                if user_query.lower() == 'exit':
                    break

                result = await Console(
                    agent.run_stream(task=user_query)
                )
                print("\nResponse:", result.messages[-1].content)
                
                # Optionally show available variables
                cancellation_token = CancellationToken()
                await executor.execute_code_blocks(
                    [CodeBlock(code="print('\nAvailable variables:', list(locals().keys()))", language="python")],
                    cancellation_token
                )
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            raise

if __name__ == "__main__":
    asyncio.run(main())