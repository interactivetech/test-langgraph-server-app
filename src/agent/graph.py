"""Define a simple chatbot agent.

This agent returns a predefined response without using an actual LLM.
"""

from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from agent.configuration import Configuration
from agent.state import State, FormatAns, Plan, GraphState
from typing_extensions import TypedDict
from typing import List, Dict

from agent.tools import (
    retrieve,
    gen_sql_query,
    execute_sql_query,
    answer,
    plan,
    gen_sql_query2,
    check,
    decide,
    rag_answer
)






# --- 

# Define a new graph
workflow = StateGraph(GraphState, config_schema=Configuration)

# Add the node to the graph
workflow.add_node("retrieve", retrieve)
workflow.add_node("check",check)
workflow.add_node('generate_sql_query',gen_sql_query2)
workflow.add_node('execute_sql_query',execute_sql_query)
workflow.add_node('planning',plan)
workflow.add_node('answer',answer)
workflow.add_node('rag_answer',rag_answer)
# Set the entrypoint as `call_model`
workflow.add_edge("__start__", "retrieve")
workflow.add_edge("retrieve",'check')
workflow.add_conditional_edges('check',
                               decide,
                               {
                                    'yes': 'planning',
                                    'no': 'rag_answer'
                               })
workflow.add_edge('planning','generate_sql_query')
workflow.add_edge('generate_sql_query','execute_sql_query')
workflow.add_edge('execute_sql_query','answer')
workflow.add_edge('answer','__end__')
# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "New Graph"  # This defines the custom name in LangSmith
