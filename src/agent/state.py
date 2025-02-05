"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import List, Dict

@dataclass
class FormatAns(BaseModel):
    """Structured output for generated SQL query."""
    user_input: str = Field(description='The user specified input for the SQL query.')
    sql_query: str = Field(description="Syntactically valid SQL query.")

@dataclass
class Plan(BaseModel):
    """Plan to follow in future."""
    plan: str = Field(description="Steps to follow, should be in sorted order")

# Data model - how to track information over states
@dataclass
class Grade(BaseModel):
    """Binary score for relevance check to generate sql or do RAG."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The current user question.
        history: List of messages (each is a dict with 'role' and 'content').
        generation: LLM generation (assistant response).
        structured_sql_query: Generated SQL query with user defined value.
        sql_results: Data returned from SQL query (list of JSON dicts).
        plan: LLM generated plan on how to generate the SQL query.
        documents: List of documents.
        sql_gen_check: Grade, a binary score ('yes' or 'no').
    """
    question: str
    history: List[Dict[str, str]]
    generation: str
    structured_sql_query: FormatAns
    sql_results: List[Dict]
    plan: Plan
    documents: List[str]

@dataclass
class State:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    for more information.
    """

    changeme: str = "example"
