"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel, Field

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


@dataclass
class State:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    for more information.
    """

    changeme: str = "example"
