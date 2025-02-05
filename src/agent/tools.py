from agent.prompts import (chat_prompt, 
                          process_prompt, 
                          plan_prompt, 
                          execute_prompt, 
                          process_prompt2, 
                          grade_prompt)

import sqlite3
from agent.state import State, FormatAns, Plan, Grade
from agent.models import llm, formatted_llm,plan_llm, embedding_model
from agent.docs import docs_list
from langchain.schema import Document  # Optional: For using Document objects
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_retriever(docs_list, embeddings):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    # (Optional) Convert dictionaries to Document objects for better compatibility
    documents = [Document(page_content=doc['page_content'], metadata=doc['metadata']) for doc in docs_list]

    # Split the documents into smaller chunks
    split_docs = text_splitter.split_documents(documents)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        collection_name="sql-rag-test3",
        embedding=embedding_model,
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    return retriever

def call_db(local_db_path=None, query=None, info=None):
    assert local_db_path is not None
    assert query is not None
    assert info is not None
    with sqlite3.connect(local_db_path) as local_conn:
        cursor = local_conn.cursor()
        cursor.execute(query, (info,))
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]
    return results



retriever = create_retriever(docs_list, embedding_model)
local_db_path = 'tickets_joined.db'





def retrieve(state):
    print("---RETRIEVE---")
    # Initialize history if not already present and add the user question
    state.setdefault("history", [])
    state["history"].append({"role": "user", "content": state["question"]})
    
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question, 'plan': None, "history": state["history"]}


def gen_sql_query(state):
    print("---GENERATE_SQL_QUERY---")
    formatted = chat_prompt.invoke({"context": state['documents'], 
                                    "question": state['question']})
    query_result = formatted_llm.invoke(formatted)
    return {"documents": state['documents'], 
            "question": state['question'],
            'plan': state['plan'],
            "structured_sql_query": query_result}

def execute_sql_query(state):
    print("---EXEC_SQL_QUERY---")
    print(state)
    query_result = state['structured_sql_query']
    r = None
    try:
        r = call_db(local_db_path=local_db_path,
                   query=query_result.sql_query,
                   info=query_result.user_input)
        print("r: ", r)
        return {"documents": state['documents'], 
                "question": state['question'],
                "structured_sql_query": state['structured_sql_query'],
                'plan': state['plan'],
                "sql_results": r}
    except Exception as e:
        print(e)
        return {"documents": state['documents'], 
                "question": state['question'],
                "structured_sql_query": state['structured_sql_query'],
                'plan': state['plan'],
                "sql_results": r}

def answer(state):
    print("---ANSWER---")
    print("state: ", state)
    # Combine the conversation history into a string
    conversation_history = "\n".join(f"{msg['role']}: {msg['content']}" for msg in state["history"])
    
    final_format = process_prompt.invoke({
        "history": conversation_history,
        "context": state["sql_results"],
        "question": state["question"]
    })
    print(final_format.to_string())
    answer_result = llm.invoke(final_format)
    
    # Append the assistant's answer to the conversation history
    state["history"].append({"role": "assistant", "content": answer_result.content})
    
    return {"documents": state["documents"], 
            "question": state["question"], 
            "structured_sql_query": state['structured_sql_query'],
            "sql_results": state['sql_results'],
            'plan': state['plan'],
            "generation": answer_result,
            "history": state["history"]}

def plan(state):
    docs = state['documents']
    question = state['question']
    plan_result = plan_prompt.invoke({"context": docs, "question": question})
    ans = plan_llm.invoke(plan_result)
    return {'question': state['question'],
            'documents': state['documents'],
            'plan': ans}

# update function
def gen_sql_query2(state):
    print("---GENERATE_SQL_QUERY---")
    plan_obj = state['plan']
    docs = state['documents']
    question = state['question']
    exec_prompt = execute_prompt.invoke({"plan": plan_obj.plan, "question": question, 'context': docs})
    print(exec_prompt.to_string())
    query_result = llm.with_structured_output(FormatAns).invoke(exec_prompt)
    print("query_result.user_input: ", query_result.user_input)
    print("query_result.sql_query: ", query_result.sql_query)
    return {"documents": state['documents'], 
            "question": state['question'],
            'plan': state['plan'],
            "structured_sql_query": query_result}

def check(state):
    docs = state['documents']
    print("docs: ", docs)
    question = state['question']
    print("question: ", question)
    grade = grade_prompt.invoke({"context": docs, "question": question})
    ans = llm.with_structured_output(Grade).invoke(grade)
    print("ans: ", ans)
    return {"documents": state['documents'], "question": state['question'], 'sql_gen_check': ans, 'plan': None}

def decide(state):
    ans = state['sql_gen_check'].binary_score
    if ans == 'yes':
        return 'yes'
    elif ans == 'no':
        return 'no'

def rag_answer(state):
    print("---ANSWER---")
    print("state: ", state)
    # Combine the conversation history into a string
    conversation_history = "\n".join(f"{msg['role']}: {msg['content']}" for msg in state["history"])
    
    final_format = process_prompt.invoke({
        "history": conversation_history,
        "context": state['documents'],
        "question": state["question"]
    })
    print(final_format.to_string())
    answer_result = llm.invoke(final_format)
    
    # Append the assistant's answer to the conversation history
    state["history"].append({"role": "assistant", "content": answer_result.content})
    
    return {"documents": state["documents"], 
            "question": state["question"], 
            'plan': state['plan'],
            "generation": answer_result,
            "history": state["history"]}
