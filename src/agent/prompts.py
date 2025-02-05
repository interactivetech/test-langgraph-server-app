from langchain.prompts import ChatPromptTemplate

# original sql generation prompt
template = """
You are an assistant for generating sql queries. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Only answer with a sql query.
**When the sql is generated, replace any user defined values with a question mark.**
===
Example:
Question: What airport am I flying out of? my passenger id is \"3442 587242.\"
Context:[Document(page_content='How to answer questions with What. Database has 11 rows: seat_no, fare_conditions, scheduled_arrival, scheduled_departure, ticket_no, book_ref, passenger_id, flight_id, flight_no, departure_airport, arrival_airport.\n        The database is named query_results.')]
user_input: 3442 587242
sql_query: SELECT departure_airport FROM query_results WHERE passenger_id = ?
===
Question: {question} 
Context: {context} 
"""

chat_prompt = ChatPromptTemplate.from_template(template)

# Updated process prompt now includes conversation history
process_template = """
You are an assistant that answers a user's question based on json. 
Here is the conversation history so far:
{history}

Answer the question using the following retrieved context:
Question: {question} 
Context: {context} 
If the context does not provide the answer, say that you cant answer the question based on the documents in the vector database. Answer in a personal, conversational tone.
"""

process_prompt = ChatPromptTemplate.from_template(process_template)

# Add prompt to create a plan on how to generate sql query
plan_template = """
You are a helpful assistant that generates two precise plans on how to implement a sql query.
Make sure to pay attention to the user input provided in the question.
===
Question: What airport am I flying out of? my passenger id is \"3442 587242.\"
Context:[Document(page_content='How to answer questions with What. Database has 11 rows: seat_no, fare_conditions, scheduled_arrival, scheduled_departure, ticket_no, book_ref, passenger_id, flight_id, flight_no, departure_airport, arrival_airport.\n        The database is named query_results.')]
Plan: This SQL command retrieves all columns (SELECT *) from a table named query_results where the passenger_id column matches the value provided by the ? placeholder. The ? is a parameter marker, indicating a value will be passed in later to complete the query.
===
Question: {question} 
Context: {context} 
"""

plan_prompt = ChatPromptTemplate.from_template(plan_template)

# Add prompt to create plan on executing plan
exec_template = """
You are an assistant for generating sql queries. Use the following pieces of a plan to answer the question. 
If you don't know the answer, just say that you don't know. Only answer with a sql query. When the sql is generated, replace the user defined values with a question mark.
*Do not have user input be in brackets or quotations at the final answer.
*Make sure to pay attention to the user input value, [ticket_no] is not correct.
Question: {question} 
Plan: {plan} 
"""
execute_prompt = ChatPromptTemplate.from_template(exec_template)

process_template2 = """
You are an assistant that answers a user's question based on json. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Do not answer based on the plan. Answer in a personal, conversational tone.
Question: {question} 
Plan: {plan}
Context: {context} 
"""

process_prompt2 = ChatPromptTemplate.from_template(process_template2)

# Add prompt to create plan on how to generate sql query
grade_template = """
Give a binary score 'yes' or 'no' to indicate whether the context and the question needs sql query generation.
Make sure to pay attention to the user input provided in the question.
===
Examples:
Question: How can I reschedule my flight?
Context: How to reschedule a flight: need to email returns@hpe.com, submit request, and someone will get back to you.
binary_score=no

Question: Can you tell me about my flight? my passenger id is \"3442 587242.\"
Context: [Document( page_content='How to answer questions with What. Database has 11 rows: seat_no, fare_conditions, scheduled_arrival, scheduled_departure, ticket_no, book_ref, passenger_id, flight_id, flight_no, departure_airport, arrival_airport.\n        The database is named query_results.'), Document( page_content='How to reschedule a flight: need to email returns@hpe.com, submit request, and someone will get back to you.'), Document( page_content='How to submit expense report: go to concur through home.hpe.com, and follow the necessary forms.')] 
binary_score=yes
===
Question: {question} 
Context: {context} 
"""

grade_prompt = ChatPromptTemplate.from_template(grade_template)

rag_template = """
You are a helpful assistant that answers question based on retrieved context.
Question: {question} 
Context: {context} 
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)