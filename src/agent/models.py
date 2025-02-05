from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from agent.state import FormatAns, Plan

# Create models
llm = ChatNVIDIA(base_url="http://10.182.1.167:8080/v1",
                  model="meta/llama-3.1-70b-instruct", 
                  api_key="\'\'",
                  verbose=True)
formatted_llm = ChatNVIDIA(base_url="http://10.182.1.167:8080/v1",
                            model="meta/llama-3.1-70b-instruct", 
                            api_key="\'\'",
                            verbose=True).with_structured_output(FormatAns)
plan_llm = ChatNVIDIA(base_url="http://10.182.1.167:8080/v1",
                     model="meta/llama-3.1-70b-instruct", 
                     api_key="\'\'",
                     verbose=True).with_structured_output(Plan)

embedding_model = NVIDIAEmbeddings(base_url='http://embedding-tyler.models.mlds-kserve.us.rdlabs.hpecorp.net',
                               model="thenlper/gte-base", 
                               api_key='',
                               truncate="NONE")