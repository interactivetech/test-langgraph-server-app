# Demo Deploying Langgraph Agent using Langgraph Serve (or LangGraph Cloud API) locally

Author: Andrew Mendez, 2025

# Setup environment Steps:
* requires python>=3.10
* `python -m venv my-venv`
* `source my-venv/bin/activate`
* `pip install -r requirements.txt`
* `unzip tickets_joined.db.zip`

# Steps to initalize repo
* `pip install -e . `
* `langgraph dev`
* url will pop up on your browswer: `https://smith.langchain.com/studio/thread?baseUrl=http%3A%2F%2F127.0.0.1%3A2024`
* In input, enter question: `Can you tell me about my flight's departure time? my ticket_no is \"7240005432906569\".`