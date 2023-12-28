from agent import initialize_components
from agent import load_and_process_data
from agent import create_vector_db_from_documents, create_vector_db_from_texts
from agent import create_sql_retriever_tool, create_name_retriever_tool
from agent import create_agent
from agent import run_query_save_results

import streamlit as st
TOKENIZERS_PARALLELISM=False

def main():
    # Initialize components
    llm, embeddings, db = initialize_components()

    # Load and process data
    csv_path = "/Users/hanif/personal-finance-agent/data/query_db.csv"
    query_docs = load_and_process_data(csv_path)

    # Create vector stores
    vector_db = create_vector_db_from_documents(query_docs, embeddings)
    # Run SQL queries and get texts
    categories = run_query_save_results(db, "SELECT category FROM transactions")
    merchants = run_query_save_results(db, "SELECT merchant_name FROM transactions")
    texts = categories + merchants    
    vector_db_text = create_vector_db_from_texts(texts, embeddings)

    # Create retrievers
    sql_retriever_tool = create_sql_retriever_tool(vector_db.as_retriever())
    name_retriever_tool = create_name_retriever_tool(vector_db_text.as_retriever())

    # Create agent
    custom_tool_list = [sql_retriever_tool, name_retriever_tool]
    agent = create_agent(llm, db, custom_tool_list)
    
    """
    # This snippet is for running with streamlit app

    st.set_page_config(page_title="Personal Finance Agent")
    st.title("Personal Finance Agent")
    st.info(
    "Personal finance agent which can answer user query based on the database"
)
    query_text = st.text_input("Enter your question:", placeholder="How much did I spend on shopping last month?")
    # Form input and query
    result = None
    with st.form("myform", clear_on_submit=True):
        submitted = st.form_submit_button("Submit")
        if submitted:
            with st.spinner("Getting the best info..."):
                response = agent({"input": query_text}, include_run_info = True)
                result = response["output"]
                run_id = response["__run"].run_id
    
    if result is not None:
        st.info(result)
        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")
        with col1:
            st.button("👍", args=(run_id, 1))
        with col2:
            st.button("👎", args=(run_id, 0))
    """   
    
    while True:
    # Get the user's query
        query = input()
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        print("\nThinking...\n")
        agent_response = agent.run(query)
        print(agent_response)

    

if __name__ == "__main__":
    main()
