from deep_translator import GoogleTranslator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import requests
import streamlit as st

st.set_page_config(page_icon="😶‍🌫️", page_title="QA", layout="wide")

header = st.container()
header.title("Query Assistance Genie")
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        .st-emotion-cache-vj1c9o {
            background-color: rgb(242, 242, 242, 0.68);
        }
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            background-color: rgb(242, 242, 242, 0.68);
            z-index: 999;
            text-align: center;
        }
        .fixed-header {
            border-bottom: 0;
        }
      div[data-testid="stVerticalBlock"] div:has(div.fixed-header) .st-emotion-cache-1wmy9hl {
            display: flex;
            flex-direction: column;
            margin-top: -70px;
        }
       
        h1 {
        font-family: "Source Sans Pro", sans-serif;
        font-weight: 500;
        color: rgb(49, 51, 63);
        padding: 1.25rem 0px 1rem;
        margin: 0px;
        line-height: 1.2;
        color: black;
        border:1px solid black;
    }
   
    .st-emotion-cache-jkfxgf p {
        word-break: break-word;
        margin-bottom: 0px;
        font-size: 16px;
        font-weight: 600;
        color : purple;
    }
    .st-emotion-cache-1puwf6r p {
    word-break: break-word;
    margin-bottom: 0px;
    font-size: 14px;
    font-weight: 600;
    }
    .st-b6 {
    border-bottom-color: black;
    }
    .st-b5 {
        border-top-color: black;
    }
    .st-b4 {
        border-right-color: black;
    }
    .st-b3 {
        border-left-color: black;
    }
    .st-emotion-cache-1igbibe{
        border: 1px solid black;
    }

   

 
       </style>
       """,
       unsafe_allow_html=True
)
 
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def generate_response(index_path, query):
    """Loads the FAISS index and retrieves content for the given query."""
    try:
        # Initialize embeddings and load the FAISS index
        embeddings = HuggingFaceEmbeddings()
        search = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        
        # Use the retriever to get relevant documents
        retriever = search.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Limit to top 5 docs
        relevant_docs = retriever.get_relevant_documents(query)  # Use appropriate method
        
        # Combine content from relevant documents
        if relevant_docs:
            content = "\n".join([doc.page_content for doc in relevant_docs])
            return content
        else:
            return None

    except Exception as e:
        print(f"Error generating response from FAISS index: {e}")
        return None
    
def set_query(query):
        st.session_state.query_input = query
        st.rerun()


def main():
        # Initialize session state for the query if it doesn't exist
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    # The text input field, populated with the query stored in session state
    query = st.text_input("Enter your query:", value=st.session_state.query_input)

    st.write("Suggested Queries:")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("What is the maximum percentage of an FLC's net worth that can be invested in corporate bonds, according to Circular FM41?"):
            set_query("What is the maximum percentage of an FLC's net worth that can be invested in corporate bonds, according to Circular FM41?")
    with col2:
            
        if st.button("Which banks are subject to the new stress testing guidelines issued by the Central Bank of Oman?"):
            set_query("Which banks are subject to the new stress testing guidelines issued by the Central Bank of Oman?")

    with col3:
                
        if st.button("What should the macro stress testing framework project for a bank under baseline and stressed scenarios?"):
            set_query("What should the macro stress testing framework project for a bank under baseline and stressed scenarios?")

    with col4:

        if st.button("What is the total storage capacity available in the Oracle Server X9-2 when configured with NVMe SSDs?"):
            set_query("What is the total storage capacity available in the Oracle Server X9-2 when configured with NVMe SSDs?")
    
    if query:
        with st.spinner("Getting Response"):
     
    
            index_path = ""


                
            # Retrieve content based on the query
            content = generate_response(index_path, query)
            if not content:
                print("No relevant content found.")

            # Limit content length for prompt
            content_snippet = content[:1500] 
            
            # Construct the prompt
            prompt_input = (
                f"You are a Question-Answering bot specialized in answering user queries based on a given index. "
                f"Your responses should be accurate and contextual.\n"
                f"User Query: {query}\nContent: {content_snippet}"
            )

            # Send the request to the external API
            try:
                response = requests.post(
                    "http://192.168.25.131:8008/generate",
                    json={"prompt": prompt_input, "model": "mistral:latest"}
                )
                
                # Check for successful response
                if response.status_code == 200:
                    generated_response = response.json().get('response', "No response provided by the model.")
                    print("Generated Response:", generated_response.strip())
                    with st.expander("Response"):
                        st.write("Generated Response:", generated_response.strip())
                else:
                    print(f"API Request failed with status code: {response.status_code}")
                    print("Error:", response.text)
                    st.write("Error:", response.text)

            except Exception as e:
                print(f"Error sending request to the API: {e}")


if __name__ == "__main__":
    main()
