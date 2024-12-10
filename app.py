from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import streamlit as st

api_key = st.secrets["OPENAI_API_KEY"]

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

def generate_response(index_path):
    """Loads the FAISS index and retrieves content for the given query."""
    # Initialize embeddings and load the FAISS index
    embeddings = HuggingFaceEmbeddings()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, api_key=api_key)
    vector = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    db = vector.as_retriever()
    template = """You are Question-answering bot specialized in answering user Queries. You possess an thorough understanding of the index provided and gives the most
                accurate response possible.Your response should be contextual.
    <context>
    {context}
    </context>

    Question: {input}
    """

    prompt = ChatPromptTemplate.from_template(template=template)
    # Create a chain
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(db, doc_chain)
    return chain

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
    
            index_path = r'C:\Users\aipro\Documents\Muscat\output_index'
            
            chain = generate_response(index_path)
            if query:
                with st.expander("Response"):
                    response = chain.invoke({"input": query})
                    out = response['answer']
                    print(out)
                    st.write(out)
                    if 'context' in response:
                        sources = [doc.metadata['source'] for doc in response['context']]
                        print("Sources:", sources[0])
                        st.write("Sources:", sources[0])
                    else:
                        print("No sources found.")

if __name__ == "__main__":
    main()
