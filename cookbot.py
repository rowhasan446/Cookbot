import os
import base64
import asyncio
import warnings
import streamlit as st
from pypdf import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint

from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory

def set_background():
    with open("food.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings("ignore", category=UserWarning, 
    message="Tried to instantiate class 'path._path'")

DB_FAISS_PATH = "vectorstore/db_faiss"
UPLOAD_FOLDER = "Data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CUSTOM_PROMPT_TEMPLATE = """[INST]You are a professional chef. Given these ingredients: {question}, create a detailed recipe.
If ingredients are insufficient, say so. Never invent recipes.

Context: {context}

Format with:
1. Dish Name
2. Ingredients (with quantities)
3. Step-by-Step Instructions
4. Cooking Time
5. Serving Suggestions[/INST]"""

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def load_llm(huggingface_repo_id, HF_TOKEN):
    if not HF_TOKEN:
        raise ValueError("Missing HuggingFace Token!")
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.7,
        max_new_tokens=1024,
        top_p=0.95,
        do_sample=True,
        token=HF_TOKEN
    )

def extract_pdf_text(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

def extract_ingredients(response):
    ingredients_section = None
    if "Ingredients" in response:
        ingredients_section = response.split("Ingredients")[1].split("Step-by-Step Instructions")[0].strip()
    elif "2. Ingredients" in response:
        ingredients_section = response.split("2. Ingredients")[1].split("3. Step-by-Step Instructions")[0].strip()
    if ingredients_section:
        ingredients_list = [line.strip() for line in ingredients_section.split("\n") if line.strip()]
        return ingredients_list
    return []

def chat_interface():
    st.title("🍳 CookBot: AI Cooking Assistant")
    st.write("👋 Hi! I'm CookBot. List ingredients or ask cooking questions!")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    if 'ingredients' not in st.session_state:
        st.session_state.ingredients = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    user_input = st.chat_input("Your ingredients/question...")
    uploaded_pdf = st.file_uploader("📄 Upload recipe PDF", type="pdf")

    query = ""
    if uploaded_pdf:
        try:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_pdf.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.success(f"✅ File successfully uploaded to: {file_path}")
        except Exception as e:
            st.error(f"❌ PDF Upload Error: {str(e)}")

    if user_input:
        query += f"\n{user_input}"

    if query:
        try:
            st.session_state.messages.append({'role': 'user', 'content': query})
            with st.chat_message('user'):
                st.markdown(query)

            HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
            HF_TOKEN = os.environ.get("HF_TOKEN")
            
            with st.spinner("🍳 Cooking up your recipe... Please wait!"):
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    retriever=get_vectorstore().as_retriever(
                        search_type="mmr",
                        search_kwargs={'k': 3}
                    ),
                    memory=st.session_state.memory,
                    combine_docs_chain_kwargs={"prompt": set_custom_prompt()},
                    verbose=False
                )
                response = qa_chain.invoke({'question': query})
                clean_response = response['answer'].split("</s>")[0].strip()
            
            response_text = f"👩🍳 CookBot: {clean_response}"
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})
            with st.chat_message('assistant'):
                st.markdown(response_text)

            st.session_state.ingredients = extract_ingredients(clean_response)

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

    with st.sidebar:
        st.header("🧾 Ingredients Checklist")
        if st.session_state.ingredients:
            selected_ingredients = []
            for ingredient in st.session_state.ingredients:
                if st.checkbox(ingredient, key=ingredient):
                    selected_ingredients.append(ingredient)
            if selected_ingredients:
                if st.button("🛒 Order Selected Ingredients"):
                    st.success(f"Order placed for: {', '.join(selected_ingredients)}")
        else:
            st.write("No ingredients to display yet. Ask for a recipe!")

if __name__ == "__main__":
    chat_interface()