import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# ✅ Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# ✅ Check if HF_TOKEN is set correctly
if not HF_TOKEN:
    raise ValueError("Error: Hugging Face API token is missing. Set HF_TOKEN in your environment variables.")

# ✅ Define LLM Model
HUGGINGFACE_REPO_ID = "tiiuae/falcon-7b-instruct"  # Correct Model ID

def load_llm(huggingface_repo_id):
    # """Load Hugging Face LLM with API token."""
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,  # Correct token usage
        task="text-generation",  # Ensure task is specified
        model_kwargs={"max_length": 512}
    )

# ✅ Define Custom Prompt (Modified to request a recipe)
CUSTOM_PROMPT_TEMPLATE = """
You are a cooking assistant. Given the list of ingredients, you need to provide a detailed recipe.
If you don't know the recipe, say you don't know. Don't make up recipes.

Ingredients: {question}
Context: {context}

Provide the full recipe, including preparation steps, cooking time, and any important tips.
"""

def set_custom_prompt():
    """Returns a customized prompt template."""
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# ✅ Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if FAISS directory exists
if not os.path.exists(DB_FAISS_PATH):
    raise FileNotFoundError(f"Error: FAISS database not found at {DB_FAISS_PATH}. Please generate FAISS embeddings first.")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# ✅ Create Retrieval-Based QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt()}
)

# ✅ Run Cooking Bot 

while True:
    user_query = input("\n🍽️ Enter a list of ingredients (comma separated, or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        print("👋 Exiting Cooking Bot. Have a great day!")
        break
    
    print("\n🔍 Querying LLM for recipe...")
    response = qa_chain.invoke({"query": user_query})
    
    # ✅ Print result
    print("\n🍲 RECIPE: ", response.get("result", "No recipe found for these ingredients."))
    print("SOURCE DOCUMENTS: ", response["source_documents"])
