
import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # DOĞRU YOL
from datasets import load_dataset
from langchain_community.text_splitter import RecursiveCharacterTextSplitter # DOĞRU YOL

# ... (kodun geri kalanı... size en son verdiğim kodun tamamı) ...
# ... (kodu buraya kısaca ekliyorum, sizde zaten olmalı) ...

# --- 1. API Anahtarını Yükleme ---
try:
    from google.colab import userdata
    os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except ImportError:
    try:
        os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    except Exception as e:
        pass

# --- 2. RAG Zincirini Kurma Fonksiyonu ---
@st.cache_resource
def load_rag_chain():
    st.write("Veri seti yükleniyor...")
    dataset = load_dataset("ShubhamChoksi/IMDB_Movies", split="train")
    bolum_metni = " ".join(dataset['movie_review'][:100])

    st.write("Metin parçalanıyor...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    metin_parcalari = text_splitter.split_text(bolum_metni)

    st.write("Embedding modeli yükleniyor...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    st.write("Vektör veritabanı oluşturuluyor...")
    vector_store = FAISS.from_texts(metin_parcalari, embedding=embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro")

    template = """
    Sana verilen film eleştirisi metinlerini kullanarak kullanıcının sorusunu cevapla.
    Eğer metinlerde cevap yoksa, "Bu konuda bir bilgi bulamadım." de.
    Bağlam: {context}
    Soru: {question}
    Cevap:
    """
    prompt = PromptTemplate.from_template(template)

    st.write("RAG Zinciri hazır!")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- 3. Web Arayüzünü Oluşturma (Streamlit) ---
st.set_page_config(page_title="Film Eleştirmeni Chatbot", layout="wide")
st.title("🎬 IMDB Film Eleştirmeni Chatbot")
st.write("IMDB film eleştirilerine dayalı sorular sorun (İlk 100 eleştiriye göre).")

try:
    rag_chain = load_rag_chain()
except Exception as e:
    st.error(f"RAG zinciri yüklenirken bir hata oluştu: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Bir film eleştirisi hakkında soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Cevap aranıyor..."):
        try:
            cevap = rag_chain.invoke(prompt)
            st.session_state.messages.append({"role": "assistant", "content": cevap})
            with st.chat_message("assistant"):
                st.markdown(cevap)
        except Exception as e:
            st.error(f"Cevap alınırken bir hata oluştu: {e}")
