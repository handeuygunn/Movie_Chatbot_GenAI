%%writefile app.py

import streamlit as st
import os
import google.generativeai as genai
from datasets import load_dataset

# --- v0.3 GÜNCELLEMELERİ ---
# Her modül artık kendi özel paketinden geliyor.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# --- GÜNCELLEMELER SONA ERDİ ---


# --- 1. API Anahtarını Yükleme ---
# Bu kısım değişmedi, Streamlit Cloud'un secrets özelliğini kullanacak.
try:
    os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except KeyError:
    st.error("Google API Anahtarı bulunamadı. Lütfen Streamlit Cloud Secrets'a ekleyin.")
    st.stop()
except Exception as e:
    st.error(f"API Anahtarı yüklenirken bir hata oluştu: {e}")
    st.stop()


# --- 2. RAG Zincirini Kurma Fonksiyonu ---
# @st.cache_resource, uygulamanın bu ağır işi sadece bir kez yapmasını sağlar.
@st.cache_resource
def load_rag_chain():
    st.write("Veri seti yükleniyor... (Bu işlem biraz zaman alabilir)")
    # 1. Veri Setini Yükle
    dataset = load_dataset("ShubhamChoksi/IMDB_Movies", split="train")
    bolum_metni = " ".join(dataset['movie_review'][:100])

    st.write("Metin parçalanıyor...")
    # 2. Metni Parçala (Chunking) - GÜNCELLENDİ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    metin_parcalari = text_splitter.split_text(bolum_metni)

    st.write("Embedding modeli yükleniyor... (Bu işlem de zaman alabilir)")
    # 3. Embedding Modelini Yükle - GÜNCELLENDİ
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    st.write("Vektör veritabanı oluşturuluyor...")
    # 4. Vektör Veritabanını Oluştur (FAISS) - GÜNCELLENDİ
    vector_store = FAISS.from_texts(metin_parcalari, embedding=embeddings)

    # 5. Retriever'ı Oluştur
    retriever = vector_store.as_retriever()

    # 6. LLM'i (Gemini) Tanımla - GÜNCELLENDİ
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro")

    # 7. Prompt Şablonunu Oluştur - GÜNCELLENDİ (langchain_core)
    template = """
    Sana verilen film eleştirisi metinlerini kullanarak kullanıcının sorusunu cevapla.
    Eğer metinlerde cevap yoksa, "Bu konuda bir bilgi bulamadım." de.
    Bağlam: {context}
    Soru: {question}
    Cevap:
    """
    prompt = PromptTemplate.from_template(template)

    st.write("RAG Zinciri hazır! Artık soru sorabilirsiniz.")
    # 8. RAG Zincirini Birleştir - GÜNCELLENDİ (langchain_core)
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
    st.error(f"RAG zinciri yüklenirken kritik bir hata oluştu: {e}")
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
