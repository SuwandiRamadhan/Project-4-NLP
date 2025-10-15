import streamlit as st
import json
import ollama
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import time

# validasi dataset
if not os.path.exists("dataset_baru.csv"):
    raise FileNotFoundError(f"Dataset tidak ditemukan di dataset_baru.csv")

@st.cache_resource(show_spinner=False)
def load_models_and_data():
    try:
        try:
            llm_agent = ollama.Client(host="http://localhost:11434")
        except Exception as e:
            st.error(f"Gagal terhubung ke Ollama server: {str(e)}")
            raise
        try:
            embedder = SentenceTransformer('BAAI/bge-m3')
        except Exception as e:
            st.error(f"Gagal memuat model embedding: {str(e)}")
            raise
        
        # muat dataset
        try:
            df_recipes = pd.read_csv("dataset_baru.csv")
        except Exception as e:
            st.error(f"Gagal memuat dataset: {str(e)}")
            raise
        
        # gabung Title dan Ingredients untuk embedding
        df_recipes['text_for_embedding'] = df_recipes['Title Cleaned'] + " " + df_recipes['Ingredients Cleaned']
        
        # cek cache embeddings
        if os.path.exists("embeddings_cache.pkl"):
            with open("embeddings_cache.pkl", 'rb') as f:
                corpus_embeddings = pickle.load(f)
            st.success("Berhasil memuat embedding dari cache.")
        else:
            # generasi embedding
            progress_text = "Menggenerasi embedding untuk resep..."
            my_bar = st.progress(0, text=progress_text)
            
            total_recipes = len(df_recipes)
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, total_recipes, batch_size):
                batch = df_recipes['text_for_embedding'].iloc[i:i+batch_size].tolist()
                batch_embeddings = embedder.encode(batch, convert_to_tensor=True)
                all_embeddings.append(batch_embeddings)
                progress = (i + len(batch)) / total_recipes
                my_bar.progress(progress, text=f"{progress_text} ({i + len(batch)}/{total_recipes})")
            
            corpus_embeddings = np.vstack([e.cpu().numpy() for e in all_embeddings])
            
            # simpan embeddings ke cache
            with open("embeddings_cache.pkl", 'wb') as f:
                pickle.dump(corpus_embeddings, f)
            
            my_bar.empty()
            st.success("Embedding resep selesai dan disimpan ke cache.")
        print("Model dan data lokal berhasil dimuat.")
        return llm_agent, embedder, df_recipes, corpus_embeddings
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model dan data: {str(e)}")
        raise

if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "llm_agent" not in st.session_state:
    st.session_state.llm_agent = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "df_recipes" not in st.session_state:
    st.session_state.df_recipes = None
if "corpus_embeddings" not in st.session_state:
    st.session_state.corpus_embeddings = None

def search_document_local(query, k_top=3):
    if not query.strip():
        return []
    if len(query) > 200:
        query = query[:200]
    try:
        # ambil model dan data dari session state
        embedder = st.session_state.embedder
        df_recipes = st.session_state.df_recipes
        corpus_embeddings = st.session_state.corpus_embeddings
        
        # embedding untuk query user
        query_embedding = embedder.encode(query, convert_to_tensor=False)
        
        # convert embedding ke numpy array
        corpus_embeddings_np = corpus_embeddings
        if hasattr(corpus_embeddings, 'cpu'):
            corpus_embeddings_np = corpus_embeddings.cpu().numpy()
            
        # hitung kesamaan kosinus
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            corpus_embeddings_np
        ).flatten()
        
        # indeks dari k_top resep paling relevan
        top_indices = np.argsort(similarities)[::-1][:k_top]
        
        # Gunakan list comprehension untuk efisiensi
        results = [{
            'text': df_recipes.iloc[idx]['text_for_embedding'],
            'original_title': df_recipes.iloc[idx]['Title Cleaned'],
            'distance': 1 - similarities[idx],
            'ingredients': df_recipes.iloc[idx].get('Ingredients Cleaned', '')
        } for idx in top_indices]
        return results
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mencari resep: {str(e)}")
        return []

def response_query(query):
    try:
        # validasi input
        if not query.strip():
            return "Mohon masukkan pertanyaan Anda tentang resep masakan Indonesia."
            
        # cari dokumen yang relevan
        retrieved_docs = search_document_local(query)
        if not retrieved_docs:
            return "Maaf, saya tidak dapat menemukan resep yang sesuai dengan permintaan Anda di dalam data resep lokal saya."
            
        # format konteks dengan informasi
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(
                f"Resep: {doc['original_title']}\n"
                f"Bahan-bahan: {doc['ingredients']}\n"
                f"Detail: {doc['text']}"
            )
        context = "\n\n".join(context_parts)

        prompt_template = """
    [role]
    Bertindaklah sebagai "Chef AI", seorang asisten koki virtual yang ahli dalam masakan Indonesia. Anda ramah, membantu, dan selalu memberikan jawaban berdasarkan data resep yang Anda miliki.

    [task]
    Jawab pertanyaan pengguna secara akurat berdasarkan konteks resep yang relevan yang disediakan di bawah ini. Fokuslah untuk memberikan jawaban yang paling membantu sesuai dengan apa yang ditanyakan pengguna.

    Pertanyaan Pengguna: {query}

    [context]
    Anda hanya boleh menggunakan informasi dari resep-resep berikut untuk menyusun jawaban Anda. Jangan membuat atau menambahkan informasi (seperti bahan atau langkah) yang tidak ada dalam teks di bawah ini.

    Konteks Resep:
    {context}

    [reasoning]
    1.  Baca dan pahami pertanyaan pengguna untuk mengidentifikasi apa yang sebenarnya mereka butuhkan (misalnya: daftar bahan, langkah-langkah memasak, atau hanya mencari resep).
    2.  Pindai konteks resep yang diberikan untuk menemukan informasi yang paling relevan dengan pertanyaan pengguna.
    3.  Strukturkan jawaban Anda sesuai dengan permintaan.
    4.  Jika pengguna bertanya tentang bahan, berikan daftar bahan dari resep yang paling relevan.
    5.  Jika pengguna bertanya cara memasak, berikan langkah-langkah memasak secara detail dari resep yang paling relevan. Jika tidak ditanya cara memasak, jangan berikan langkah-langkahnya.
    6.  Pastikan semua informasi dalam jawaban Anda berasal langsung dari [context] yang telah disediakan.
    7. Jika ditanya langkah memasak, berikan langkah-langkah yang jelas dan terstruktur.

    [output format]
    Sajikan jawaban Anda dalam format yang jelas dan mudah dibaca. Gunakan poin-poin bernomor atau bullet points untuk daftar bahan atau langkah-langkah. Gunakan paragraf singkat untuk penjelasan. Awali jawaban Anda dengan menyebutkan nama resep yang Anda rujuk.

    Contoh Format:
    "Berdasarkan resep **[Nama Resep]**, berikut adalah bahan-bahan yang Anda butuhkan:"
    - Bahan 1
    - Bahan 2
    
    "Langkah-langkah memasak:"
    1. Langkah pertama
    2. Langkah kedua
    
    (jika tidak ditanya cara memasak, jangan sertakan bagian langkah-langkah, hanya berikan bahan, nah di dalam dataset yang ada itu tidak ada langkah memasaknya, jadi anda harus improvisasi sendiri langkah memasaknya berdasarkan pengetahuan umum anda tentang masakan indonesia, tapi jangan sampai keluar dari konteks bahan yang ada)

    [stop condition]
    Jawaban dianggap selesai ketika pertanyaan pengguna telah dijawab sepenuhnya menggunakan informasi dari konteks yang diberikan, sesuai dengan format yang diminta, dan tidak ada informasi tambahan di luar konteks yang disertakan.
    """

        prompt = prompt_template.format(context=context, query=query)
        llm_agent = st.session_state.llm_agent
        response = llm_agent.chat(model="gemma3:4b", messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        return response['message']['content']
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}")
        return "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi."

if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "llm_agent" not in st.session_state:
    st.session_state.llm_agent = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "df_recipes" not in st.session_state:
    st.session_state.df_recipes = None
if "corpus_embeddings" not in st.session_state:
    st.session_state.corpus_embeddings = None

# UI Streamlit
st.set_page_config(
    page_title="Chef AI - Asisten Resep Masakan Indonesia",
    page_icon="🗨️",
    layout="wide"
)
st.title("🗨️ Chef AI - Asisten Resep Masakan Indonesia")
st.caption("Dibangun dengan Arsitektur RAG (Retrieval Augmented Generation)")
st.caption("Chatbot ini menggunakan data resep lokal dan model LLM Gemma3 dari Ollama.")

# muat model dan data
loading_placeholder = st.empty()
try:
    if not st.session_state.model_loaded:
        with loading_placeholder.container():
            with st.spinner(""):
                st.info("⌛ Sedang memuat model dan data... Mohon tunggu sebentar.")
                progress = st.progress(0)
                llm_agent, embedder, df_recipes, corpus_embeddings = load_models_and_data()
                progress.progress(100)
                
                # Simpan ke session state
                st.session_state.llm_agent = llm_agent
                st.session_state.embedder = embedder
                st.session_state.df_recipes = df_recipes
                st.session_state.corpus_embeddings = corpus_embeddings
                st.session_state.model_loaded = True
                
                # Tampilkan pesan sukses
                st.success("✅ Model dan data berhasil dimuat!")
                time.sleep(1)
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
    st.stop()

if st.session_state.model_loaded:
    loading_placeholder.empty()

# sidebar
with st.sidebar:
    st.header("💡 Panduan Penggunaan")
    st.write("""
    1. Tanyakan tentang resep masakan Indonesia
    2. Anda bisa menanyakan bahan-bahan atau cara memasak
    3. Bot akan memberikan jawaban berdasarkan resep yang tersedia
    
    Contoh pertanyaan:
    - "Bagaimana cara membuat rendang?"
    - "Apa saja bahan untuk membuat nasi goreng?"
    - "Resep ayam goreng kalasan"
    """)

# batasi jumlah pesan dalam riwayat chat
if len(st.session_state.messages) > 50:
    st.session_state.messages = st.session_state.messages[-50:]

# tampilkan pesan-pesan dari riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# terima input dari user
if prompt := st.chat_input("Mau masak apa hari ini?"):
    # Validasi input
    if len(prompt.strip()) == 0:
        st.warning("Mohon masukkan pertanyaan Anda.")
    else:
        # tambahkan pesan user ke riwayat chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # tampilkan pesan user
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # tampilkan respons dari bot
        with st.chat_message("assistant"):
            with st.spinner("Sedang mencari resep yang sesuai..."):
                try:
                    # ambil resep dan respons
                    retrieved_docs = search_document_local(prompt)
                    if retrieved_docs:
                        # ambil dan tampilkan respons
                        response = response_query(query=prompt)
                        st.markdown(response)
                    else:
                        st.warning("Maaf, tidak ditemukan resep yang sesuai dengan pencarian Anda.")
                        response = "Maaf, saya tidak dapat menemukan resep yang sesuai dengan permintaan Anda."
                except Exception as e:
                    st.error(f"Maaf, terjadi kesalahan: {str(e)}")
                    response = "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi."
            
        # tambah respons ke riwayat chat
        st.session_state.messages.append({"role": "assistant", "content": response})