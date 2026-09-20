import os
import zipfile
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Assistente de Regras do Futebol",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ Assistente Oficial de Regras do Futebol")
st.caption("Consulte dúvidas sobre as regras de futebol respondidas diretamente com base no livro oficial.")

PASTA_BANCO = "./chroma_db"
ARQUIVO_ZIP = "chroma_db.zip"

# Carrega e inicializa o RAG apenas uma vez na memória do servidor
@st.cache_resource(show_spinner="Carregando base de regras e modelo de IA...")
def inicializar_rag():
    # 1. Descompacta o banco vetorial se ainda não foi extraído
    if not os.path.exists(PASTA_BANCO) and os.path.exists(ARQUIVO_ZIP):
        with zipfile.ZipFile(ARQUIVO_ZIP, "r") as zip_ref:
            zip_ref.extractall(PASTA_BANCO)

    # 2. Carrega embeddings locais
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 3. Conecta ao banco vetorial pré-computado
    vectorstore = Chroma(
        persist_directory=PASTA_BANCO,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 4. Obtém a chave da Groq (definida nos Secrets do Streamlit)
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        groq_api_key=groq_api_key
    )

    # 5. Cria a cadeia RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Inicializa o pipeline
qa_chain = inicializar_rag()

# Gerencia o histórico de mensagens da sessão
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Pergunte-me qualquer dúvida sobre as regras do futebol."}
    ]

# Renderiza as mensagens anteriores na tela
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Campo de entrada de nova pergunta
if prompt := st.chat_input("Digite sua dúvida (ex: Quando é marcado impedimento?)"):
    # Exibe a pergunta do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera a resposta via RAG
    with st.chat_message("assistant"):
        with st.spinner("Consultando as regras..."):
            resultado = qa_chain.invoke({"query": prompt})
            resposta_texto = resultado["result"]

            # Formata as fontes consultadas
            trechos = resultado.get("source_documents", [])
            fontes_texto = ""
            if trechos:
                fontes_texto = "\n\n---\n**Fontes consultadas:**\n"
                for i, doc in enumerate(trechos, start=1):
                    pag = doc.metadata.get("page", "N/A")
                    pag_humana = int(pag) + 1 if str(pag).isdigit() else pag
                    fontes_texto += f"- *Trecho {i}* — Página {pag_humana}\n"

            resposta_completa = resposta_texto + fontes_texto
            st.markdown(resposta_completa)

    st.session_state.messages.append({"role": "assistant", "content": resposta_completa})