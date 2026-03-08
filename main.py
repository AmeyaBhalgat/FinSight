import os
import tempfile
import torch
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline

load_dotenv()

FAISS_INDEX_PATH  = "faiss_index"
LORA_ADAPTER_PATH = "lora_adapter"
BASE_MODEL        = "meta-llama/Llama-2-7b-hf"
BULLISH_THRESHOLD =  0.15
BEARISH_THRESHOLD = -0.15

CITATION_PROMPT = PromptTemplate(
    template="""You are a financial analyst assistant.
Answer using ONLY the context below. Cite every fact as [Source: filename, Page: N].
If context is insufficient, say "Insufficient information in provided documents."

Context: {summaries}
Question: {question}
Answer:""",
    input_variables=["summaries", "question"]
)


# ── Load models once ──────────────────────────────────────────────────────────
@st.cache_resource
def load_finbert():
    return pipeline(
        task="text-classification",
        model="ProsusAI/finbert",
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1
    )


@st.cache_resource
def load_llm():
    # If LoRA adapter exists, load fine-tuned model
    # If not, fall back to base model
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_ADAPTER_PATH if os.path.exists(LORA_ADAPTER_PATH) else BASE_MODEL
    )
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    if os.path.exists(LORA_ADAPTER_PATH):
        # Load and merge LoRA adapter into base model
        # Merging means W_final = W_base + (alpha/r) * B * A
        # After merging: zero inference overhead, same speed as base model
        model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
        model = model.merge_and_unload()
    else:
        model = base

    text_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True
    )

    return HuggingFacePipeline(pipeline=text_pipeline)


# ── Pipeline functions ────────────────────────────────────────────────────────
def build_index(sources: list):
    docs = []
    for src in sources:
        loader = PyPDFLoader(src) if src.endswith(".pdf") else WebBaseLoader(src)
        docs.extend(loader.load())

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""]
    ).split_documents(docs)

    embeddings  = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore, len(chunks)


def load_index() -> FAISS:
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        allow_dangerous_deserialization=True
    )


def get_answer(vectorstore: FAISS, question: str, llm):
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": CITATION_PROMPT}
    )
    result = chain.invoke({"question": question})
    return result.get("answer", ""), result.get("source_documents", [])


def predict_sentiment(finbert, chunks_with_scores: list) -> dict:
    total   = 0.0
    details = []

    for i, (doc, dist) in enumerate(chunks_with_scores):
        scores    = {x["label"]: x["score"] for x in finbert(doc.page_content[:1500])[0]}
        delta     = scores["positive"] - scores["negative"]
        relevance = 1 / (1 + dist)
        weighted  = delta * relevance
        total    += weighted

        details.append({
            "chunk":    i + 1,
            "source":   doc.metadata.get("source", "unknown"),
            "page":     doc.metadata.get("page", "?"),
            "preview":  doc.page_content[:120] + "...",
            "positive": round(scores["positive"], 3),
            "negative": round(scores["negative"], 3),
            "neutral":  round(scores["neutral"],  3),
            "delta":    round(delta,    3),
            "weight":   round(relevance, 3),
            "weighted": round(weighted,  3),
        })

    S = total / len(chunks_with_scores)

    if S > BULLISH_THRESHOLD:
        label, emoji = "BULLISH", "🟢"
    elif S < BEARISH_THRESHOLD:
        label, emoji = "BEARISH", "🔴"
    else:
        label, emoji = "NEUTRAL", "🟡"

    return {"label": label, "emoji": emoji, "score": round(S, 4), "details": details}


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="FinSight", page_icon="📈", layout="wide")
st.title("📈 FinSight — Financial News Intelligence")
st.caption("Upload financial PDFs or URLs → Ask questions → Cited answers + sentiment prediction")

finbert = load_finbert()
llm     = load_llm()

with st.sidebar:
    st.header("📂 Add Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    url_input      = st.text_area("Or paste URLs (one per line)")
    build_btn      = st.button("🔨 Build Knowledge Base", type="primary", use_container_width=True)
    st.divider()
    if os.path.exists(LORA_ADAPTER_PATH):
        st.success("🧠 LoRA fine-tuned model loaded")
    else:
        st.warning("⚠️ LoRA adapter not found — using base model")
    if os.path.exists(FAISS_INDEX_PATH):
        st.success("✅ Knowledge base ready")
    else:
        st.warning("⚠️ No knowledge base yet.")

if build_btn:
    sources = []
    for f in (uploaded_files or []):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(f.read())
        sources.append(tmp.name)
    if url_input.strip():
        sources.extend([u.strip() for u in url_input.splitlines() if u.strip()])

    if not sources:
        st.error("Please upload a PDF or enter a URL.")
    else:
        with st.spinner("Building knowledge base..."):
            vs, n = build_index(sources)
            st.session_state["vectorstore"] = vs
        st.success(f"✅ {n} chunks indexed.")
        st.rerun()

if "vectorstore" not in st.session_state and os.path.exists(FAISS_INDEX_PATH):
    st.session_state["vectorstore"] = load_index()

if "vectorstore" in st.session_state:
    st.divider()
    question = st.text_input("Ask a financial question", placeholder="What are the key revenue risks?")

    if st.button("🔍 Analyze", type="primary") and question:
        vs = st.session_state["vectorstore"]
        chunks_with_scores = vs.similarity_search_with_score(question, k=5)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("📝 Answer")
            with st.spinner("Generating answer..."):
                answer, sources = get_answer(vs, question, llm)
            st.markdown(answer)
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} chunks)"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**{i+1}.** `{doc.metadata.get('source','?')}` — Page {doc.metadata.get('page','?')}")
                        st.caption(doc.page_content[:300] + "...")
                        st.divider()

        with col2:
            st.subheader("📊 Sentiment")
            with st.spinner("Running FinBERT..."):
                result = predict_sentiment(finbert, chunks_with_scores)

            if result["label"] == "BULLISH":
                st.success(f"## {result['emoji']} {result['label']}")
            elif result["label"] == "BEARISH":
                st.error(f"## {result['emoji']} {result['label']}")
            else:
                st.warning(f"## {result['emoji']} {result['label']}")

            st.metric("Aggregate Score (S)", f"{result['score']:+.4f}")

            with st.expander("🔬 Per-Chunk Breakdown"):
                for c in result["details"]:
                    st.markdown(f"**Chunk {c['chunk']}** — `{c['source']}` p.{c['page']}")
                    st.caption(c["preview"])
                    a, b, d = st.columns(3)
                    a.metric("Positive", c["positive"])
                    b.metric("Negative", c["negative"])
                    d.metric("Neutral",  c["neutral"])
                    e, f_, g = st.columns(3)
                    e.metric("Delta",    f"{c['delta']:+}")
                    f_.metric("Weight",  c["weight"])
                    g.metric("Weighted", f"{c['weighted']:+}")
                    st.divider()
                st.info(f"S = Σ(delta × weight) / 5 = **{result['score']:+.4f}**")

else:
    st.info("👈 Upload documents in the sidebar to get started.")