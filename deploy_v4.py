import gradio as gr
import ollama
from ollama import chat, ChatResponse
from pymilvus import MilvusClient
from tqdm import tqdm
from pypdf import PdfReader
import os


# ---------- 1️⃣ PDF loading and text chunking ----------
def load_pdf_chunks(pdf_path, chunk_size=500, overlap=50):
    reader = PdfReader(pdf_path,strict=True)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    # Split text into overlapping chunks
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ---------- 2️⃣ Embedding helper ----------
def emb_text(text):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
    return response["embedding"]

# ---------- 3️⃣ Globals ----------
milvus_path = "./milvus_demo.db"
collection_name = "my_rag_collection"
milvus_client = None
#my first line with nvim

# ---------- 4️⃣ Build Milvus DB from uploaded PDF ----------
def build_database_from_pdf(pdf_file):
    global milvus_client

    if pdf_file is None:
        return "⚠️ Please upload a PDF file first."

    pdf_path = pdf_file.name
    print(f"Processing file: {pdf_path}")

    # Connect to Milvus Lite (local)
    milvus_client = MilvusClient(uri=milvus_path)

    # Drop previous collection (optional)
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    chunks = load_pdf_chunks(pdf_path)
    print(f"Loaded {len(chunks)} text chunks from PDF.")

    # Create collection
    test_embedding = emb_text("This is a test")
    embedding_dim = len(test_embedding)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",
    )

    # Insert embeddings
    data = []
    for i, chunk in enumerate(tqdm(chunks, desc="Creating embeddings")):
        data.append({"id": i, "vector": emb_text(chunk), "text": chunk})
    milvus_client.insert(collection_name=collection_name, data=data)

    return f"✅ base de données crée avec {len(chunks)} chunks, fichier utilisé: {os.path.basename(pdf_path)}."


# ---------- 5️⃣ RAG query with context display ----------
def answer_question(question: str):
    global milvus_client

    if milvus_client is None:
        return "⚠️ Please upload and process a PDF first.", ""

    # Search for top similar chunks
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(question)],
        limit=3,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    retrieved_chunks = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]

    # Combine retrieved text for context
    context = "\n\n".join(
        [f"Chunk {i+1} (score={dist:.3f}):\n{text}" for i, (text, dist) in enumerate(retrieved_chunks)]
    )

    SYSTEM_PROMPT = "You are an AI assistant. Use the provided context to answer clearly and concisely."
    USER_PROMPT = f"""
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """

    response: ChatResponse = chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )

    return response["message"]["content"], context


# ---------- 6️⃣ Gradio Interface ----------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    #gr.Markdown("# 💻 RAG DEMO NEXTGENPC 💻")
    gr.Markdown(
        """
        <h1 style="text-align: center;">💻 RAG DEMO NEXTGENPC 💻</h1>
        """,
        elem_id="title"
    )
    gr.Markdown("Chargez vos pdfs pour créer votre RAG")

    with gr.Row():
        pdf_input = gr.File(label="📂 Upload PDF", file_types=[".pdf"])
        build_btn = gr.Button("⚙️ Construction base de données")
    build_status = gr.Textbox(label="Database build status:", interactive=False)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Posez votre question",
                placeholder="Example: What is the main topic discussed?",
                lines=2,
            )
            ask_btn = gr.Button("Ask")
            answer_output = gr.Textbox(label="💬 Réponse modèle", lines=8)
        with gr.Column(scale=1):
            context_output = gr.Textbox(label="🔎 Retrieved context", lines=20, interactive=False)

    # Wiring
    build_btn.click(fn=build_database_from_pdf, inputs=pdf_input, outputs=build_status)
    ask_btn.click(fn=answer_question, inputs=question_input, outputs=[answer_output, context_output])

# ---------- 7️⃣ Launch ----------
if __name__ == "__main__":
    demo.launch(share=True)

