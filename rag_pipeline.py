import fitz  # PDF reader (PyMuPDF)
import numpy as np
import faiss

class RAGSummarizer:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.embed_model = None
        self.summarizer = None

    # Load models only when needed
    def load_models(self):
        if self.embed_model is None or self.summarizer is None:
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline

            # Embedding model
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Generation model (FLAN-T5)
            self.summarizer = pipeline(
                "text2text-generation",
                model="google/flan-t5-base"
            )

    # Extract text from PDF or TXT
    def extract_text(self, file):
        if file.type == "application/pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text("text")
            return text.strip()
        else:
            return file.read().decode("utf-8")

    # Split text into chunks
    def chunk_text(self, text, chunk_size=100):
        words = text.split()
        return [
            " ".join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

    # Build FAISS index
    def build_index(self, text):
        self.load_models()

        self.chunks = self.chunk_text(text)

        if len(self.chunks) == 0:
            raise ValueError("No text found!")

        self.embeddings = self.embed_model.encode(self.chunks)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

    # 🔥 REAL RAG RETRIEVAL
    def retrieve(self, query, top_k=3):
        query_embedding = self.embed_model.encode([query])

        distances, indices = self.index.search(
            np.array(query_embedding), top_k
        )

        return [self.chunks[i] for i in indices[0]]

    # Generate answer using retrieved chunks
    def summarize(self, query, max_len=120, min_len=40):
        retrieved_chunks = self.retrieve(query)

        if len(retrieved_chunks) == 0:
            return "⚠️ No relevant content found."

        context = " ".join(retrieved_chunks)

        prompt = f"""
        Answer the question based on the context below:

        Context:
        {context}

        Question:
        {query}
        """

        summary = self.summarizer(
            prompt,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )

        return summary[0]['generated_text']
