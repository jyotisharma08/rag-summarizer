import fitz
import numpy as np
import faiss

class RAGSummarizer:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.embed_model = None
        self.summarizer = None

    def load_models(self):
        if self.embed_model is None or self.summarizer is None:
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline

            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6"
            )

    def extract_text(self, file):
        if file.type == "application/pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text("text")
            return text
        else:
            return file.read().decode("utf-8")

    def chunk_text(self, text, chunk_size=100):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def build_index(self, text):
        self.load_models()  # ✅ load here safely

        self.chunks = self.chunk_text(text)

        if len(self.chunks) == 0:
            raise ValueError("No text found!")

        self.embeddings = self.embed_model.encode(self.chunks)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

    def get_key_chunks(self, top_k=3):
        if len(self.chunks) == 0:
            return []

        top_k = min(top_k, len(self.chunks))

        centroid = np.mean(self.embeddings, axis=0)

        distances = np.linalg.norm(self.embeddings - centroid, axis=1)
        indices = np.argsort(distances)[:top_k]

        return [self.chunks[i] for i in indices]

    def summarize(self, max_len=120, min_len=40):
        key_chunks = self.get_key_chunks()

        if len(key_chunks) == 0:
            return "No content"

        combined_text = " ".join(key_chunks)

        summary = self.summarizer(
            combined_text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )

        return summary[0]['summary_text']
