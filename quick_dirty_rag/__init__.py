import os
import json
import ollama
import numpy as np
import time
import threading
from evaluation.rag_system import RagSystem

class QuickDirtyRag(RagSystem):
    def __init__(self):
        super().__init__("QuickDirtyRag", "quick_dirty_rag/work_dir")
        self.documents = []
        self.embeddings = []
        
        if os.path.isfile(f"{self.base_directory}/qdr-embeddings.json"):
            with open(f"{self.base_directory}/qdr-embeddings.json", "r") as f:
                data = json.load(f)
                self.documents = data["documents"]
                self.embeddings = data["embeddings"]

    def process_document(self, path):
        """Embeds a single file and updates shared data."""
        with open(path, "r") as f:
            document = f.read()
            try:
                embedding = ollama.embeddings(model='nomic-embed-text', prompt=document)['embedding']
                self.documents.append(document)
                self.embeddings.append(embedding)
            except ollama.ResponseError as e:
                print(f"Ollama embedding error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    
    def save(self):
        data = {"documents": self.documents, "embeddings": self.embeddings}

        with open(f"{self.base_directory}/qdr-embeddings.json", "w") as f:
            json.dump(data, f)

    def query(self, question) -> str:
        # (query function remains the same)
        try:
            question_embedding = ollama.embeddings(model='nomic-embed-text', prompt=question)['embedding']
        except ollama.ResponseError as e:
            print(f"Ollama embedding error: {e}")
            return

        question_embedding_np = np.array(question_embedding)
        embeddings_np = np.array(self.embeddings)

        similarities = np.dot(embeddings_np, question_embedding_np) / (np.linalg.norm(embeddings_np, axis=1) * np.linalg.norm(question_embedding_np))

        top_indices = np.argsort(similarities)[::-1][:10]

        context_documents = [self.documents[i] for i in top_indices]

        prompt = f"Beantwoord de volgende vraag kort en duidelijk, gebaseerd op de meegestuurde context.:\n\nVraag: {question}\n\nContext:\n"
        for doc in context_documents:
            prompt += f"- {doc}\n\n"

        try:
            response = ollama.chat(model='gemma2:2b', messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            return response['message']['content']
        except ollama.ResponseError as e:
            print(f"Ollama chat error: {e}")