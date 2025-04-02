from evaluation.rag_system import RagSystem

class VoidRag(RagSystem):
    def __init__(self):
        super().__init__("VoidRag", "void_rag/work_dir")
        

    def process_document(self, path):
        """Embeds a single file and updates shared data."""
        pass
    
    def save(self):
        pass

    def query(self, question) -> str:
        pass