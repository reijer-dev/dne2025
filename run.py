from evaluation.rag_evaluater import RagEvaluater
from quick_dirty_rag import QuickDirtyRag
from void_rag import VoidRag

evaluater = RagEvaluater()
evaluater.add_system(QuickDirtyRag())
evaluater.add_system(VoidRag())

#evaluater.evaluate_training()

evaluater.evaluate_inference()