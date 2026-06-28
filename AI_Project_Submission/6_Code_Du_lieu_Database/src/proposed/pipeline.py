from .hybrid_search import ProposedHybridRetriever
from .llm import BaseGenerator


class ProposedHybridRAGPipeline:
    def __init__(self, lora_path: str = None):
        self.retriever = ProposedHybridRetriever()
        self.generator = BaseGenerator(lora_path=lora_path, is_unsloth=True)

    def run(
        self,
        query: str,
        options: dict = None,
        dataset_type: str = "medqa",
        context: str = None,
        retriever_mode: str = "hybrid",
    ) -> str:
        if dataset_type == "medqa":
            options_text = "\n".join([f"{k}) {v}" for k, v in (options or {}).items()])
            full_query = f"{query}\n{options_text}"
            top_context_docs = self.retriever.retrieve(
                full_query, top_k=3, mode=retriever_mode
            )
        elif dataset_type == "pubmedqa":
            # For pubmedqa we might already have the built-in context, but if we need to retrieve:
            full_query = query
            if context and retriever_mode != "none":
                top_context_docs = [{"text": context}]
            elif retriever_mode == "none":
                top_context_docs = []
            else:
                top_context_docs = self.retriever.retrieve(
                    full_query, top_k=3, mode=retriever_mode
                )
        else:
            full_query = query
            top_context_docs = self.retriever.retrieve(
                full_query, top_k=3, mode=retriever_mode
            )

        answer, throughput = self.generator.generate(
            top_context_docs, query, options, dataset_type=dataset_type
        )
        return answer, throughput
