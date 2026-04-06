from embedding import create_vector_db, create_keyword_retriever
from dataset_loader import titles
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document




class IngestionPipeline:
    def __init__(self, embed_model = 'Dqdung205/medical_vietnamese_embedding'):
        self.titles = titles 
        self.docs = [Document(page_content=t, metadata={"source": "dataset"}) for t in self.titles]
        self.embed_model = HuggingFaceEmbeddings(model_name = embed_model, model_kwargs={'trust_remote_code': True})


    def ingest(self):
        """Runs the full ingestion process."""

        if not titles:
            print("No titles found!")
            return None
        
    
        # Store to vector db
        vector_db = create_vector_db(doc=self.docs, embedd_model=self.embed_model)
        # Key word retriever
        keyword_retriever = create_keyword_retriever(doc=self.docs)
        # print(vector_db)

        return vector_db, keyword_retriever

if __name__ == '__main__':
    ingest = IngestionPipeline()
    print(ingest.ingest())