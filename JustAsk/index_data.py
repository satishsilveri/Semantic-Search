from nltk.tokenize import sent_tokenize
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import re
from embed_data import EmbedData

class IndexData:

    def __init__(self):
        self.embed_data_obj = EmbedData()

    def clean_text(self, text):
        # Remove escape characters
        cleaned_text = text.encode('ascii', 'ignore').decode()
        
        # Remove unwanted spaces
        cleaned_text = ' '.join(cleaned_text.split())
        
        cleaned_text = cleaned_text.lower()
        
        return cleaned_text

    def split_text_by_token_length(self, text: str = "", token_length: int = 400):
        '''
        '''
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            current_chunk.append(sentence)
            if sum(len(s.split()) for s in current_chunk) > token_length:
                chunks.append('. '.join(current_chunk[:-1]))
                current_chunk = [current_chunk[-1]]

        if current_chunk:
            chunks.append('. '.join(current_chunk))

        return chunks

    def split_text_by_token_length_with_sliding_window(self, text: str = "", token_length: int = 400):
        '''
        '''
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        window_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            current_chunk.append(sentence)
            window_size += len(sentence.split())

            while window_size > token_length:
                chunks.append('. '.join(current_chunk[:-1]))
                window_size -= len(current_chunk[0].split())
                current_chunk.pop(0)

        if current_chunk:
            chunks.append('. '.join(current_chunk))

        return chunks

    def index_text(self, documents, token_length : int = 400, overlap: bool = False, k :int = 5):
        '''
        '''

        # split documents
        splitted_documents = []

        for document in documents:
            if overlap:
                chunks = self.split_text_by_token_length_with_sliding_window(text = document.page_content, token_length=token_length)
            else:
                chunks = self.split_text_by_token_length(text = document.page_content, token_length=token_length)

            for chunk_idx, chunk in enumerate(chunks):
                temp_metadata = document.metadata
                temp_metadata['chunk_id'] = chunk_idx
                # superficial cleaning
                cleaned_chunk = self.clean_text(text = chunk)
                temp_doc = Document(page_content = cleaned_chunk, metadata=temp_metadata)
                splitted_documents.append(temp_doc)
        
        # initialize vector database and add docs
        text_db = Chroma.from_documents(docs = splitted_documents, collection_name="documents_collection", embedding_function=self.embed_data_obj.get_text_embedding())

        # convert to retriever
        retriever = text_db.as_retriever(search_type="similarity",search_kwargs={"k": k})

        return retriever


    def index_images(self, images):
        '''
        '''
        # generate image embeddings
        images_embeddings = []
        docs = []
        for img_obj in images:
            image_embeddings.append(self.embed_data_obj.get_image_embeddings(images = img_obj['image']))
            temp_doc = Document(page_content = "", metadata={'image_path':img_obj['image_path']})
            docs.append(temp_doc)

        # initialize image vector db and add embeddinghs
        images_db = Chroma.from_documents(docs=docs, embeddings=image_embeddings, embedding_function=self.embed_data_obj.get_query_embeddings_for_image(), collection_name="images_collection")

        return images_db