from transformers import pipeline
from process_document import ProcessDocument
from index_data import IndexData
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
import os

class SearchPipeline:

    def __init__(self, url_or_path):

        self.url_or_path = url_or_path

        process_doc_obj = ProcessDocument(url_or_path=url_or_path)

        self.documents, self.images, self.tables = process_doc_obj.extract_data()

        index_data_obj = IndexData()

        self.retriever = index_data_obj.index_text(self.documents, token_length = 400, overlap = True, k = 5)

        self.image_db = index_data_obj.index_images(self.images)

        self.tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")

        self.llm_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        

    def search(self, query : str =""):
        
        #------------------------------------------------------------
        # get llm response
        #------------------------------------------------------------
        text_generation_pipeline = transformers.pipeline(
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            task="text-generation",
            temperature=0,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=300,
        )

        prompt_template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """

        mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        ##### Create prompt from prompt template 
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        ##### Create llm chain 
        llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

        rag_chain = ( 
        {"context": self.retriever, "question": RunnablePassthrough()}
            | llm_chain
        )

        llm_response = rag_chain.invoke(query)

        #------------------------------------------------------------
        # get image response
        #------------------------------------------------------------
        images_docs = self.images_db.similarity_search(query)

        #------------------------------------------------------------
        # get table response
        #------------------------------------------------------------
        file_name = os.path.basename(self.url_or_path)
        table_directory = '{}/{}'.format(os.get_cwd(), file_name, 'tables')

        # extract page numbers from llm_response
        pages = []
        for doc in llm_response['context']:
            pages.append(doc.metadata['page'])

        dfs = []
        for page_num in pages:
            table_directory = '{}/page_{}'.format(table_directory, str(page_num))
            for filename in os.listdir(table_directory):
                if filename.endswith(".csv"):
                    # Construct the full file path
                    file_path = os.path.join(directory, filename)
                    
                    dfs.append(pd.read_csv(file_path))

        table_responses = []
        for df in dfs:
            temp={}
            temp['answer'] = tqa(table=df.astype(str), query="What is the engine of spirit of america car?")['cells'][0]
            temp['table'] = df
            table_responses.append(temp)

        return llm_response, images_docs, table_responses