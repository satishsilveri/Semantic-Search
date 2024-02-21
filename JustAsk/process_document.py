import validators
import fitz
from langchain_community.document_loaders import PyMuPDFLoader, OnlinePDFLoader
import os
from urllib.parse import urlparse
import tabula

class ProcessDocument:

    def __init__(self, url_or_path):
        self.url_or_path  = url_or_path

    def extract_data(self):

        documents = None
        images = []
        file_name = ""
    
        if validators.url(self.url_or_path):
            file_name = os.path.basename(url_parse_obj.path)
            loader = OnlinePDFLoader(self.url_or_path)
            url_parse_obj = urlparse(url)
            documents = loader.load()
        else:
            #------------------------------------------------------------------------
            # Extract documents
            #------------------------------------------------------------------------

            file_name = os.path.basename(self.url_or_path)
            loader = PyMuPDFLoader(self.url_or_path)
            documents = loader.load()

            #------------------------------------------------------------------------
            # Extract images
            #------------------------------------------------------------------------
            doc = fitz.open(self.url_or_path) # open a document

            total_pages = len(doc)

            for page_index in range(len(doc)): # iterate over pdf pages
                page = doc[page_index] # get the page
                image_list = page.get_images()
                image_base_path = os.path.join(os.getcwd(), file_name, 'images')
                if not os.path.exists(image_base_path):
                    os.makedirs(image_base_path)

                for image_index, img in enumerate(image_list, start=1): # enumerate the image list
                    xref = img[0] # get the XREF of the image
                    pix = fitz.Pixmap(doc, xref) # create a Pixmap

                    if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    image_path = "{}/{}/images/page_{}-image_{}.png".format(os.getcwd(),file_name, page_index, image_index)

                    if not os.path.exists(image_path):
                        pix.save(image_path) # save the image as png
                    
                    temp = {}
                    temp['image'] = pix
                    temp['image_path'] = image_path
                    images.append(temp)
                    pix = None
            
            #------------------------------------------------------------------------
            # Extract tables
            #------------------------------------------------------------------------

            table_base_path = os.path.join(os.getcwd(), file_name, 'tables')
            if not os.path.exists(table_base_path):
                os.makedirs(table_base_path)

            for page_num in range(total_pages):

                page_num = page_num + 1
                
                tables = tabula.read_pdf(pdf_path, pages=str(page_num), multiple_tables=True)
            
                for idx, table in enumerate(tables):
                    os.makedirs("{}/{}/tables/page_{}".format(os.getcwd(),file_name, page_num))
                    table_path = "{}/{}/tables/page_{}/table_{}.csv".format(os.getcwd(),file_name, page_num, str(idx+1))
                    table.to_csv(table_path, index = False)

        return documents, images
