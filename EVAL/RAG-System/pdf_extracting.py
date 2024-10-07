# pip install PyMuPDF Pillow

from subprocess import CalledProcessError
import fitz  # PyMuPDF
import io, os, easyocr
import tabula
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader

def extract_pdf(document_path):

    # Clearing the text file after each use
    text_file= open("output.txt","w+")
    text_file.write("")
    text_file.close()

    # Output directory for the extracted images
    output_dir = "/home/asrix03/Documents/rag/rag_v4.1/temp/pdf_image/"

    # Desired output image format
    output_format = "png"

    # Minimum width and height for extracted images
    min_width = 10
    min_height = 10

    # Setting the language of the reader for text extraction from image
    reader = easyocr.Reader(['en']) 

    # Load the PDF document
    loader_for_text = PyPDFLoader(document_path)
    loader_for_image = fitz.open(document_path)

    # Extract text
    pdf_text = loader_for_text.load()

    # looping through PDF pages
    for page_index in range(len(loader_for_image)):

        print("Extrating page " + str(page_index))

        # Writing text into text file
        text_file= open("output.txt","a+")
        text_file.write(pdf_text[page_index].page_content)
        text_file.close()

        # Get the item in the page
        page = loader_for_image[page_index]

        # Get image list
        image_list = page.get_images(full=True)

        # Iterate over the images on the page
        for image_index, img in enumerate(image_list, start=1):
            # Get the XREF of the image
            xref = img[0]
            # Extract the image bytes
            base_image = loader_for_image.extract_image(xref)
            image_bytes = base_image["image"]
            # Load it to PIL
            image = Image.open(io.BytesIO(image_bytes))

            # Check if the image meets the minimum dimensions and save it
            if image.width >= min_width and image.height >= min_height:
                image_path = os.path.join(output_dir, f"image{page_index + 1}_{image_index}.{output_format}")
                image.save(
                    open(image_path, "wb"),
                    format=output_format.upper())
                    
                # read the image
                result = reader.readtext(image_path, detail = 0, paragraph=True)

                # write image text into the text file
                text_file= open("output.txt","a+")
                for text in result:
                    text_file.write(text + "\n")
                text_file.close()

        # Deleting image after extracting text
        for image_index,img in enumerate(image_list, start=1):
            imageFileName = os.path.join(output_dir, f"image{page_index + 1}_{image_index}.{output_format}")
            if image.width >= min_width and image.height >= min_height:
                os.remove(imageFileName)
            
    loader = UnstructuredFileLoader("output.txt")
    pages = loader.load_and_split()

    os.remove("output.txt")

    return pages

