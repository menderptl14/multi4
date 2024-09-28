from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import os 
import pathlib
import textwrap

from PIL import Image

import google.generativeai as genai 

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,image,prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input,image[0],prompt])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,  
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")



# st.set_page_config(page_title="Gemini Ai Image Creation")
st.header("Image Analyser ")
input=st.text_input("Input Prompt: ",key="input")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image=""
if uploaded_file is not None :
    image = Image.open(uploaded_file)
    # newImg = image.resize((100, 100))
    st.image(image, caption="Uploaded Image.",width=2, use_column_width=True)


submit=st.button("explain the image to me 👇 ")

input_prompt=""" what do you understand this picture
                 and What do you mean this picture
              """

if submit:
    image_data=input_image_setup(uploaded_file)
    response=get_gemini_response(input_prompt,image_data,input)
    st.subheader("The Response is")
    st.write(response)

