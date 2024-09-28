import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# def main():
#     # st.set_page_config(page_title="EduPlatform - Learn Anywhere, Anytime", page_icon="📚", layout="wide")

    # st.title("Welcome to EduByte")
    # st.subheader("Empowering Learners Worldwide")

    # st.write("""
    # EduByte is your gateway to making knowledge into small chunks and learning at your own pace.
    # """)

    # st.header("Our Features")
    # col1, col2 = st.columns(2)

    # with col1:
    #     st.subheader("📚 Chatbot")
    #     st.write("Explore a wide range of subjects by interacting with our chatbot.")

    #     st.subheader("PDF Summarizer")
    #     st.write("Go through any PDF in seconds with this feature.")

    #     st.subheader("📝 Quiz System")
    #     st.write("Test your knowledge with interactive quizzes.")

    # with col2:
    #     st.subheader("🌐 Image Analyser")
    #     st.write("Analyze any kind of image with text recognition.")

    #     st.subheader("📊 Audio Summarizer")
    #     st.write("Analyze and summarize any audio into text.")

    # st.header("Get Started Today!")
    # if st.button("Sign Up Now"):
    #     st.success("Thanks for your interest! Sign-up functionality coming soon.")

    # st.markdown("---")
    # st.write("© 2024 EduPlatform. All rights reserved.")

# if __name__ == "__main__":
#     main()


st.subheader("Empowering Learners Worldwide")

st.write("""
    EduByte is your gateway to making knowledge into small chunks and learning at your own pace.
    """)

st.header("Our Features")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Chatbot")
    st.write("Explore a wide range of subjects by interacting with our chatbot.")

    

    # st.subheader("PDF Summarizer")
    # st.write("Go through any PDF in seconds with this feature.")

    st.subheader("📝 Quiz System")
    st.write("Test your knowledge with interactive quizzes.")

with col2:
    st.subheader("🌐 Image Analyser")
    st.write("Analyze any kind of image with text recognition.")
    
    st.subheader("Audio Analyser")
    st.write("Analyser audio files in form of text summary")
      
st.header("Get Started Today!")
if st.button("Sign Up Now"):
    st.success("Thanks for your interest! Sign-up functionality coming soon.")

    st.markdown("---")
    st.write("© 2024 EduPlatform. All rights reserved.")
