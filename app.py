import streamlit as st

# --- PAGE SETUP ---
about_page = st.Page(
    "views/home.py",
    title="Home",
    icon=":material/account_circle:",
    default=True,
)
project_1_page = st.Page(
    "views/chat.py",
    title="Chat_bot",
    icon=":material/bar_chart:",
)
project_2_page = st.Page(
    "views/image.py",
    title="Image_Analyzer",
    icon=":material/smart_toy:",
)

# project_3_page = st.Page(
#     "views/pdf_.py",
#     title="pdf_Summarise",
#     icon=":material/bar_chart:",
# )

project_3_page = st.Page(
    "views/audio.py",
    title="Audio Analyzer",
    icon=":material/bar_chart:",
)

# project_5_page = st.Page(
#     "views/pdf_4.py",
#     title="pdf__",
#     icon=":material/bar_chart:",
# )



# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [about_page],
        "Projects": [project_1_page, project_2_page,project_3_page],
    }
)


# --- SHARED ON ALL PAGES ---
st.logo("assests/book.jpg")
st.sidebar.markdown("Made with ❤️ by [Byte Builders]")


# --- RUN NAVIGATION ---
pg.run()
