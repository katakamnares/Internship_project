

import streamlit as st

st.title('Upload Your Resume File')
file_uploader = st.file_uploader('Upload Your Resume',type = ['pdf','docx'])

if file_uploader is not None:
  st.session_state['file'] = file_uploader
  st.switch_page(r"pages/resume_streamlit (3).py")