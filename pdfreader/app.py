


import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF Q&A with OpenAI")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Enter OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password")

if uploaded_file and api_key:
    client = OpenAI(api_key=api_key)

    # Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() + "\n"

    st.success("PDF loaded! Creating Q&A system...")

    st.info("Ask questions about your PDF below:")

    query = st.text_input("Your question:")
    if query:
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",   # Or "gpt-4o" / "gpt-3.5-turbo"
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided PDF text."},
                        {"role": "user", "content": f"PDF Content:\n{pdf_text}\n\nQuestion: {query}"}
                    ],
                    max_tokens=500,
                    temperature=0
                )

                answer = response.choices[0].message.content
                st.write("**Answer:**", answer)

            except Exception as e:
                st.error(f"Error: {str(e)}")
