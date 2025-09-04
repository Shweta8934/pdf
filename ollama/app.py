import streamlit as st
from pypdf import PdfReader
import ollama

st.set_page_config(page_title="PDF Q&A (Ollama)", page_icon="📄")
st.title("📄 PDF Q&A with Ollama")

# Function to chunk text
def chunk_text(text, chunk_size=2000):
    """Split text into chunks for better handling."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text + "\n"

    st.success("✅ PDF loaded successfully! Ask your question below:")

    query = st.text_input("Your question:")
    if query:
        with st.spinner("Thinking with Ollama..."):
            try:
                # Break large PDF into chunks
                pdf_chunks = chunk_text(pdf_text, chunk_size=2000)

                # Use the most relevant chunk (for simplicity, take first one for now)
                # You can improve this with embeddings or similarity search
                context = pdf_chunks[0]

                # Prepare prompt
                prompt = f"""You are a helpful assistant.
Use the following PDF content to answer the question accurately.

PDF Content:
{context}

Question:
{query}

Answer in detail based on the PDF content only."""

                # Call Ollama in NON-streaming mode
                response = ollama.chat(
                    model="llama3",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False  # ✅ Force non-streaming
                )

                # Extract answer
                if "message" in response and "content" in response["message"]:
                    answer = response["message"]["content"]
                else:
                    answer = "⚠️ Could not extract answer from Ollama response."

                st.subheader("Answer:")
                st.write(answer)

            except Exception as e:
                st.error(f"Error: {str(e)}")
