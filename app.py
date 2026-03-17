import streamlit as st
import validators
import requests
from bs4 import BeautifulSoup

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi


st.set_page_config(page_title="Summarizer", page_icon="📖")
st.title("📖 YouTube / Website Summarizer")

groq_api_key = st.text_input("Groq API Key", type="password")
generic_url = st.text_input("Enter URL", placeholder="Paste YouTube or website link here")


prompt_template = """
Provide a summary of the following content in 300 words:

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

#extract text from website using bs4
def extract_text_from_url(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text

    except:
        return ""



if st.button("Summarize"):

    if not groq_api_key:
        st.error("Please enter Groq API key")
        st.stop()

    if not generic_url.strip():
        st.error("Please enter a URL")
        st.stop()

    if not validators.url(generic_url):
        st.error("Invalid URL")
        st.stop()

    try:
        with st.spinner("Fetching content..."):

            
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=groq_api_key,
                streaming=False
            )

            if "youtube.com" in generic_url or "youtu.be" in generic_url:

                    video_id = generic_url.split("v=")[-1]
                    transcript = YouTubeTranscriptApi().fetch(video_id=video_id, languages=['en']).to_raw_data()
                    text = " ".join([entry['text'] for entry in transcript])
                    docs = [Document(page_content=text)]

            else:
                text = extract_text_from_url(generic_url)


                if not text:
                    st.error("Could not extract content from the website")
                    st.stop()


            docs = [Document(page_content=text)]  # for stuff document chain

            # # for map-reduce,when large docs are used
            # text_splitter = RecursiveCharacterTextSplitter(
            #     chunk_size=2000,
            #     chunk_overlap=200
            # )

            # docs = text_splitter.create_documents([text])

            chain = load_summarize_chain(
                llm,
                chain_type="stuff",
                prompt=prompt
            )

            result = chain.invoke(docs)

            st.success("### Summary")
            st.write(result["output_text"])

    except Exception as e:
        st.error(f"Error: {str(e)}")