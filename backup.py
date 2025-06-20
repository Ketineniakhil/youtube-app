#-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
# Importing Libraries
import streamlit as st
from pathvalidate import sanitize_filename
from streamlit_option_menu import option_menu
import base64
from bs4 import BeautifulSoup
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse
from textwrap import dedent
from pytube import YouTube
from deep_translator import GoogleTranslator
from gtts import gTTS
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
from string import punctuation
from heapq import nlargest
import spacy
import en_core_web_sm
import math
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import pandas as pd
import plotly.express as px
import time


# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Setup page config
st.set_page_config(
    page_title="Text Summarizer & Translator",
    page_icon='favicon.ico',
    layout="wide",
    initial_sidebar_state="collapsed",
)

#-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
# All Functions    
def translate_text(text, target_lang='en', source_lang='auto'):
    """
    Translates text while safely handling character limits using hybrid chunking.
    """
    from deep_translator import GoogleTranslator
    
    # Configuration
    MAX_CHARS = 4500  # Conservative limit for API safety
    MIN_SPLIT_LENGTH = 100  # Minimum length for sentence splitting
    
    if not text:
        return ""
    
    # Clean text and handle newlines
    text = text.replace('\n', ' ').strip()
    if len(text) <= MAX_CHARS:
        try:
            return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        except Exception as e:
            raise Exception(f"Translation error: {str(e)}")

    # Hybrid chunking algorithm
    def chunk_text(input_text):
        chunks = []
        
        # First pass: Split by sentences where possible
        sentences = []
        temp = input_text.split('. ')
        for s in temp:
            s = s.strip()
            if s:
                if not s.endswith('.'):
                    s += '.'
                sentences.append(s)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > MAX_CHARS:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            current_chunk += sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Second pass: Split remaining large chunks by paragraphs
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > MAX_CHARS:
                paragraphs = [p.strip() for p in chunk.split('\n') if p.strip()]
                current_para_chunk = ""
                for para in paragraphs:
                    if len(current_para_chunk) + len(para) + 1 > MAX_CHARS:
                        if current_para_chunk:
                            final_chunks.append(current_para_chunk)
                            current_para_chunk = ""
                    current_para_chunk += para + "\n"
                if current_para_chunk:
                    final_chunks.append(current_para_chunk.strip())
            else:
                final_chunks.append(chunk)

        # Final safety check for character limits
        safe_chunks = []
        for chunk in final_chunks:
            if len(chunk) > MAX_CHARS:
                safe_chunks.extend([chunk[i:i+MAX_CHARS] for i in range(0, len(chunk), MAX_CHARS)])
            else:
                safe_chunks.append(chunk)
                
        return safe_chunks

    # Split and translate chunks
    chunks = chunk_text(text)
    translated_chunks = []
    
    for i, chunk in enumerate(chunks):
        try:
            translated = GoogleTranslator(source=source_lang, target=target_lang).translate(chunk)
            translated_chunks.append(translated)
        except Exception as e:
            raise Exception(f"Failed to translate chunk {i+1}/{len(chunks)}: {str(e)}")
    
    # Reconstruct text with better spacing
    return ' '.join(translated_chunks).replace(' .', '.').replace(' ,', ',')

# Sumy Summarization
def sumy_summarize(text_content, percent):
    parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    summarizer = LsaSummarizer()
    num_sentences = max(1, int(len(text_content.split('.')) * (int(percent) / 100)))
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

def summarize_text(text_content, percent):
    return sumy_summarize(text_content, percent)

# NLTK Summarization
def nltk_summarize(text_content, percent):
    tokens = word_tokenize(text_content)
    stop_words = stopwords.words('english')
    punctuation_items = punctuation + '\n'

    word_frequencies = {}
    for word in tokens:
        if word.lower() not in stop_words:
            if word.lower() not in punctuation_items:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
                    
    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_token = sent_tokenize(text_content)
    sentence_scores = {}
    for sent in sentence_token:
        sentence = sent.split(" ")
        for word in sentence:
            if word.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.lower()]

    select_length = int(len(sentence_token) * (int(percent) / 100))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word for word in summary]
    summary = ' '.join(final_summary)
    return summary

# Spacy Summarization
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
def spacy_summarize(text_content, percent):
    stop_words = list(spacy.lang.en.stop_words.STOP_WORDS)
    punctuation_items = punctuation + '\n'
    nlp = spacy.load('en_core_web_sm')

    nlp_object = nlp(text_content)
    word_frequencies = {}
    for word in nlp_object:
        if word.text.lower() not in stop_words:
            if word.text.lower() not in punctuation_items:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
                    
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_token = [sentence for sentence in nlp_object.sents]
    sentence_scores = {}
    for sent in sentence_token:
        sentence = sent.text.split(" ")
        for word in sentence:
            if word.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.lower()]

    select_length = int(len(sentence_token) * (int(percent) / 100))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    return summary

# TF-IDF Summary Functions
def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def _find_average_score(sentenceValue):
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    average = (sumValues / len(sentenceValue))
    return average

def _score_sentences(tf_idf_matrix):
    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table

def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix

def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix

def tf_idf_summarize(transcript, percent):
    sentences = sent_tokenize(transcript)
    total_documents = len(sentences)
    
    freq_matrix = _create_frequency_matrix(sentences)
    tf_matrix = _create_tf_matrix(freq_matrix)
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    sentence_scores = _score_sentences(tf_idf_matrix)
    threshold = _find_average_score(sentence_scores)
    summary = _generate_summary(sentences, sentence_scores, 1.0 * threshold)
    
    return summary

def abstractive_summarize(transcript):
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs, 
        max_length=150, 
        min_length=40, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True)
    
    return tokenizer.decode(outputs[0])

# Create audio from text
def create_audio(text, language_code):
    speech = gTTS(text=text, lang=language_code, slow=False)
    speech.save('user_trans.mp3')
    return 'user_trans.mp3'

# CSS for modern theme
def load_css():
    st.markdown("""
    <style>
    /* Overall Styling */
    .main {
        background-color: #f9f9f9;
        color: #333;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #1E3A8A;
        font-weight: 600;
    }
    
    /* Container Styling */
    .css-1y4p8pa {
        padding: 1rem;
    }
    
    /* Card Styling */
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #1E3A8A;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Hiding footer and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    
    /* Select box styling */
    .stSelectbox > div > div > div {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        background-color: #f1f5f9;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    
    /* Navigation Menu Styling */
    .nav-link {
        color: #1E3A8A !important;
        font-weight: 600;
    }
    
    .nav-link.active {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    
    /* Summary box styling */
    .summary-box {
        background-color: #f8fafc;
        border-left: 5px solid #1E3A8A;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
    
    /* Audio player styling */
    .stAudio > div {
        border-radius: 10px;
        overflow: hidden;
    }
    
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
    }
    
    .logo-container img {
        height: 50px;
        margin-right: 1rem;
    }
    
    .nav-container {
        display: flex;
        gap: 1rem;
    }
    
    /* Transcript styling */
    .transcript-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 15px;
        background-color: #f8fafc;
        border-radius: 5px;
        border-left: 5px solid #1E3A8A;
        font-family: monospace;
        white-space: pre-wrap;
        line-height: 1.6;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        min-height: 200px;
    }
    </style>
    """, unsafe_allow_html=True)

#-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
# Main Application

def main():
    # Load CSS
    load_css()
    
    # App Logo and Header
    col1, col2 = st.columns([1, 5])
    
    with col1:
        st.image("app_logo.gif", width=100)
    
    with col2:
        st.title("Text Summarizer & Translator")
        st.markdown("<p style='margin-top:-10px;'>Summarize English text and convert to Indian languages with audio</p>", unsafe_allow_html=True)
    
    # Create top navigation
    selected = option_menu(
        menu_title=None,
        options=["Summarize", "History", "About"],
        icons=["file-text", "clock-history", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "margin": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": "#1E3A8A", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "padding": "10px 20px", "--hover-color": "#e6effc"},
            "nav-link-selected": {"background-color": "#1E3A8A", "color": "white"},
        }
    )
    
    # Summarize Section
    if selected == "Summarize":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # Text Input Area
        text_content = st.text_area('Enter English Text to Summarize', height=300,
                                  placeholder="Paste your English text here...")
        
        # Two columns layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if text_content.strip():
                # Display text stats
                word_count = len(text_content.split())
                st.markdown(f"**Word Count:** {word_count}")
                
                # Show sample of the text
                with st.expander("View Input Text"):
                    st.text(text_content[:1000] + ("..." if len(text_content) > 1000 else ""))
        
        with col2:
            st.markdown("### Summary Options")
            
            # Summary type selection
            sumtype = st.radio(
                'Summarization Type',
                options=['Extractive', 'Abstractive (T5 Algorithm)']
            )
            
            if sumtype == 'Extractive':
                # Algorithm selection for extractive
                sumalgo = st.selectbox(
                    'Summarization Algorithm',
                    options=['Sumy', 'NLTK', 'Spacy', 'TF-IDF']
                )
                
                # Summary length slider
                length = st.select_slider(
                    'Summary Length',
                    options=['10%', '20%', '30%', '40%', '50%']
                )
            
            # Indian Languages selection
            indian_languages = {
                'hi': 'Hindi',
                'bn': 'Bengali',
                'te': 'Telugu',
                'mr': 'Marathi',
                'ta': 'Tamil',
                'ur': 'Urdu',
                'gu': 'Gujarati',
                'kn': 'Kannada',
                'ml': 'Malayalam',
                'pa': 'Punjabi',
                'or': 'Odia',
                'as': 'Assamese',
                'ne': 'Nepali'
            }
            
            target_lang = st.selectbox(
                "Translate To",
                options=indian_languages.values()
            )
            
            # Summarize button
            if st.button('Generate Summary', key='summary_button'):
                if text_content.strip():
                    try:
                        # Show progress
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        # Step 1: Generate summary
                        progress_text.text("Generating summary...")
                        progress_bar.progress(33)
                        
                        if sumtype == 'Extractive':
                            percent = int(length.strip('%'))
                            if sumalgo == 'Sumy':
                                summ = summarize_text(text_content, percent)
                            elif sumalgo == 'NLTK':
                                summ = nltk_summarize(text_content, percent)
                            elif sumalgo == 'Spacy':
                                summ = spacy_summarize(text_content, percent)
                            else:  # TF-IDF
                                summ = tf_idf_summarize(text_content, percent)
                        else:  # Abstractive
                            summ = abstractive_summarize(text_content)
                        
                        # Step 2: Translate summary
                        progress_text.text("Translating summary...")
                        progress_bar.progress(66)
                        lang_code = [k for k, v in indian_languages.items() if v == target_lang][0]
                        translated = translate_text(summ, target_lang=lang_code)
                        
                        # Step 3: Complete
                        progress_bar.progress(100)
                        progress_text.text("Summary generated successfully!")
                        time.sleep(1)
                        progress_text.empty()
                        progress_bar.empty()
                        
                        # Display summary
                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown("### 📄 Original Summary (English)")
                        st.markdown(f'<div class="summary-box">{summ}</div>', unsafe_allow_html=True)
                        
                        st.markdown(f"### 📄 Translated Summary ({target_lang})")
                        st.markdown(f'<div class="summary-box">{translated}</div>', unsafe_allow_html=True)
                        
                        # Audio generation
                        st.markdown(f"### 🎧 Listen to {target_lang} Summary")
                        try:
                            st.audio(create_audio(translated, lang_code))
                        except Exception as e:
                            st.warning(f"Audio generation failed: {str(e)}")

                        # Download options
                        st.download_button(
                            label="Download English Summary as Text",
                            data=summ,
                            file_name="english_summary.txt",
                            mime="text/plain"
                        )
                        
                        st.download_button(
                            label=f"Download {target_lang} Summary as Text",
                            data=translated,
                            file_name=f"{target_lang.lower()}_summary.txt",
                            mime="text/plain"
                        )

                        # Save to history
                        if 'history' not in st.session_state:
                            st.session_state.history = []

                        entry = {
                            'content': text_content[:100] + "..." if len(text_content) > 100 else text_content,
                            'english_summary': summ,
                            'translated_summary': translated,
                            'language': target_lang,
                            'timestamp': pd.Timestamp.now()
                        }

                        st.session_state.history.append(entry)
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                else:
                    st.error("Please enter some text to summarize")

        st.markdown("</div>", unsafe_allow_html=True)

    # History Section
    elif selected == "History":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Previous Summaries")
        
        if 'history' not in st.session_state or not st.session_state.history:
            st.info("No summary history found. Generate summaries to see them here.")
        else:
            # Clear history button
            if st.button("Clear History"):
                st.session_state.history = []
                st.experimental_rerun()
            
            # Display history entries
            for i, entry in enumerate(reversed(st.session_state.history)):
                st.markdown(f"#### Summary {i+1} - {entry['language']}")
                st.markdown(f"*Generated on {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}*")
                
                # Original content
                with st.expander("Show Original Text"):
                    st.markdown(entry['content'])
                
                # English summary
                with st.expander("Show English Summary"):
                    st.markdown(entry['english_summary'])
                
                # Translated summary
                with st.expander(f"Show {entry['language']} Summary"):
                    st.markdown(entry['translated_summary'])
                
                if i < len(st.session_state.history) - 1:
                    st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # About Section
    elif selected == "About":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        st.markdown("## About Text Summarizer & Translator")
        
        st.markdown("""
        This application helps you summarize English text and convert it to various Indian languages with audio support.
        """)

        st.markdown("### **How It Works**")
        st.markdown("""
        1. Enter English text in the input field
        2. Choose your preferred summarization method and length
        3. Select an Indian language for translation
        4. Click "Generate Summary" to process the text
        5. Review your summary in English and translated version
        6. Listen to the audio version of the translated summary
        """)

        st.markdown("### **Supported Indian Languages**")
        st.markdown("""
        - Hindi (हिन्दी)
        - Bengali (বাংলা)
        - Telugu (తెలుగు)
        - Marathi (मराठी)
        - Tamil (தமிழ்)
        - Urdu (اردو)
        - Gujarati (ગુજરાતી)
        - Kannada (ಕನ್ನಡ)
        - Malayalam (മലയാളം)
        - Punjabi (ਪੰਜਾਬੀ)
        - Odia (ଓଡ଼ିଆ)
        - Assamese (অসমীয়া)
        - Nepali (नेपाली)
        """)
        
        st.markdown("### **Technologies Used**")
        st.markdown("""
        * **Streamlit:** For the interactive web interface
        * **NLTK, Spacy, Sumy:** Natural language processing libraries for text summarization
        * **Hugging Face Transformers:** For T5 abstractive summarization
        * **Google Translate API:** For language translation
        * **gTTS (Google Text-to-Speech):** For audio generation
        """)
        
        st.markdown("---")
        
if __name__ == "__main__":
    main()
