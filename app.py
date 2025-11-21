import streamlit as st
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline

# --- CONFIGURATION ---
MODEL_NAME = "facebook/bart-large-cnn"

# Use Streamlit's resource caching to load the model and tokenizer only once.
# This makes the app run much faster after the initial load.
@st.cache_resource
def load_summarization_pipeline():
    """
    Loads the Hugging Face summarization pipeline for efficiency.
    We use the 'pipeline' API for simplicity in a web application.
    """
    # Initialize the pipeline with the specified model
    summarizer = pipeline(
        "summarization",
        model=MODEL_NAME,
        device=0 if st.session_state.use_gpu else -1 # Use GPU (0) if available, otherwise CPU (-1)
    )
    return summarizer

# --- PAGE CONFIGURATION AND UI ---

# Set up the Streamlit page details
st.set_page_config(
    page_title="AI-Powered Text Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 AI-Powered Text Summarizer")
st.markdown("A simple tool using the **BART Large CNN** model from Hugging Face for abstractive summarization.")

# Check for GPU (simple flag for the user)
st.sidebar.header("Settings")
# Use session state for persistent GPU check
if 'use_gpu' not in st.session_state:
    st.session_state.use_gpu = True if torch.cuda.is_available() else False
st.session_state.use_gpu = st.sidebar.checkbox(
    "Use GPU (if available)",
    value=st.session_state.use_gpu,
    disabled=not torch.cuda.is_available()
)

# Initialize the pipeline (this runs once due to @st.cache_resource)
with st.spinner(f"Loading model **{MODEL_NAME}**... this may take a moment."):
    summarizer = load_summarization_pipeline()

# --- INPUT AND PARAMETERS ---

col1, col2 = st.columns(2)

with col1:
    st.header("Input Text")
    # Text area for user input
    input_text = st.text_area(
        "Paste your long text here:",
        height=300,
        placeholder="Paste a long article, document, or essay to get a concise summary."
    )

with col2:
    st.header("Summary Parameters")
    # Sliders for generation control
    min_len = st.slider(
        "Minimum Summary Length",
        min_value=20,
        max_value=100,
        value=40,
        step=5
    )
    max_len = st.slider(
        "Maximum Summary Length",
        min_value=min_len + 10,  # Ensure max > min
        max_value=500,
        value=150,
        step=10
    )
    
    # Button to trigger summarization
    if st.button("Generate Summary", type="primary", use_container_width=True):
        if input_text:
            # --- SUMMARIZATION EXECUTION ---
            with st.spinner("Generating summary..."):
                try:
                    # Pass the input text and parameters to the pipeline
                    summary_result = summarizer(
                        input_text,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False  # Use beam search for higher quality
                    )
                    # The pipeline returns a list of dictionaries; we extract the text
                    st.session_state.summary = summary_result[0]['summary_text']
                    st.session_state.input_len = len(input_text.split())
                    st.session_state.summary_len = len(st.session_state.summary.split())
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please paste some text into the input box to summarize.")

# --- OUTPUT DISPLAY ---

st.divider()
st.header("Output Summary")

# Display the summary if it has been generated
if 'summary' in st.session_state:
    st.success("Summary Generated!")
    st.info(st.session_state.summary)
    
    # Show statistics
    st.markdown(f"""
    * **Original Text Word Count:** {st.session_state.input_len}
    * **Summary Word Count:** {st.session_state.summary_len}
    """)
else:
    st.info("The generated summary will appear here.")