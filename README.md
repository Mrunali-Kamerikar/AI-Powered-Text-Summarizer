
# 🤖 AI-Powered Text Summarizer Web App

This project implements a web-based text summarizer using a state-of-the-art Hugging Face transformer model, powered by a Streamlit interface. It allows users to paste long documents and generate concise, abstractive summaries with controllable length parameters.

## ✨ Features

* **Abstractive Summarization:** Uses the powerful **BART Large CNN** model to generate human-like, coherent summaries (not just extracting key sentences).
* **Interactive Web Interface:** Built with Streamlit for easy, intuitive interaction.
* **Customizable Length:** Users can define the minimum and maximum length of the output summary using sliders.
* **Performance Optimized:** Utilizes Streamlit's caching (`@st.cache_resource`) to load the large model only once, ensuring quick subsequent summarization requests.
* **GPU Detection:** Automatically detects and offers the option to use a CUDA-enabled GPU for faster processing.

## 🛠️ Prerequisites

To run this application, you must have Python (3.8+) installed.

You will need the following libraries:

* `transformers` (for the BART model and pipeline)
* `torch` (for the deep learning backend)
* `streamlit` (for the web interface)

## 🚀 Getting Started

Follow these steps to set up and run the Text Summarizer on your local machine.

### 1. Clone the Repository (Optional)

If you host your code on Git, users would clone it here:

```bash
git clone <your-repository-url>
cd AI-Powered-Text-Summarizer
````

### 2\. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies:

```bash
# Create the environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate
# Activate the environment (macOS/Linux)
source venv/bin/activate
```

### 3\. Install Dependencies

With the environment active, install all necessary packages:

```bash
pip install streamlit transformers torch
```

### 4\. Run the Application

Execute the Streamlit script to launch the web app:

```bash
streamlit run app.py
```

### 5\. Access the App

The command above will automatically open the application in your default web browser (or provide the link):

  * **Local URL:** `http://localhost:8501`

## ⚙️ How to Use

1.  **Paste Text:** Copy the long article or document snippet into the **Input Text** box on the left.
2.  **Adjust Parameters:** Use the **Summary Parameters** sliders to set the desired *Minimum* and *Maximum* word count for the output summary.
3.  **Generate:** Click the **Generate Summary** button.
4.  **View Results:** The final concise summary will appear in the **Output Summary** section below, along with the word counts for comparison.

## 📄 Core Model Details

| Component | Detail |
| :--- | :--- |
| **Summarization Model** | `facebook/bart-large-cnn` |
| **Model Type** | BART (Bidirectional and Auto-Regressive Transformer) |
| **Summarization Type** | Abstractive (Generates new sentences) |
| **Source** | Hugging Face `transformers` library |

## 💡 Future Enhancements

  * [ ] Add support for multiple summarization models (e.g., T5, Pegasus).
  * [ ] Implement a feature to summarize text directly from a provided URL.
  * [ ] Integrate a download button to save the generated summary as a text file.
  * [ ] Improve performance by using quantised models if high-quality GPUs are unavailable.

## 🤝 Contact and Support

If you have any questions or suggestions, please feel free to open an Issue in this repository.

-----

## Created by Mrunali Kamerikar



