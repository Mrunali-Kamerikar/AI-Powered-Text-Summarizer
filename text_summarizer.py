import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# --- CONFIGURATION ---
# We'll use the 'facebook/bart-large-cnn' model, which is a common and high-performing choice for text summarization.
MODEL_NAME = "facebook/bart-large-cnn"

def initialize_model_and_tokenizer():
    """
    Loads the pre-trained BART model and its corresponding tokenizer.
    """
    try:
        # 1. Load the Tokenizer
        print(f"Loading Tokenizer: {MODEL_NAME}...")
        tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
        
        # 2. Load the Model
        print(f"Loading Model: {MODEL_NAME}...")
        model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
        
        # 3. Check for GPU/CPU (Optional but good practice)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        print(f"Model loaded successfully and using device: {device}")
        return model, tokenizer, device
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        return None, None, None

def generate_summary(text_to_summarize, model, tokenizer, device, max_length=150, min_length=40):
    """
    Generates a concise summary from a long piece of text.

    Args:
        text_to_summarize (str): The input text.
        model, tokenizer, device: The initialized components.
        max_length (int): The maximum length of the generated summary.
        min_length (int): The minimum length of the generated summary.

    Returns:
        str: The generated summary.
    """
    if not text_to_summarize or not model:
        return "Error: Model not loaded or no text provided."

    print("\n--- Starting Summarization ---")
    
    # 1. Tokenize the input text
    # The 'return_tensors="pt"' ensures the output is a PyTorch tensor.
    inputs = tokenizer(
        text_to_summarize,
        max_length=1024, # Standard maximum context length for many models
        truncation=True, # Truncate if the text is too long
        padding="max_length",
        return_tensors="pt"
    ).to(device) # Move tensors to the correct device (CPU/GPU)

    # 2. Generate the summary using the model
    # 'num_beams' is set to 4 for better quality output (beam search).
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0, # Encourages longer summaries
        early_stopping=True
    )

    # 3. Decode the summary tokens back into a readable string
    summary = tokenizer.decode(
        summary_ids.squeeze(),
        skip_special_tokens=True, # Removes special tokens like [CLS], [SEP]
        clean_up_tokenization_spaces=True
    )
    
    print("--- Summarization Complete ---")
    return summary

# --- MAIN EXECUTION BLOCK ---

# 1. Initialize the essential components once
bart_model, bart_tokenizer, device_used = initialize_model_and_tokenizer()

if bart_model and bart_tokenizer:
    # 2. Define the input text (User's input)
    # The user will replace this with their actual text
    INPUT_TEXT = """
    Artificial intelligence (AI) has rapidly transformed from a niche academic field into a pervasive technology influencing nearly every sector of the global economy. Its applications range from complex predictive analytics and autonomous vehicles to personalized customer service and medical diagnostics. The foundation of modern AI relies heavily on machine learning, particularly deep learning models, which use neural networks with many layers to process vast amounts of data. This allows them to identify patterns and make decisions or predictions with high accuracy. However, this advancement is not without challenges. Concerns about ethical implications, such as algorithmic bias and job displacement, are becoming increasingly prominent. Furthermore, the immense computational power required to train and run large models contributes to a significant carbon footprint, prompting researchers to seek more energy-efficient architectures. The future of AI is expected to involve more integration with edge computing, fostering faster, more decentralized intelligence closer to the data source.
    """
    
    # 3. Generate and display the summary
    final_summary = generate_summary(
        text_to_summarize=INPUT_TEXT,
        model=bart_model,
        tokenizer=bart_tokenizer,
        device=device_used
    )

    ## --- OUTPUT ---
    print("\n" + "="*50)
    print("        🤖 AI-POWERED TEXT SUMMARIZER RESULTS")
    print("="*50)
    print(f"Original Text Length: {len(INPUT_TEXT.split())} words")
    print("-" * 50)
    print("📜 GENERATED SUMMARY:")
    print(final_summary)
    print("-" * 50)