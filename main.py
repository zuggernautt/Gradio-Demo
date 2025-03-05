import gradio as gr
import re
import traceback
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def extract_video_id(youtube_url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&\s]+)',
        r'(?:https?:\/\/)?youtu\.be\/([^&\s]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^&\s]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^&\s]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    raise ValueError("Invalid YouTube URL")

def safe_summarize(text, tokenizer, model, max_length=250, min_length=50):
    """
    Safely summarize text with error handling for different text lengths
    """
    try:
        # Tokenize the input text
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
        
        # Generate summary
        summary_ids = model.generate(
            inputs['input_ids'], 
            max_length=max_length + 50,  # Allow some buffer
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Summarization error: {e}")
        return "Could not generate summary due to an error."

def summarize_youtube_video(youtube_url, max_length=250, min_length=50):
    try:
        # Load summarization model
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Extract video ID
        video_id = extract_video_id(youtube_url)
        
        # Fetch transcript
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as transcript_error:
            return f"Error fetching transcript: {str(transcript_error)}"
        
        # Combine transcript text
        if not transcript:
            return "No transcript available for this video."
        
        full_text = ' '.join([entry['text'] for entry in transcript])
        
        # Ensure we have some text
        if not full_text.strip():
            return "Empty transcript found."
        
        # Summarize
        summary = safe_summarize(
            full_text, 
            tokenizer, 
            model,
            max_length=max_length, 
            min_length=min_length
        )
        
        return summary
    
    except Exception as e:
        # Comprehensive error tracking
        error_trace = traceback.format_exc()
        print(error_trace)
        return f"An unexpected error occurred: {str(e)}"

# Create Gradio Interface
demo = gr.Interface(
    fn=summarize_youtube_video,
    inputs=[
        gr.Textbox(label="YouTube Video URL"),
        gr.Slider(minimum=50, maximum=500, value=250, label="Maximum Summary Length", step=1),
        gr.Slider(minimum=10, maximum=100, value=50, label="Minimum Summary Length", step=1)
    ],
    outputs=gr.Textbox(label="Video Summary"),
    title="YouTube Video Summarizer",
    description="Paste a YouTube video URL to get an AI-generated summary of its content.",
    examples=[
        ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        ["https://youtu.be/J8O9_ugpDjE"]
    ]
)

demo.launch()