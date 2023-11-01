import gradio as gr
import whisper
from transformers import pipeline

#load model
model = whisper.load_model("base")

sentiment_analysis = pipeline(
  "sentiment-analysis",
  framework="pt",
  model="SamLowe/roberta-base-go_emotions"
)

#functions takes text input and performs sentiment analysis
def analyze_sentiment(text):
  results = sentiment_analysis(text)
  sentiment_results = {
    result['label']: result['score'] for result in results
  }
  return sentiment_results

#This function assigns emojis for each sentiment
def get_sentiment_emoji(sentiment):
  # Define the mapping of sentiments to emojis
  emoji_mapping = {
    "disappointment": "😞",
    "sadness": "😢",
    "annoyance": "😠",
    "neutral": "😐",
    "disapproval": "👎",
    "realization": "😮",
    "nervousness": "😬",
    "approval": "👍",
    "joy": "😄",
    "anger": "😡",
    "embarrassment": "😳",
    "caring": "🤗",
    "remorse": "😔",
    "disgust": "🤢",
    "grief": "😥",
    "confusion": "😕",
    "relief": "😌",
    "desire": "😍",
    "admiration": "😌",
    "optimism": "😊",
    "fear": "😨",
    "love": "❤️",
    "excitement": "🎉",
    "curiosity": "🤔",
    "amusement": "😄",
    "surprise": "😲",
    "gratitude": "🙏",
    "pride": "🦁"
  }
  return emoji_mapping.get(sentiment, "")

#Function displays sentiment results based on a selected option
def display_sentiment_results(sentiment_results, option):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
    if option == "Sentiment Only":
        sentiment_text += f"{sentiment} {emoji}\n"
    elif option == "Sentiment + Score":
        sentiment_text += f"{sentiment} {emoji}: {score}\n"
    return sentiment_text

#This function performs Hugging Face’s inference process, including language identification, speech recognition, and sentiment analysis
def inference(audio, sentiment_option):
  audio = whisper.load_audio(audio)
  audio = whisper.pad_or_trim(audio)

  mel = whisper.log_mel_spectrogram(audio).to(model.device)

  _, probs = model.detect_language(mel)
  lang = max(probs, key=probs.get)

  options = whisper.DecodingOptions(fp16=False)
  result = whisper.decode(model, mel, options)

  sentiment_results = analyze_sentiment(result.text)
  sentiment_output = display_sentiment_results(sentiment_results, sentiment_option)

  return lang.upper(), result.text, sentiment_output

#Creating the user interface
title = """🎤 Audio Analysis 💬"""
image_path = "thumbnail.jpg"

description = """
  This demonstration highlights the capabilities of Whisper, a versatile speech recognition model. It has been trained on a vast and diverse audio dataset, enabling it to handle multilingual speech recognition and language identification tasks. 📝 To delve deeper into the specifics, you can refer to the GitHub repository(https://github.com/openai/whisper). ⚙️ Here are some key features of this tool:<br><br>

  - Real-time multilingual speech recognition<br>
  - Language identification<br>
  - Sentiment analysis of transcriptions<br><br>

  🎯 The sentiment analysis results are presented in the form of a dictionary that includes various emotions and their corresponding scores. 😃 Emojis are used to visually represent these sentiments in the results. ✅ A higher score for a particular emotion indicates a more pronounced presence of that emotion in the transcribed text. ❓ If you'd like to experience real-time speech recognition, simply activate your microphone. ⚡️ The model will then transcribe the audio and perform sentiment analysis on the resulting text.
"""

custom_css = """
  #title {
    text-align: center;
  }
  #banner-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
  #chat-message {
    font-size: 14px;
    min-height: 300px;
  }
"""

#Gradio’s UI framework is based on the concept of blocks. A block is used to define layouts, components, and events combined to create a complete interface with which users can interact.
block = gr.Blocks(css=custom_css)

with block:
    gr.HTML(f"<h1 id='title'>{title}</h1>")
    with gr.Row():
        with gr.Column():
            gr.Image(image_path, elem_id="banner-image", show_label=False)
        with gr.Column():
            gr.HTML(description,elem_id="description")
    #creating the form component
    with gr.Group():
        with gr.Box():
            # Audio Input
            audio = gr.Audio(
            label="Input Audio",
            show_label=False,
            source="microphone",
            type="filepath"
            )

            # Sentiment Option
            sentiment_option = gr.Radio(
            choices=["Sentiment Only", "Sentiment + Score"],
            label="Select an option",
            default="Sentiment Only"
            )

            # Transcribe Button
            btn = gr.Button("Transcribe")
        #Next, we define Textbox() components as output components for the detected language, transcription, and sentiment analysis results.
        lang_str = gr.Textbox(label="Language")
        text = gr.Textbox(label="Transcription")
        sentiment_output = gr.Textbox(label="Sentiment Analysis Results", output=True)
        #Button
        btn.click(
        inference,
        inputs=[
            audio,
            sentiment_option
        ],
        outputs=[
            lang_str,
            text,
            sentiment_output
        ]
        )
        #footer html
        gr.HTML('''
        <div class="footer">
            <p>Model by <a href="https://github.com/openai/whisper" style="text-decoration: underline;" target="_blank">OpenAI</a>
            </p>
        </div>
        ''')
    
    block.launch()