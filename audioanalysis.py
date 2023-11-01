from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

def get_text_from_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print('Adjusting for background noise...')
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print('Listening to your message...')
        recorded_audio = recognizer.listen(source)
        print('Done recording')

    try:
        print('Transcribing the message...')
        text = recognizer.recognize_google(recorded_audio, language='en-US')
        print('Your message: {}'.format(text))
        return text
    except Exception as e:
        print(e)
        return None

def analyze_sentiment(text):
    if text:
        sentence = [text]
        analyzer = SentimentIntensityAnalyzer()
        for i in sentence:
            sentiment = analyzer.polarity_scores(i)
            print(sentiment)
    else:
        print("No text to analyze.")

# Getting text from audio
text_from_audio = get_text_from_audio()

# Analyzing sentiment if text is available
analyze_sentiment(text_from_audio)
