from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import pipeline

#The transformers Pipeline
sent_pipeline = pipeline("sentiment-analysis")


sia = SentimentIntensityAnalyzer()

text = input("Enter your text: ")

# Vader result
vader_result = sia.polarity_scores(text)
print("Vader Sentiment Analysis:")
print(vader_result)

#Loading roberta model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Run for Roberta Model
encoded_text = tokenizer(text, return_tensors='pt')
output = model(**encoded_text)
scores= output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2],
}
# Result of Roberta model
print("Roberta Sentiment Analysis:")
print(scores_dict)

# Result of model from huggingface
huggingface_result = sent_pipeline(text)
print("Hugging Face Sentiment Analysis:")
print(huggingface_result)