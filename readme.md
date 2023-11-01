## Sentiment Analysis Project

Welcome to the Sentiment Analysis project, a comprehensive tool designed for evaluating both text and audio content.

### Overview
This project employs various libraries for text and audio analysis. For text analysis, it integrates Vader, Hugging Face's RoBERTa, and the Transformer pipeline. The text analysis modules utilize the following resources:

- **Vader**: For sentiment analysis in text. [VaderSentiment](https://pypi.org/project/vaderSentiment/)
- **RoBERTa**: Utilizing Hugging Face's RoBERTa model for text evaluation. [RoBERTa Documentation](https://huggingface.co/docs/transformers/model_doc/roberta)
- **Transformers**: Utilized in the text analysis pipeline. [Transformers](https://pypi.org/project/transformers/)

In the realm of audio analysis, the project integrates VaderSentiment and Speech Recognition libraries for a comprehensive understanding of audio content. The audio analysis incorporates:

- **VaderSentiment**: For sentiment analysis in audio. [VaderSentiment](https://pypi.org/project/vaderSentiment/)
- **Speech Recognition**: Utilized for recognizing speech in audio data. [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)

To enhance user experience in audio analysis, the project integrates OpenAI's Whisper. This interface simplifies tasks like language identification, speech recognition, and sentiment analysis. The user-friendly interface is powered by Gradio, offering an accessible experience for users.

- **Whisper**: Providing tools for multilingual audio analysis. [Whisper GitHub](https://github.com/openai/whisper)
- **Gradio**: Providing an intuitive user interface for interaction. [Gradio](https://www.gradio.app/)

### Installation
To set up the environment and initiate the project, follow these steps:

1. Run the following command in Anaconda to create the required environment:
   ```bash
   conda env create -f environment.yml
   ```
   Replace `environment.yml` with the path to your environment configuration file.

### Usage
Once the environment is configured, run the project applications as needed to conduct both text and audio analysis.

Explore the various features and functionalities offered in this Sentiment Analysis project to evaluate text and audio content effectively.

### References
https://www.smashingmagazine.com/2023/09/generating-real-time-audio-sentiment-analysis-ai/