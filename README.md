# Aspect-based-sentiment-analysis
 The goal of this project is to perform aspect-based sentiment analysis on restaurant reviews. This goal was divided into two subtasks:
 1. Extract aspects from reviews and define the category for each aspect: Whole, Service, Food, Interior and Price
 2. Classify the sentiment of each aspect.
 
 ## Dataset
 Our dataset consists of 284 restaurant reviews written in Russian. For each review the extracted aspects are provided, 
 each aspect has a category tag: Whole, Service, Interior, Food, Price and a sentiment tag: Neutral, Negative, Positive, Both
 
 ## Solution
 For both subtasks, I finetuned [`multiligual deberta`](https://huggingface.co/microsoft/mdeberta-v3-base). The first task was solved as a classic token classification problem. 
 The second task was solved as sequence classification problem. The input for the model was the concatenation of aspect and
 the abstract of text containing this aspect, the aim of the model was to predict the sentiment in respect to this aspect.
 
 You may perform inference on your data with [`inference.py`](https://github.com/ZaitsevaDasha/Aspect-based-sentiment-analysis/blob/main/inference.py)
 ```
 python inference.py --aspect_model_path <path_to_aspect_model> --sentiment_model_path <path_to_sentiment_model> --path_to_review <path_to_review>
 ```
 You need to provide the path to both models and a txt file with review text
 
 The link for downloading models: https://drive.google.com/drive/folders/1mYsl9gg5Nqgd1y29uXSorwJAmi-kFZ7X
