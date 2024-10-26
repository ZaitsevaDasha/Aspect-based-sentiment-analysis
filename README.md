# Aspect-based-sentiment-analysis
 The goal of this project is to perform aspects-based sentiment analysis on restaurent reviews. Our dataset consists of 284 reviews written in Russian language
 This goal is divided into two subtasks:
 1. Extract aspects from reviews and define the category for each aspect: Whole, Service, Food,Interior and Price
 2. Classify the sentiment of each aspect in to positive, negative, neutral or both
 
 For both subtasks multiligual deberta was used. The first task was solved as token classification problem. 
 The second task was solved as sequence classification. The input for the model was the concatenation of aspect and
 the abstract of text with this aspect
 
 You may inference the models on your data with [`inference.py`](https://github.com/ZaitsevaDasha/Aspect-based-sentiment-analysis/blob/main/inference.py)
 ```
 python inference.py --aspect_model_path <path_to_aspect_model> --sentiment_model_path <path_to_sentiment_model> --path_to_review <path_to_review>
 ```
 You need to provide path to each model and a txt file with review text
 
 The link for downloading models: https://drive.google.com/drive/folders/1mYsl9gg5Nqgd1y29uXSorwJAmi-kFZ7X
