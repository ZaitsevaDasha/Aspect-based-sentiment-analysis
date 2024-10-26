# Aspect-based-sentiment-analysis
 The goal of this project is to perform aspects-based sentiment analysis on restaurent reviews.
 This goal is divided into two subtasks:
 1. Extract aspects from reviews and define the category for each aspect: Whole, Service, Food,Interior and Price
 2. Classify the sentiment of each aspect.
 
 For both subtasks multiligual deberta was used. The first task was solved as token classification problem. 
 The second task was solved as sequence classification. The input for the model was the concatenation of aspect and
 the abstract of text with this aspect
