import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import numpy as np
import nltk
from nltk import sent_tokenize
import copy
import argparse

nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser(description='ABSA')
    parser.add_argument('--aspect_model_path', type=str)
    parser.add_argument('--sentiment_model_path', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--text', type=str)
    args = parser.parse_args()
    return args

class AspectExtractionPipeline():
    def __init__(self, tokenizer=None, model=None, device='cpu', id2tag=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.id2tag = id2tag

    def predict_classes(self, text):
        self.model.to(self.device)
        self.text = text
        encoding = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        self.encoding = copy.deepcopy(encoding)
        encoding.pop('offset_mapping')
        inputs = {key: value.to(self.device) for key, value in encoding.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)

        predictions = predictions.cpu().numpy()
        predictions_list = self.postprocess(predictions[0].tolist())

        return predictions_list

    def postprocess(self, predictions):
        tokens = self.tokenizer.convert_ids_to_tokens(self.encoding['input_ids'][0])
        offsets = self.encoding['offset_mapping'][0]
        all_predictions = []
        current_token = ''
        current_label = ''
        current_offset = []
        for ind, label in enumerate(predictions):
            if self.id2tag[label].startswith('B'):
                if current_token != '':
                    all_predictions.append([current_token, current_offset, current_label])
                    current_token = ''
                    current_label = ''
                    current_offset = []
                current_token = tokens[ind].strip('▁')
                current_label = self.id2tag[label].split('-')[1]
                offset = offsets[ind].tolist()
                if tokens[ind].startswith('▁'):
                    offset[0] += 1    
                current_offset = offset      
            if self.id2tag[label].startswith('I'):
                if current_token != '':
                    if tokens[ind].startswith('▁') or tokens[ind - 1] == '▁':
                        current_token = current_token + ' ' + tokens[ind].strip('▁')
                        current_offset[1] = offsets[ind][1]
                    else:
                        current_token = current_token + tokens[ind]
                        current_offset[1] = offsets[ind][1]
            if self.id2tag[label] == 'O':
                if current_token != '':
                    all_predictions.append([current_token, current_offset, current_label])
                    current_token = ''
                    current_label = ''
                    current_offset = []
        return all_predictions
    
class SentimentClassificationPipeline():
    def __init__(self, tokenizer=None, model=None, device='cpu', id2tag=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.id2tag = id2tag

    def make_pairs(self, aspects, review_text):
        all_pairs = []
        sentences = sent_tokenize(review_text)
        first_aspect = 0
        for ind, sentence in enumerate(sentences):
            start_of_sentence = review_text.index(sentence)
            end_of_sentence = start_of_sentence + len(sentence)
            for i in range(first_aspect, len(aspects)):
                aspect = aspects[i]
                if aspect[1][0] >= start_of_sentence and aspect[1][1] < end_of_sentence:
                    abstract = [sentence]
                    if ind != len(sentences) - 1:
                        abstract.append(sentences[ind+1])
                    abstract = ' '.join(abstract)
                    all_pairs.append((aspect[0], abstract))
                    first_aspect += 1
        return all_pairs

    def predict_classes(self, aspects, text):
        pairs = self.make_pairs(aspects, text)
        aspects = [pair[0] for pair in pairs]
        abstracts = [pair[1] for pair in pairs]
        self.model.to(self.device)
        inputs = self.tokenizer(aspects, abstracts, return_tensors="pt", padding='longest')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probabilities = sigmoid(logits).detach().cpu().numpy()

        return self.postprocess(probabilities)

    def postprocess(self, predictions):
        pred_classes = []
        for row in predictions:
            classes = np.where(row >= 0.5)[0]
            if 0 in classes and 2 in classes:
                pred_classes.append('both')
            else:
                pred_classes.append(self.id2tag[np.argmax(row)])
        return pred_classes
    
def main(args):
    print(args.aspect_model_path)
    aspect_model = AutoModelForTokenClassification.from_pretrained(args.aspect_model_path, local_files_only=True, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base')

    device = args.device

    tags = ['O']
    categories = ['Whole', 'Service', 'Food', 'Interior', 'Price']
    for category in categories:
        tags.append('B-'+category)
        tags.append('I-'+category)
    aspect_id2_label = {ind: tag for ind, tag in enumerate(tags)}

    aspect_pipeline = AspectExtractionPipeline(model=aspect_model, tokenizer=tokenizer, device=device, id2tag=aspect_id2_label)
    aspects = aspect_pipeline.predict_classes(args.text)

    sent_model = AutoModelForSequenceClassification.from_pretrained(args.sentiment_model_path, local_files_only=True, use_safetensors=True )
    sent_id2label = {0: 'neutral', 1: 'negative', 2: 'positive'}

    sentiment_pipeline = SentimentClassificationPipeline(model=sent_model, tokenizer=tokenizer, device=device, id2tag=sent_id2label)
    sentiment = sentiment_pipeline.predict_classes(aspects, args.text)
 
    result = []
    for ind in range(len(aspects)):
        result.append([aspects[ind][0], aspects[ind][2], sentiment[ind]])

    print(result)

if __name__ == '__main__':
    args = parse_args()
    main(args)