from transformers import AutoTokenizer
import pandas as pd



class Utilities:
    def preprocess_text_line(text):

        """
        pre-processing a string:
        
        input:  text (string) 
        output text (strung) 
        """

        new_text = []
 
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)

        return " ".join(new_text)

    def preprocess_text_list(self, text):
        """
        pre-processing on a given list of string - performed in place:
        
        input:  text (list(string)) 
        """

        for line in range(0, len(text)):
            new_text = []
            twit = text[line]

            for t in twit.split(" "):
                t = '@user' if t.startswith('@') and len(t) > 1 else t
                t = 'http' if t.startswith('http') else t

                new_text.append(t)

            text[line] = " ".join(new_text)

    def tokenize_data(self, data, tokenizer):
        return tokenizer(data, return_tensors='pt', padding=True)

    def preprocess_and_tokenized_text_list(self, tokenizer, text):
        text.dropna(axis=0, inplace=True)
        text.reset_index(drop=True, inplace=True)
        t = text['text'].to_list()
        self.preprocess_text_list(t)
        text['text'] = self.tokenize_data(t, tokenizer)
        return text


