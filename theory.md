# Disaster-Related Twitts Detection - Theory
In this document, we will try and explain the theory behind the methods we used to perform the task.
It will not go into things that are covered by the course (Into to Deep Learning), with the exception of justifying our choices.

## RNN
One of the methods we have decided to test is using an RNN to solve the problem. The most important reason for that is: text is ordered, and the meaning of a character, word, or sentence changes depending on the context.

for example, the word "bank" could mean:
*  "a financial establishment that invests money deposited by customers, pays it out when required, makes loans at interest, and exchanges currency".
*  "the land alongside or sloping down to a river or lake."
  
which are two very different meanings of the word.\
RNNs are designed to learn the state of the sequence, which in text tasks would be the context of the word/sentence.\

For the specific variation of RNN, LSTM seems to be the "go-to" approach as the context of a specific word could vary depending on distant words. 

A limitation of RNNs for the use of text-based tasks is the non-causal nature of text. RNNs use the past to evaluate the state while the context of a word may be dependent on "future" words.

## BERT
**B**idirectional **E**ncoder **R**epresentations from **T**ransformers, is a transformer designed - by google - to generate a generic representation of a text by evaluating the context of previous and next words. This approach aims to create a model that can be fine-tuned with ease to a wide variety of specific tasks.

BERT was trained using two methods:
* MLM - **M**asked **L**anguage **M**odel.
* NSP - **N**ext **S**entence **P**rediction.

MLM - The model is trained by removing a percentage of the input and replacing it with a [MASK] token. The model then tries to predict the missing words.

NSP - The model is given a pair of sequences and tries to predict if the second sentence is the next sequence in the document, or from an unrelated place in the corpus (collection of documents).

## RoBERTa
**R**obustly **O**ptimized **BERT** pre-training **A**pproach - a largely successful attempt to improve on bert. The most interesting differences are the dropping of NSP stage of training, as it was observed not to impact the results,  and dynamic masking - On training BERT a static method was used where the same places in the input sequence were masked.

It is important to note that RoBERTa was also trained on a larger dataset.

##  twitter-roberta-base-sentiment
We decided to use a public model trained from RoBERTa on tweets, this model was fine-tuned on sentiment analysis of tweets. Using a model trained on the tweet format may help improve the result since it is familiar with the "dialect" used on Twitter - phrases and structural changes to text that appear due to online culture and the format - and social nature - of the application.