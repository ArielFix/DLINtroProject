# Disaster-Related Twitts Detection
# Theory
In this document, we will try and explain the theory behind the methods we used to perform the task.
It will not go into things that are covered by the course (Intro to Deep Learning), with the exception of justifying our choices.

## The task:
Clasiffying "tweet"s , short text public broadcast messages on the platform "twitter", as a report of a disaster. The phenomenon of people using twitter in a time of crisis is recorded in the given [dataset](https://www.kaggle.com/competitions/nlp-getting-started/overview). Our task is to classify a tweet as either:
 `0` - not disaster related or,
 `1` - disaster related

The task can be considered compplex of the following reasons:
* Unstructured Data: The data mostly cosists of free-form text that can contain misspelled words, slang, and emoticons. It can also contain format specific "tokens" like hashtags, mentions and links.
* Imbalanced Classes: In the dataset there is a, slight but noticable, bias towards "non-disaster" tweets, This is a common possible pitfall in model training which needs to be addressed.
* Ambiguous and misleading information: Tweets can contain misleading or ambiguous information, which makes it challenging for the model to accurately classify them as disaster-related or not. For example, a tweet may contain disaster-related keywords but not actually be related to a disaster.

The task falls under the umbrella of "Sentiment Analysis". Which complicates things, due to the complex nature of human emotion, text - especially informal text - can be highly subjective, and the meaning can change based on factor external to the text - like the platform it was meant for.


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
We decided to use a public model trained from RoBERTa on tweets, this model was fine-tuned on sentiment analysis of tweets. Using a model trained on the tweet format may help improve the result since it is familiar with the "dialect" used on Twitter - phrases and structural changes to language that emerge due to online culture and the format - and social nature - of the application.

# Implementation

## Pre-Processing
An important part of any data-based project is preprocessing. In this stage, the data is transformed into a representation that can be used in the chosen model. In this task the preoprocessing can be separated into two groups:
* Dealing with special cases
* Augmentation

### Special Cases
There are multiple special cases we needed to deal with the list contains but isn't limited to:
1. Links
2. Twitter specific tokens: (#,@)
3. Abbreviations

Dealing with those cases ensures that the model will run smoothly, and in the case of last one may also imrove the results
### Augmentation
Augmentation in short, is creating more data from the existing data. While on the surface this practice seems dubious, in practice it can improve the performance of a model.\
Even though no information is added to the dataset, the transofrmation to a different but similar representation of the data can be usefull, helping the model converge faster and can prevent overfitting.\
It is important to note that overdoing augmentations can have a negative impact on the end-result.\
Augmentation methods have to preserve the information of the data, which begs the question: How do you change a sentence without changing it's meaning.

We have used multipile methods the two most notable are: `Back translate` and `Synonym Replace`.

`Back translate` is the process of translating the sentence from one language to the other and back. e.g. the sentence:

>This project is part of the introduction to deep learning course at Ben Gurion University.

is translated to hebrew by google translate to:
>פרויקט זה הוא חלק מקורס מבוא ללמידה עמוקה באוניברסיטת בן גוריון.

and when translated back:
> This project is part of an introductory deep learning course at Ben Gurion University.

We can see how the method subtly changes the data in a way that preserves the original meaning but presents it in a different way.

`Synonym Replace` as the name suggest is switching a word in the sentence with another word with the same meaning:

> I got perfect score on the project in the Intoduction to Deep Learning course

Will become:

> I got perfect marks on the project in the Intoduction to Deep Learning class

(*)  I took some liberty with the grammar in the example, please excuse me - and I'd like to note that this method does create instances where the grammar is botched , like this one. This is an unintentional but , at least in our case, positive side-effect of the method. Since grammar is not essential to the delivery of sentiment in the vast majority of cases.

## LSTM
### Model Architecture:
* **Embedding layer**: The first layer of the model is an Embedding layer. This layer maps each word in the input sequence to a dense vector of fixed size, called the embedding vector. The `embed_dim` argument specifies the size of the embedding vector, in this case, it is `32`. The `input_length` argument specifies the length of the input sequence, in this case, it is `X.shape[1]`. The `max_features` argument specifies the number of unique words in the vocabulary, which is used to initialize the weights of the embedding layer.

* **Dropout layer**: The second layer of the model is a Dropout layer. This layer randomly sets input units to `0` with a frequency of `0.2` at each step during training time, which helps prevent overfitting.

* **LSTM layer**: The third layer of the model is an LSTM layer. This layer is a type of Recurrent Neural Network (RNN) that is well-suited for processing sequences of data. The `lstm_out` argument specifies the size of the output of the LSTM layer, in this case, it is `32`. The dropout and `recurrent_dropout` arguments specify the dropout rates for the LSTM layer.

* **Dense layer**: The fourth and final layer of the model is a Dense layer with one output node and a sigmoid activation function. This layer is used to make the binary classification by producing a probability value between `0` and `1`, where values close to `0` correspond to class `0` (not disaster-related) and values close to `1` correspond to class `1` (disaster-related).
