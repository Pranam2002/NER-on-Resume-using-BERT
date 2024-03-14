# NER-on-Resume-using-BERT

## Bidirectional Encoder Representations from Transformers:


### About BERT:
BERT is an open source machine learning language model for natural language processing (NLP). BERT is designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context. The BERT framework was pre-trained using text from Wikipedia and can be fine-tuned with question and answer datasets.

BERT, which stands for Bidirectional Encoder Representations from Transformers, is based on Transformers, a deep learning model in which every output element is connected to every input element, and the weightings between them are dynamically calculated based upon their connection. (In NLP, this process is called attention.)

BERT is a deep bidirectional, unsupervised language representation, pre-trained using a plain text corpus. BERT is a neural-network-based technique for language processing pre-training.

Historically, language models could only read text input sequentially -- either left-to-right or right-to-left -- but couldn't do both at the same time. BERT is different because it is designed to read in both directions at once. This capability, enabled by the introduction of Transformers, is known as bidirectionality. 

Using this bidirectional capability, BERT is pre-trained on two different, but related, NLP tasks: Masked Language Modeling and Next Sentence Prediction.

The objective of Masked Language Model (MLM) training is to hide a word in a sentence and then have the program predict what word has been hidden (masked) based on the hidden word's context. The objective of Next Sentence Prediction training is to have the program predict whether two given sentences have a logical, sequential connection or whether their relationship is simply random.

### Transformers:
Transformers were first introduced by Google in 2017. At the time of their introduction, language models primarily used recurrent neural networks (RNN) and convolutional neural networks (CNN) to handle NLP tasks.

Although these models are competent, the Transformer is considered a significant improvement because it doesn't require sequences of data to be processed in any fixed order, whereas RNNs and CNNs do. Because Transformers can process data in any order, they enable training on larger amounts of data than ever was possible before their existence. This, in turn, facilitated the creation of pre-trained models like BERT, which was trained on massive amounts of language data prior to its release. 

### Background:
In 2018, Google introduced and open-sourced BERT. In its research stages, the framework achieved groundbreaking results in 11 natural language understanding tasks, including sentiment analysis, semantic role labelling, sentence classification and the disambiguation of polysemous words, or words with multiple meanings.

Completing these tasks distinguished BERT from previous language models such as word2vec and GloVe, which are limited when interpreting context and polysemous words. BERT effectively addresses ambiguity, which is the greatest challenge to natural language understanding according to research scientists in the field. It is capable of parsing language with a relatively human-like "common sense".

In October 2019, Google announced that they would begin applying BERT to their United States based production search algorithms. In December 2019, BERT was applied to more than 70 different languages.

### How BERT works?
The goal of any given NLP technique is to understand human language as it is spoken naturally. In BERT's case, this typically means predicting a word in a blank. To do this, models typically need to train using a large repository of specialized, labeled training data.

BERT, however, was pre-trained using only an unlabelled, plain text corpus (namely the entirety of the English Wikipedia, and the Brown Corpus). It continues to learn unsupervised from the unlabeled text and improve even as its being used in practical applications (ie Google search). Its pre-training serves as a base layer of "knowledge" to build from. From there, BERT can adapt to the ever-growing body of searchable content and queries and be fine-tuned to a user's specifications. This process is known as transfer learning.

BERT is made possible by Google's research on Transformers. The transformer is the part of the model that gives BERT its increased capacity for understanding context and ambiguity in language. The transformer does this by processing any given word in relation to all other words in a sentence, rather than processing them one at a time. By looking at all surrounding words, the Transformer allows the BERT model to understand the full context of the word, and therefore better understand searcher intent.

BERT uses a method of masked language modeling to keep the word in focus from "seeing itself" -- that is, having a fixed meaning independent of its context. BERT is then forced to identify the masked word based on context alone. In BERT words are defined by their surroundings, not by a pre-fixed identity.

BERT is also the first NLP technique to rely solely on self-attention mechanism, which is made possible by the bidirectional Transformers at the center of BERT's design. This is significant because often, a word may change meaning as a sentence develops. Each word added augments the overall meaning of the word being focused on by the NLP algorithm. The more words that are present in total in each sentence or phrase, the more ambiguous the word in focus becomes. 

### BERT Architecture
We primarily report results on two model sizes: 
+ BERTBASE (L=12, H=768, A=12, Total Parameters=110M) 
+ BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M).


### What is BERT used for?
BERT is currently being used at Google to optimize the interpretation of user search queries. BERT excels at several functions that make this possible, including:  

-  Sequence-to-sequence based language generation tasks such as:
+ Question answering
+ Abstract summarization
+ Sentence prediction
+ Conversational response generation  
-  Natural language understanding tasks such as:
+ Polysemy and Coreference (words that sound or look the same but have different meanings) resolution
+ Word sense disambiguation
+ Natural language inference
+ Sentiment classification

BERT is expected to have a large impact on voice search as well as text-based search, which has been error-prone with Google's NLP techniques to date. BERT is also expected to drastically improve international SEO, because its proficiency in understanding context helps it interpret patterns that different languages share without having to understand the language completely.

BERT is open source, meaning anyone can use it. Google claims that users can train a state-of-the-art question and answer system in just 30 minutes on a cloud tensor processing unit (TPU), and in a few hours using a graphic processing unit (GPU). Many other organizations, research groups and separate factions of Google are fine-tuning the BERT model architecture with supervised training to either optimize it for efficiency (modifying the learning rate, for example) or specialize it for certain tasks by pre-training it with certain contextual representations. 

### Why is BERT important?
BERT converts words into numbers. This process is important because machine learning models use numbers, not words, as inputs. This allows you to train machine learning models on your textual data. That is, BERT models are used to transform your text data to then be used with other types of data for making predictions in a ML model.

### Can BERT be used for topic modelling?
Yes. BERTopic is a topic modelling technique that uses BERT embeddings and a class-based TF-IDF to create dense clusters, allowing for easily interpretable topics while keeping important words in the topic descriptions.



## Bert vs Other Technologies & Methodologies:


### BERT vs GPT
Along with GPT (Generative Pre-trained Transformer), BERT receives credit as one of the earliest pre-trained algorithms to perform Natural Language Processing (NLP) tasks.

Below is a table to help you better understand the general differences between BERT and GPT.

BERT is encoder-only architecture.	GPT is decoder-only architecture.
Bidirectional. Can process text left-to-right and right-to-left. BERT uses the encoder segment of a transformation model.	Autoregressive and unidirectional. Text is processed in one direction. GPT uses the decoder segment of a transformation model.
Applied in Google Docs, Gmail, smart compose, enhanced search, voice assistance, analyzing customer reviews, and so on.	Applied in application building, generating ML code, websites, writing articles, podcasts, creating legal documents, and so on.
GLUE score = 80.4% and 93.3% accuracy on the SQUAD dataset.	64.3% accuracy on the TriviaAQ benchmark and 76.2% accuracy on LAMBADA, with zero-shot learning
Uses two unsupervised tasks, masked language modeling, fill in the blanks and next sentence prediction e.g. does sentence B come after sentence A?	
Straightforward text generation using autoregressive language modeling.

A technical comparison of a decoder-only vs encoder-only architecture is like comparing Ferrari vs Lamborghini — both are great but with completely different technology under the chassis.



### BERT vs transformer
BERT uses an encoder that is very similar to the original encoder of the transformer, this means we can say that BERT is a transformer-based model.



### BERT vs word2vec
Consider the two examples sentences

“We went to the river bank.”
“I need to go to the bank to make a deposit.”
Word2Vec generates the same single vector for the word bank for both of the sentences. BERT will generate two different vectors for the word bank being used in two different contexts. One vector will be similar to words like money, cash, etc. The other vector would be similar to vectors like beach and coast.



### BERT vs RoBERTa
Compared to RoBERTa (Robustly Optimized BERT Pretraining Approach), which was introduced and published after BERT, BERT is a significantly undertrained model and could be improved. RoBERTa uses a dynamic masking pattern instead of a static masking pattern. RoBERTa also replaces next sentence prediction objective with full sentences without NSP.



## Language models
A language model is a type of machine learning model trained to conduct a probability distribution over words. Language modeling is the task of predicting what word comes next. More formally: given a sequence of words x(1),x(2), …x(t), compute the probability distribution of the next word x(t+1). A system that does language modeling is called a Language Model.You can also think of a Language Model as a system that assigns a probability to a piece of text. For example, if we have some text x(1),x(2), …x(t), then the probability of this text (according to the Language model is) is shown on the left picture. Language Modeling is a benchmark task that helps us measure our progress in understanding language. Language Modeling is a subcomponent of many NLP tasks, especially those involving generating text or estimating the probability of text: predictive typing, speech recognition, handwriting recognition, spelling/grammar correction, authorship identification, machine translation, summarization, dialogue Etc. Language models form the backbone of Natural Language Processing.

### RNN:
Advantages: it can process any length input; computation for step t can (in theory) use information from many steps back; model size doesn’t increase for longer input; same wights applied on every timestep, so there is symmetry in how inputs are processed. 

Disadvantages: recurrent computation is slow; in practice, difficult to access information from many steps back.

How to train an RNN LM? 

First, we need to get a big corpus of text which is a sequence of words x¹,x²,x³,…. Then we feed these inputs into RNN-LM and compute output distribution y_hat(t) for every step t. However, computing loss and gradients across the entire corpus are too expensive. Stochastic Gradient Descent (SGD) allows us to compute loss and gradient for a small chunk of data, and update. So we could apply SGD into computing loss for a sentence (actually a batch of the sentence), compute gradients and update weights and repeat this process.

### Long short-term memory (LSTM):
How to fix the vanishing gradient problem? The main problem is that it’s too difficult for the RNN to learn to preserve information over many timesteps. In a vanilla RNN, the hidden state is constantly being rewritten. How about an RNN with separate memory? The LSTM can erase, write, and read information from the cell. The selection of which information is erased/written read is controlled by three corresponding gates. The gates are also vectors length n. On each time step, each element of gates can be open (1), closed(0), or somewhere in-between. The gates are dynamic: their value is computed based on the current context. How does LSTM solve vanishing gradient? The LSTM architecture makes it easier for the RNN to preserve information over many timesteps. 

Advantages: LSTMs can capture long-term dependencies and handle sequential data well.

Disadvantages: LSTMs can be computationally expensive and require a large amount of training data.

### Amazing facts:

+ How long does it take to pre-train BERT? The 2 original BERT models were trained on 4(BERTbase) and 16(BERTlarge) Cloud TPUs for 4 days.

+ Google has been using your reCAPTCHA selections to label training data since 2011. The entire Google Books archive and 13 million articles from the New York Times catalog have been transcribed/digitized via people entering reCAPTCHA text. Now, reCAPTCHA is asking us to label Google Street View images, vehicles, stoplights, airplanes, etc.

+ How long does it take to fine-tune BERT?

For common NLP tasks discussed above, BERT takes between 1-25mins on a single Cloud TPU or between 1-130mins on a single GPU.


### References:
https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model#:~:text=BERT%2C%20which%20stands%20for%20Bidirectional,calculated%20based%20upon%20their%20connection.

https://h2o.ai/wiki/bert/#:~:text=BERT%2C%20short%20for%20Bidirectional%20Encoder,framework%20for%20natural%20language%20processing.

https://jalammar.github.io/illustrated-bert/

https://medium.com/@rachel_95942/language-models-and-rnn-c516fab9545b

https://typeset.io/questions/what-are-the-advantages-and-disadvantages-of-using-lstms-for-29rc9pw9zp

https://arxiv.org/pdf/1810.04805.pdf

https://arxiv.org/pdf/1706.03762.pdf

https://arxiv.org/pdf/1904.08398.pdf



### Implementing Guide:
https://www.analyticsvidhya.com/blog/2023/06/step-by-step-bert-implementation-guide/

https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891

 https://thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python

https://www.analyticsvidhya.com/blog/2022/01/building-language-models-in-nlp/#:~:text=A%20language%20model%20in%20NLP,appear%20next%20in%20the%20sentence.



### Code Reference:
https://github.com/google-research/bert

https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
