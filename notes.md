# NLP notes

These are notes from learning NLP. It's mainly focused on practical side describing libraries like Spacy, platforms like NLTK and HuggingFace.

## Terms

Most important terms in NLP.

### Tokenizer

Breaks unstructured data and natural language text into chunks of information that can be considered as discrete elements - tokens.

### Stemming

In linguistic morphology and information retrieval, stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form—generally a written word form (usuwanie końcówek fleksyjnych sprowadzając wyraz do podstawowej formy). The stem need not be identical to the morphological root of the word; it is usually sufficient that related words map to the same stem, even if this stem is not in itself a valid root (tylko i wyłącznie obcinanie końcówek fleksyjnych słowa).

### Lemmatization

In linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form. Unlike stemming, lemmatisation depends on correctly identifying the intended part of speech and meaning of a word in a sentence, as well as within the larger context surrounding that sentence, such as neighboring sentences or even an entire document (zamiast usuwać końcówki, szukamy podstawowej formy; konieczne rozpoznanie o jakie słowo chodzi; przykładem jej słowo "mam", które może pochodzić od "mama", "mieć" lub "mamić").

### Morphological analysis

**Morphology** - **the study of words, how they are formed, and their relationship to other words in the same language**.
It analyzes the structure of words and parts of words such as **stems, root words, prefixes, and suffixes**.
Morphology also looks at **parts of speech, intonation and stress, and the ways context can change a word's pronunciation and meaning**.
Morphology differs from morphological typology, which is the classification of languages based on their use of words, and lexicology, which is the study of words and how they make up a language's vocabulary.

**Morfologia** – dziedzina lingwistyki zajmująca się **formami odmiennymi części mowy (fleksja) oraz słowotwórstwem**.
W podejściu morfemicznym elementarną jednostką analizy morfologicznej jest abstrakcyjny **morfem**, charakteryzowany zwykle jako **najmniejsza część wyrazu będąca nośnikiem znaczenia**.
Wyraz jest tu rozumiany jako znak prosty, jeśli złożony jest z jednego morfemu, albo znak złożony, jeśli można w nim wyróżnić więcej takich podstawowych jednostek.
W podejściach opartych o pojęcie morfemu przyjmuje się zwykle, że wszystkie morfemy, zarówno słowotwórcze, jak i fleksyjne, wchodzą w skład inwentarza leksykalnego użytkowników danego języka, zwanego zwykle **leksykonem**.

**Leksem** – wyraz rozumiany jako **abstrakcyjna jednostka systemu słownikowego** języka. Składa się na nią **znaczenie leksykalne** oraz **zespół wszystkich funkcji gramatycznych**, jakie dany leksem może spełniać, a także **zespół form językowych** reprezentujących w tekście leksem w jego poszczególnych funkcjach. Przykładowo wyrazy *czytać, czytam, czytali, przeczytasz* są formami tego samego leksemu. Pokrewnym pojęciem jest **lemma**, oznaczająca kanoniczną, **podstawową formę leksemu**, którą najczęściej podaje się w słownikach. Lemma może być reprezentowana przez jeden wyraz tekstowy.

**Analiza morfologiczna** - przypisanie analizowanemu słowu pewnych własnosci morfologicznych. Podejścia mogą się różnić w zależności od analizowanych cech ([link do materiałów](https://www.mimuw.edu.pl/~jsbien/publikacje/JSB-KS-PTJ01.pdf)).

### Semantic analysis

[Link](https://monkeylearn.com/blog/semantic-analysis/) to source of knowledge.

Semantic analysis is the process of **drawing meaning from text**. It allows computers to **understand and interpret sentences, paragraphs, or whole documents**, by analyzing their **grammatical structure**, and identifying **relationships between individual words in a particular context**.

Lexical semantics plays an important role in semantic analysis, allowing machines to understand relationships between lexical items:

- *Hyponyms*: specific lexical items of a generic lexical item (hypernym) e.g. orange is a hyponym of fruit (hypernym).
- *Meronomy*: a logical arrangement of text and words that denotes a constituent part of or member of something e.g., a segment of an orange
- *Polysemy*: a relationship between the meanings of words or phrases, although slightly different, share a common core meaning e.g. I read a paper, and I wrote a paper)
- *Synonyms*: words that have the same sense or nearly the same meaning as another, e.g., happy, content, ecstatic, overjoyed
- *Antonyms*: words that have close to opposite meanings e.g., happy, sad
- *Homonyms*: two words that are sound the same and are spelled alike but have a different meaning e.g., orange (color), orange (fruit)

Semantic analysis also takes into account signs and symbols (semiotics) and collocations (words that often go together).

#### Word Sense Disambiguation

The true meaning of the word depending of context. For example "orange" can refer to fruit or color.

#### Relation Extraction

This task consists of detecting the semantic relationships present in a text. For example, the phrase “Steve Jobs is one of the founders of Apple, which is headquartered in California” contains two different relationships: "Steve Jobs (person) founder of Apple (company)" and "Apple (company) headquartered in California (place)".

#### Semantic Classification Models

- Topic classification
- Sentiment analysis
- Intent classification - a form of text classification where message from client point what action he/she want to take (i.e. *Interested, Need Information, Unsubscribe, Wrong Person, Email Bounce, Autoreply*)

#### Semantic Extraction Models

Keyword extraction: finding relevant words and expressions in a text. This technique is used alone or alongside one of the above methods to gain more granular insights. For instance, you could analyze the keywords in a bunch of tweets that have been categorized as “negative” and detect which words or topics are mentioned most often.
Entity extraction: identifying named entities in text, like names of people, companies, places, etc. A customer service team might find this useful to automatically extract names of products, shipping numbers, emails, and any other relevant data from customer support tickets.

### Distribution hypotesis

*Linguistic items with similar distributions have similar meanings.*
Expressing words as vectors allows to better analysing text (sentiment, semantic meaning, ect.).
First aproach was to represent words as a matrix of co-occurence matrix (LSA and HAL techniques), then factorize it finding transformation matrix and multiply old by this one. Obtained matrix rows (or columns) contain vectors representing each word. This solution is expensive and not effective. Matrix is sparse, huge and factorization is computationally complex.
Second solution was to use simple neural network to create hidden representation: CBOW and Skip-gram. In both cases NN can give not best output and produce informative embedding.

#### Continuous Bag of Words (CBOW)

Input is *n* words (context) around one that has to be predicted. Output is one vector of probabilities for every word in dictionary.

#### Skip-gram

Input is a word from which context (other *n* words) has to be predicted. Output is *n* vectors for every analised position with probability of occurance every word from dictionary.

#### GloVe

[TowardsDS](https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010)
[YT](https://www.youtube.com/watch?v=InCWrgrUJT8&ab_channel=Scarlett%27sLog)

Based on matrix of co-occurence is created matrix of probabilities of occuring two words in given distance. Occurances are scaled based on distances (1->1, 2->1/2, ...).

#### Embedding from Language Models (ELMo)

[How ELMo works](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/).

Problem with CBOW and Skip-gram is that it represent different words with same spelling (i.e. orange - "color" and "fruit") using the same vector.
ELMo create embedding of words base of their context using Bi-LSTM. Sentence is passed to the model and for every word vectors from all layers are used to produce final embedding.
BERT embedding could be better, because it doesn't need to sum up vectors due to directions. It process all words frome sentence at once "including both directions".

### TFIDF

![tfidf](https://miro.medium.com/max/1200/1*V9ac4hLVyms79jl65Ym_Bw.jpeg)

### Named Entity Recognition (NER)

**Named entity** - phrase that clearly identifies one item from a set of other items that have similar attributes. Examples of named entities are first and last names, geographic locations, ages, addresses, phone numbers, companies and addresses.

Model returns *m* logits per word, describing how likely is that it belongs to one of *m* NER class (i.e. next tuesday - date).
Classes assigned to tokens describe not only if token is part of named entity, but sometimes also if it is a start of another named entity (i.e. B-ORG - beginning of an organization right after another organization).

### Question Answering

[Link](https://www.deepset.ai/blog/modern-question-answering-systems-explained) to explanation.

System that select range of words of text (context) based on query (question).
The input of model is concatenated question and context with special signs between.
Model returns two tensors of logits describing probability of position to be start and end of selection.

### Text Summarization

[Link](https://medium.com/luisfredgs/automatic-text-summarization-with-machine-learning-an-overview-68ded5717a25) to knowledge.

Summarization is the task of condensing a piece of text to a shorter version, reducing the size of the initial text while at the same time preserving key informational elements and the meaning of content.
Text summarization can be done by selecting part of text which contains most relevant informations.
It can be also done by shorter text generation.

#### The Extractive Approach

You can select most salient *k* sentences by scoring them with TF metric.
Other option is to use NN (perhaps like in *question answering* task.)

#### Abstractive summarization

This approach is to generate summary. Since output is not only selected sentences, it is more complex.

### Zero Shot Learning

[Link](https://www.kdnuggets.com/2021/04/zero-shot-learning.html) to knowledge.

Model recognise unseen target classes that have not labelled for training.
Zero-shot techniques associate observed and non observed classes through some form of so-called "auxiliary" information, that encodes distinguishing properties of objects. That has been a very popular technique in computer vision for long, that is now more and more used in Natural Language Processing.
![ZSL](https://miro.medium.com/max/1400/1*6I13f85d0Y8auK5316xwMw.png)
In ZSL model produce semantic embedding (e.i. tail, fur, feathers, hooves, paws). You give categories that model should consider and it calculate relevance (probabilities) of them based on semantic vectors. You can also try multiclass categorization and it will give you probabilities that don't sum up to 1.

The zero-shot text classification model is trained on Natural Language Interface (NLI). Text classification is the process of categorizing the text into a set of words. By using NLI, text classification can automatically perform text analysis and then assign a set of predefined tags or categories based on its context.

### Table Question Answering

Refers to providing precise answers from tables (could be sql databases) to answer a user's question.

### Sentence Similarity

Asking model for 2 sentences embeddings and calculating their cosine similarity.
Sentence embeddign can be obtain by summing up or averaging senctence words vectors. Other approach is to use SentenceBERT ([link](https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/) to simple explanation).

## PyDictionary

Library to get meanings, translations, synonyms and Antonyms of words.

## Spacy

Spacy library provides many nlp features that are user-friendly. It contains ready to use pipelines (in nltk you have to do it manually).

### Containers

Spacy creates objects called containers containing text and infos about it:

- Doc
- DocBin
- Example
- Language
- Lexeme
- Span
- SpanGroup
- Token

Doc container holds all text. It's main object that has different attributes and subcontainers.
Doc.sents (generator) contains all segmented sentences.
Span containers are slices of doc. They can store group of tokens. Doc sentence is also a span.
Span can be grouped in SpanGroups.

### Part of speech (POS)

**Tagger** classify token as one of the following types:

- ADJ: adjective, e.g. big, old, green, incomprehensible, first
- ADP: adposition, e.g. in, to, during
- ADV: adverb, e.g. very, tomorrow, down, where, there
- AUX: auxiliary, e.g. is, has (done), will (do), should (do)
- CONJ: conjunction, e.g. and, or, but
- CCONJ: coordinating conjunction, e.g. and, or, but
- DET: determiner, e.g. a, an, the
- INTJ: interjection, e.g. psst, ouch, bravo, hello
- NOUN: noun, e.g. girl, cat, tree, air, beauty
- NUM: numeral, e.g. 1, 2017, one, seventy-seven, IV, MMXIV
- PART: particle, e.g. ’s, not,
- PRON: pronoun, e.g I, you, he, she, myself, themselves, somebody
- PROPN: proper noun, e.g. Mary, John, London, NATO, HBO
- PUNCT: punctuation, e.g. ., (, ), ?
- SCONJ: subordinating conjunction, e.g. if, while, that
- SYM: symbol, e.g. $, %, §, ©, +, −, ×, ÷, =, :), 😝
- VERB: verb, e.g. run, runs, running, eat, ate, eating
- X: other, e.g. sfpksdpsxmsa
- SPACE: space, e.g.

### Sentence diagram (rozbiór logiczny zdania)

[How it works](https://grammar.yourdictionary.com/sentences/diagramming-sentences.html)
[About dependency parsing](https://web.stanford.edu/~jurafsky/slp3/14.pdf)

A root of the sentence is verb.

## Hugging Face

A community and data science platform that provides: Tools that enable users to build, train and deploy ML models based on open source (OS) code and technologies.

### Tokenization

There are 3 types of tokenizers:

- word tokenizers - because of huge amount of words should be limited by eliminating less frequent words
- character tokenizers - tokenizes characters instead of words
- subwords tokenizers - tokenize parts of words (i.e. token - ization)

In tokenization process special tokens like start of sentence [CLS] are added.

**Fast tokenizers** - tokenizers implemented in Rust instead of Python.

## Bonus

Extra libraries, tips and insights that could be useful.

## BertViz

![image](https://github.com/jessevig/bertviz/raw/master/images/model-view-noscroll.gif)

## Does BERT need text preprocessing?

Mostly not. BERT implementation include actions like:

1. stemming or lemmatization : Bert uses BPE (Byte- Pair Encoding to shrink its vocab size), so words like run and running will ultimately be decoded to run + ##ing. So it's better not to convert running into run because, in some NLP problems, you need that information.
2. De-Capitalization - Bert provides two models (lowercase and uncased). One converts your sentence into lowercase, and others will not change related to the capitalization of your sentence. So you don't have to do any changes here just select the model for your use case.
3. Removing high-frequency words - Bert uses the Transformer model, which works on the attention principal. So when you finetune it on any problem, it will look only on those words which will impact the output and not on words which are common in all data.

[Link](https://stackoverflow.com/questions/63979544/using-trained-bert-model-and-data-preprocessing) to stackoverflow answer.
