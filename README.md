# Fake News Detection using Deep Nerual Networks

This project is part of a challenge which can be found at http://www.fakenewschallenge.org/
### By Vyas Anirudh Akundy & Atish Harish Telang Patil

![Fake News](https://media.giphy.com/media/3ohzdJeMka5heqSbZu/giphy.gif)

- Fake news is a nagging annoyance these days which can lead to the spread of misleading and fabricated information. The task of assessing the veracity of a news article is a complicated one.
- However, a first step that can be taken in identifying fake news is to detect the stance of two pieces of text, i.e estimating the relative perspective of two pieces of text
# We use Natural Language Processing and Machine Learning to tackle this problem.
- We have improved upon a previously developed baseline model whose repository can be found [here](https://github.com/FakeNewsChallenge/fnc-1-baseline)

## BERT(Bidirectional Encoder Representations from Transformers) language model was used to embed the text data into vector representations
- We have used **bert-as-service** to obtain the embeddings which can be found [here](https://raw.githubusercontent.com/hanxiao/bert-as-service/master/.github/demo.gif)
![image](https://raw.githubusercontent.com/hanxiao/bert-as-service/master/.github/demo.gif)

The image below, taken from the official website, clearly illustrates the task. A news headline and article are taken and the relation between them is classified into 4 classes -- (unrelated, agrees, disagrees, discusses)
![image](https://raw.githubusercontent.com/Anirudh42/MSCI641-Project/master/task.png)

## The neural network architecture we built is shown below

![image](https://raw.githubusercontent.com/Anirudh42/MSCI641-Project/master/NNarchitecture.png)

