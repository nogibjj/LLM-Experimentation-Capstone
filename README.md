# Neural Network Acceleration & Compression Experimentations

## Table of Contents
1. Abstract
2. Project Objective & Goals
    <br>2.1 Proofpoint
3. Methodology
    <br>3.1 Experimentation Set Up
    <br>.   3.1.1 Evaluation Metrics
    <br>.   3.1.2 Experimentation Configurations
    <br>3.2 Methods
    <br>.   3.2.1 Pruning
    <br>.   3.2.2 Quantization
    <br>.   3.2.3 Network Exchange
    <br>.   3.2.4 Knowledge Distillation
4. Results
5. Conclusion
6. Usage Instructions
7. Contributors

## 1. Abstract
Our project addressed the significant challenge of reducing computational and storage costs associated with the deployment of large language models (LLMs). Our objective was to achieve cost-efficiency by minimizing the model size and inference speed without compromising their performance capabilities. This pursuit was motivated by the pressing need to manage the considerable expenses tied to operating LLMs, as exemplified by the substantial daily costs incurred by similar operations at scale.

Our solution involved a meticulous experimental design that leveraged state-of-the-art compression techniques, including pruning, quantization, and knowledge distillation, applied both individually and in combination. We focused our efforts on BERT and a 1B-LLM model, utilizing multilabel and multiclass classification datasets to evaluate our strategies' effectiveness. We achieved up to a **2.35** times increase in inference speed and a **6.64** times reduction in model size, with minimal impact on accuracy. Through detailed documentation and the development of optimization scripts for cross-platform interoperability, we have laid a foundation for future advancements and broader applicability of our findings.

## 2. Project Objective & Goals
Large language models have shown remarkable effectiveness in understanding and classifying textual content to identify potential cyber threats within the past few years. However, these models come with significant computational and storage overhead, and finding compression and fine-tuning techniques to reduce the scale and computational cost has become a key challenge for companies to overcome. Therefore, the primary problem we aim to solve is **how can we reduce the computational and storage requirements of LLMs by using neural network compression and acceleration techniques without significantly compromising their performance**?

We aim to make the process of inference faster and more efficient by leveraging recent strides in LLM research to develop models that are smaller in size yet proficient in performance. Finding such methods requires us to perform experimentations on our chosen models using techniques such as pruning, distillation, and quantization individually and in combination. 

### 2.1 ProofPoint
This project was conducted in collaboration with Proofpoint Inc. Proofpoint is a leading cybersecurity company in the US that utilizes natural language and machine learning techniques to detect and mitigate cyberthreats in various communication channels. We developed an interface that Proofpoint may leverage internally to apply on their internal models. However, this library can be utilized by anyone who wishes to deploy large language models at relatively lower costs. 

## 3. Methodology
Our experimentation scope is focused on a few network acceleration techniques, specifically in quantization, pruning, and distillation methods. These are techniques of our specific interest as they can be leveraged independently to compress the network, but also in combination with each other to assess synergistic effects on latency and model size. 

We performed our experiments on a smaller model (BERT) and a larger model (TinyLlama-1.1B) to evaluate the generalizability of our methods.

### 3.1 Experimentation Set Up
We utilized each technique independently and then in combination with each other to evaluate the method's effect on the model **inference speed**, **model size**, and **accuracy**. 

#### 3.1.1 Evaluation Metrics
The primary objective of our study is to identify methods that effectively reduce the latency and size of models. Accordingly, the principal metrics adopted for assessing the efficacy of our techniques included average inference time (measured in milliseconds) and model size (measured in megabytes). It is widely acknowledged that optimizing a LLM for reduced size or enhanced speed often involves a trade-off with model accuracy. Therefore, accuracy (measured in percentage) was monitored as a secondary metric in our experiments to ascertain that the integrity of model performance remained largely intact following the application of our methods. Additionally, we documented the training time needed for fine-tuning the model in techniques involving pruning and distillation, although no specific expectations were established for this metric.

#### 3.1.2 Experimentation Configurations
There were a number of configurations we had to account for in our experimentations: machine, compression methods applied, and task. Due to computational limitations, we weren't able to harmonize all of the configurations across the two models. However, since we are interested in the relative gains of each compression experiment, we can still compare results within the model giving us good insight into the capabilities of each methods. 

Both models were fine-tuned to perform emotions classification on textual content. The task chosen for the BERT model was Emotion Classification of Tweets which was a multi-label classification problem. The dataset had 11 different emotion labels with around ~11k records. Fine-tuning was performed for 5 epochs with a 60-30-10 train-test-validation split. The task chosen for the TinyLlama model was an Emotion Classification of Digital Conversations which was a multiclass classification problem. The dataset had 7 different emotion labels with around ~40k records. All the experimentations were performed on a T4 GPU. 

![High-level overview of our experimentation approach and configurations](image.png)

### 3.2 Methods

#### 3.2.1 Pruning

#### 3.2.2 Quantization

#### 3.2.3 Network Exchange

#### 3.2.3 Knowledge Distillation

## 4. Results

## 5. Conclusion

## 6. Usage Instructions

## 7. Contributors