# Neural Network Acceleration & Compression Experimentations

## Table of Contents
1. Abstract
2. Project Objective & Goals
    <br>2.1 Proofpoint
3. Methodology
    <br>3.1 Experimentation Set Up
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

### 3.1 Experimentation Set Up

### 3.2 Methods

#### 3.2.1 Pruning

#### 3.2.2 Quantization

#### 3.2.3 Network Exchange

#### 3.2.3 Knowledge Distillation

## 4. Results

## 5. Conclusion

## 6. Usage Instructions

## 7. Contributors