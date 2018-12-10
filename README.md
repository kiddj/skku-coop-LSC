# 성균관대학교 산학협력프로젝트 LSC Systems

## Introduction

### 1. 목표
산학협력프로젝트를 진행하면서 Fraud Detection System에 사용될 수 있는 기법들을 조사하고, 구현을 통해 직접 성능 검증 시도

### 2. 성능 측정
- **데이터:** Kaggle의 'Credit Card Fraud Detection'에서 주어지는 데이터를 이용
https://www.kaggle.com/mlg-ulb/creditcardfraud

- **평가척도:** Confusion Matrix

## KNN

설명 추가 필요

## SVM (Support Vector Machine)

설명 추가 필요

## Decision Tree

설명 추가 필요

## One Class Neural Network (OC-NN)
https://github.com/kiddj/skku-coop-LSC/tree/master/OC-NN

Autoencoder를 이용하여 주요 feature들을 추출한 후에, 해당 feature들을 한 개의 hidden layer를 가지는 Nerual Net에 입력하여 해당 값이 Fraud인지 판별.

이 때, Neural Net의 Loss Function을 OCSVM-like하게 설정하여 OCSVM보다 유동적으로 데이터를 판별할 수 있을 것이라 기대

## Autoencoder

설명 추가 필요