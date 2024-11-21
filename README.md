# Local Binary Patterns descriptor with Support Vector Machine, Random Forest and Multi-layer Perceptron classifiers to predict COVID-19 in x-ray pictures

## Table of Contents

1. [Descritor Implementado](#descritor-implementado)
2. [Repositório do Projeto](#repositório-do-projeto)
3. [Classificadores e Acurácia](#classificadores-e-acurácia)
   - [Multi-layer Perceptron](#multi-layer-perceptron)
     - [MLP Classifier Confusion Matrix com feature extraction do Gray Histogram](#mlp-classifier-confusion-matrix-com-feature-extraction-do-gray-histogram)
     - [MLP Classifier Confusion Matrix com Descritor LBP](#mlp-classifier-confusion-matrix-com-descritor-lbp)
   - [Random Forest](#random-forest)
     - [Random Forest Confusion Matrix com feature extraction do Gray Histogram](#random-forest-confusion-matrix-com-feature-extraction-do-gray-histogram)
     - [Random Forest Confusion Matrix com Descritor LBP](#random-forest-confusion-matrix-com-descritor-lbp)
   - [Support Vector Machine](#support-vector-machine)
     - [Support Vector Machine Confusion Matrix com feature extraction do Gray Histogram](#support-vector-machine-confusion-matrix-com-feature-extraction-do-gray-histogram)
     - [Support Vector Machine Confusion Matrix com Descritor LBP](#support-vector-machine-confusion-matrix-com-descritor-lbp)
   - [Acurácia dos modelos](#acurácia-dos-modelos)
     - [Acurácia dos modelos com feature extraction do Gray Histogram](#acurácia-dos-modelos-com-feature-extraction-do-gray-histogram)
     - [Acurácia dos modelos com Descritor LBP](#acurácia-dos-modelos-com-descritor-lbp)
4. [Setup](#setup)

## Descritor Implementado

O descritor LBP (Local Binary Pattern) é uma técnica popular de extração de características em visão computacional e análise de imagens. Desenvolvido para representar texturas em imagens, o LBP é particularmente eficaz em aplicações como reconhecimento facial, detecção de texturas e classificação de imagens.
A ideia fundamental por trás do LBP é capturar informações locais sobre a textura de uma imagem. Ele opera em nível de pixel, comparando o valor do pixel central com os valores dos pixels vizinhos ao redor. Para cada vizinho, o LBP atribui um bit (1 ou 0) dependendo se o valor do pixel é maior ou menor que o valor do pixel central. Isso gera um padrão binário local para cada região da imagem.
Esses padrões binários locais são então convertidos para uma representação decimal, criando um histograma de frequência. O histograma resultante é um vetor de características que descreve a distribuição dos padrões binários locais na imagem. Essa representação compacta é invariante a mudanças globais de iluminação e é robusta para variações locais na textura.
O LBP é conhecido por sua simplicidade, eficiência computacional e capacidade de capturar informações discriminativas sobre texturas. Ele tem sido amplamente utilizado em diversas aplicações, incluindo reconhecimento de objetos, segmentação de imagens e análise de texturas em imagens médicas. Devido à sua natureza robusta e eficácia em diferentes cenários, o descritor LBP continua sendo uma escolha valiosa na área de processamento de imagens e visão computacional.

## Repositório do Projeto
[LBP descriptor with SVM, RF and MLP classifiers](https://github.com/inteiros/LBP_ML)

## Classificadores e Acurácia

### Multi-layer Perceptron

#### MLP Classifier Confusion Matrix com feature extraction do Gray Histogram

<img src="results/mlp_classifier-04122023-1941.png" height="450px" width="600px" />

#### MLP Classifier Confusion Matrix com Descritor LBP 

<img src="results/mlp_classifier-04122023-2202.png" height="450px" width="600px" />

### Random Forest

#### Random Forest Confusion Matrix com feature extraction do Gray Histogram

<img src="results/rf_classifier-04122023-1941.png" height="450px" width="600px" />

#### Random Forest Confusion Matrix com Descritor LBP 

<img src="results/rf_classifier-04122023-2202.png" height="450px" width="600px" />

### Support Vector Machine

#### Support Vector Machine Confusion Matrix com feature extraction do Gray Histogram

<img src="results/svm_classifier-04122023-1941.png" height="450px" width="600px" />

#### Support Vector Machine Confusion Matrix com Descritor LBP 

<img src="results/svm_classifier-04122023-2202.png" height="450px" width="600px" />

### Acurácia dos modelos

#### Acurácia dos modelos com feature extraction do Gray Histogram

<img src="results/run_all_classifiers-04122023-1941.png" height="450px" width="600px" />

#### Acurácia dos modelos com Descritor LBP 

<img src="results/run_all_classifiers-04122023-2202.png" height="450px" width="600px" />

## Setup

Para o funcionamento deste projeto é necessario possuir [Python 3.10+](https://www.python.org/) instalado na sua maquina

Com isso em mente, primeiro instale as dependencias necessárias

```sh
pip install scikit-image
pip install sklearn
pip install Bar
pip install split-folders
pip install matplotlib
pip install opencv-python
```

então baixe o [dataset](https://www.kaggle.com/datasets/tarandeep97/covid19-normal-posteroanteriorpa-xrays) e extraia as pastas normal e covid em "images_full"

após a extração das imagens, abra o terminal no repositório e realize o data splitting em train e test com o comando

```sh
python data_splitting.py
```

feito isso, execute o descritor LBP para extração das features e labels (ou Gray Histogram Feature Extraction executando o arquivo grayHistogram_FeatureExtraction.py)

```sh
python localBinaryPattern_textureDescriptor.py
```

e com isso, podemos executar qualquer um dos classifiers para nosso dataset, individualmente ou simplesmente usar o seguinte comando para a execução de todos

```sh
python run_all_classifiers.py
```

após a execução os resultados estarão disponíveis na pasta results.
