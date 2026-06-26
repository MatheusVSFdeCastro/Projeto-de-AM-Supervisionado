# 🧠 Medical Image Segmentation with U-Net

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Linux](https://img.shields.io/badge/Linux-Ubuntu-purple.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue.svg)

Este repositório contém um pipeline completo de Deep Learning Supervisionado para a **segmentação pixel a pixel de estruturas anatômicas/lesões em imagens médicas (Tomografias/Ressonâncias)**. O projeto aborda desde a ingestão automatizada de dados via API até o treinamento de uma arquitetura de rede neural convolucional avançada (**U-Net**).

---

## 📌 Visão Geral do Problema

Diferente da classificação tradicional de imagens (que gera apenas um rótulo global), a **segmentação biomédica** exige uma classificação a nível de pixel. O modelo recebe uma imagem médica bruta e gera uma máscara binária delimitando com precisão milimétrica a região de interesse (lesão/órgão).

* **Dataset:** `a0-2025-medical-image-segmentation` (Kaggle)
* **Entrada ($X$):** Imagens de exames médicos estruturadas em matrizes numéricas normalizadas.
* **Saída ($Y$):** Máscaras de segmentação (*Ground Truth*) criadas por especialistas médicos.

---

## 🛠️ Arquitetura do Repositório e Tecnologias

O projeto foi desenvolvido utilizando o **VS Code** conectado a um kernel do **Google Colab** para processamento em GPU de alto desempenho, utilizando armazenamento persistente via Google Drive.

### Stack Tecnológica
* **Linguagem Principal:** Python 3
* **Deep Learning:** TensorFlow / Keras (ou PyTorch)
* **Manipulação de Dados:** NumPy e Pandas
* **Processamento de Imagem:** PIL (Pillow) e OpenCV
* **Visualização:** Matplotlib & Seaborn

### Estrutura de Pastas Esperada
```text
├── dataset/
│   ├── train/
│   │   ├── image/   # 1.087 imagens médicas (.jpg)
│   │   └── mask/    # 1.087 máscaras correspondentes (.png)
│   └── test/        # Imagens para validação final
├── notebooks/
│   └── pipeline_treinamento.ipynb  # Fluxo end-to-end desenvolvido
├── README.md
└── requirements.txt
```

# ⚙️ Funcionalidades do Pipeline em PythonAutomação de Download: 

1.Script integrado com o novo protocolo de autenticação do Kaggle via tokens temporários (access_token).

2.Pipelines de Dados Otimizados (tf.data): Carregamento preguiçoso (lazy loading) dos arrays de pixels direto do disco sob demanda, evitando estouro da memória RAM do ambiente.

3.Pré-processamento e Normalização: Conversão automática das imagens e máscaras para matrizes NumPy, com reescala de valores de intensidade de pixel para o intervalo $[0, 1]$.

4.Data Augmentation Sincronizado: Algoritmo customizado que garante que qualquer transformação geométrica aplicada na imagem médica (rotação, zoom, crop) seja replicada com exatidão matemática na sua máscara correspondente.
