<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.5em"> Vision-Enhanced Message Passing: Cross-Attention Fusion in Multimodal Knowledge Graphs</h1>

<p align='center' style="text-align:center;font-size:1em;">
    Dvir Sela
    Peleg Michael
    Diana Moadi
    Shukri Zmero
    <br/> 
    Technion - Israel Institute of Technology
</p>


# Contents
- [Overview](#Overview)
- [Abstract](#Abstract)
- [Environment Setup](#Environment-Setup)
- [Running the code](#Running-the-code)
- [Data Preparation](#Data-Preparation) 🧪
- [Pretraining Multimodal Encoder](#Pretraining-Multimodal-Encoder) 🧠
- [Relational Link Prediction Training](#Relational-Link-Prediction-Training) 🔗
- [Qualitative & Diagnostic Analysis](#Qualitative--Diagnostic-Analysis) 🔍
- [Configuration](#Configuration) ⚙️

# Overview

This project builds a multimodal (image + text) representation of knowledge graph entities and performs relational link prediction over a the graph. A multimodal encoder is first pretrained, node features are fused and cached, and a downstream relational model predicts the edges.

# Abstract

Knowledge graphs (KGs) are powerful tools for representing structured relationships between real-world entities. However, traditional KGs typically rely solely on symbolic or textual data, neglecting the visual information that can enrich entity understanding. In this project, we propose a multimodal knowledge graph framework that integrates textual, visual, and relational signals to learn unified entity representations. Our approach combines the pretrained CLIP model for text–image encoding with a trainable cross-attention fusion module, followed by a graph-based relational reasoning stage using a Relational Graph Convolutional Network (R-GCN). The training process consists of two complementary stages: (1) multimodal pretraining using contrastive losses to align text and image embeddings, and (2) relational fine-tuning through link prediction to incorporate graph structure. We evaluate our model on a subset of the DBpedia dataset, comparing four feature configurations: text-only, image-only, concatenation, and fused multimodal representations. Experimental results demonstrate that our fused model achieves superior performance in link prediction tasks, outperforming all unimodal baselines in terms of Mean Reciprocal Rank (MRR) and Hits@10.

# Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate            # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch_scatter torch_sparse torch_geometric -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

# Data Preparation 🧪
1. Place raw metadata / graph edge lists into [data/raw](data/raw).
2.  Run the preprocessing notebook [pre_processing.ipynb](pre_processing.ipynb) to:
   - Download all the images
   - Process the graph data
3. Output artifacts saved to [data/processed](data/processed).

# Pretraining Multimodal Encoder 🧠
Run:
```bash
python pretrain_end_to_end.py
```
Resulting checkpoint saved in ./checkpoints/multimodal_encoder_final.pt.

# Relational Link Prediction Training 🔗
Run the following to start the training:
```bash
python train_link_prediction_relational.py
```
Models wil be saved to ./saved_models

# Qualitative & Diagnostic Analysis 🔍
To run the qualitative analysis:
```bash
python analyze_qualitative.py
```
# Configuration ⚙️
Adjust hyperparameters in [config.py](config.py), such as:
```python

# Model / feature dims
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FUSED_DIM = 512
PROJ_DIM = 256

# Training hyperparams
BATCH_SIZE = 64
EPOCHS = 20
LR = 5e-5
```
etc.
