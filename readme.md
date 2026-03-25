# TAIIE: Transformational AI Innovation Engine 🚀🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange)](https://pytorch.org/)

[cite_start]**TAIIE** (Transformational AI Innovation Engine) is a novel architecture operationalizing computational creativity theories[cite: 5]. Moving beyond standard generative AI, TAIIE systematically navigates high-dimensional concept spaces to discover novel-yet-valid scientific branches.

[cite_start]It integrates Bayesian information foraging, utility-guided reinforcement learning, and automated paradigm shifts to generate novel-yet-valid conceptual artifacts[cite: 6]. 

## 🔬 Core Philosophy

[cite_start]Current AI systems exhibit limitations in constrained creative generation[cite: 10]. [cite_start]TAIIE addresses these by optimizing the following objective function[cite: 11]:

$$topic^{*} = argmax_{topic}[Validity(t) + \beta Novelty(t)]$$

[cite_start]When the Kullback-Leibler divergence of the novelty distribution exceeds a threshold ($D_{KL} > \tau$), the system triggers a Kuhnian paradigm shift[cite: 22, 24]. [cite_start]This effectively resets the constraints, shifting the model from "Normal Science" to "Extraordinary Science" to explore entirely new semantic manifolds[cite: 21, 23, 24].

## 🏗️ Architecture & Features

This repository contains the full implementation of the TAIIE framework, including:

* **Custom GPT-2 Style Transformer (`ResearchTopicGPT`):** An 8-layer, 8-head self-attention architecture built from scratch in PyTorch to model concept embeddings.
* [cite_start]**The Creativity Engine (`CreativityEngine`):** Operates on three levels[cite: 14]:
    * [cite_start]*Combinatorial:* Token recombination[cite: 15].
    * [cite_start]*Exploratory:* Bayesian sampling within constraints[cite: 16].
    * [cite_start]*Transformational:* Constraint modification[cite: 17].
* **Human-in-the-Loop (HITL) RL Pipeline:** A dynamic feedback loop that evaluates generated topics for validity and scales a novelty threshold ($\tau$), penalizing semantic voids and rewarding genuine innovation.

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/TAIIE-AI-Innovation-Engine.git](https://github.com/YOUR_USERNAME/TAIIE-AI-Innovation-Engine.git)
cd TAIIE-AI-Innovation-Engine
pip install -r requirements.txt
