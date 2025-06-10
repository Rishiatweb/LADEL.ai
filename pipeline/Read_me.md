# LADEL: Log Anomaly Detection and Explanation Layer (Google Colab Version)

[![Python Version](https://img.shields.io/badge/Colab%20Python-3.x-blue.svg)](https://research.google.com/colaboratory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/models)

This repository demonstrates the LADEL (Log Anomaly Detection and Explanation Layer) pipeline, adapted for easy execution and experimentation within a Google Colaboratory notebook. It allows users to automatically detect anomalies in log data and understand them through AI-generated explanations.

## Table of Contents

- [Overview](#overview)
- [How It Works (In Colab)](#how-it-works-in-colab)
- [Colab Notebook Structure](#colab-notebook-structure)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [Experimentation and Configuration](#experimentation-and-configuration)
- [Technology Stack](#technology-stack)
- [Future Steps](#future-steps)

## Overview

LADEL leverages semantic understanding of log messages to identify unusual patterns and provides explanations using large language models. This Colab version sets up a modular project structure directly within the notebook environment, allowing for a professional development experience without local setup hassles.

**Key Features (Colab Adapted):**

-   **Semantic Analysis**: Converts log messages into numerical vectors to capture their meaning.
-   **Unsupervised Detection**: Learns from unlabeled log data to identify anomalies.
-   **Pluggable Detection Models**: Easily switch between `IsolationForest`, `Local Outlier Factor (LOF)`, and `One-Class SVM` via a configuration file.
-   **LLM-Powered Explanations**: Uses `google/flan-t5-base` for generating concise anomaly explanations.
-   **Dynamic Configuration**: Control models and parameters by editing the `config.yaml` file generated within the Colab environment.
-   **Modular Code in Colab**: Python code is organized into logical modules (`.py` files) created programmatically using `%%writefile`.

## How It Works (In Colab)

The pipeline within the Colab notebook executes the following steps:

1.  **Environment Setup**:
    *   Installs necessary Python libraries (`sentence-transformers`, `scikit-learn`, `transformers`, etc.).
    *   Programmatically creates the project directory structure (`ladel/`) and all Python modules (`data_loader.py`, `main.py`, etc.) directly in the Colab file system.
    *   Generates a `config.yaml` file and a sample `app.log` file.
2.  **Log Ingestion & Preprocessing**: Reads and cleans lines from the `app.log` file (or a user-uploaded file).
3.  **Semantic Embedding**: Transforms log lines into numerical vectors using a pre-trained `Sentence-Transformer` model.
4.  **Anomaly Detection**: Feeds these vectors into the selected scikit-learn anomaly detection model (controlled by `config.yaml`).
5.  **Explanation Generation**: For each detected anomaly, a T5 language model generates an explanation.
6.  **Results Display**: Outputs are printed directly in the Colab cell.

## Colab Notebook Structure

The Colab notebook is organized into distinct cells:

1.  **Cell 1: Install Dependencies**: Installs all required Python packages.
2.  **Cell 2: Create Project Directories**: Creates the `ladel/` directory.
3.  **Cell 3: Create `config.yaml`**: Generates the main configuration file. **This is the cell you edit to experiment.**
4.  **Cells 4-8: Create Python Modules**: Use `%%writefile` to create `data_loader.py`, `embedding.py`, `detection.py`, `explanation.py`, and `main.py` inside the `ladel/` directory.
5.  **Cell 9: Create Sample Log File**: Generates `app.log` for quick testing.
6.  **Cell 10: Run the Analysis Pipeline**: Executes `!python ladel/main.py` to run the entire process.

The resulting file structure in your Colab environment's `/content/` directory will be:
/content/
├── app.log
├── config.yaml
└── ladel/
├── data_loader.py
├── detection.py
├── embedding.py
├── explanation.py
└── main.py

## Getting Started

1.  Open the LADEL Google Colab notebook.
2.  **Run cells 1 through 9 sequentially**. This will:
    *   Install all necessary libraries.
    *   Create all the project files and directories within your Colab session.
    *   Generate a sample `config.yaml` and `app.log`.

## Running the Pipeline

1.  After running cells 1-9, **run Cell 10 ("Run the Analysis Pipeline")**.
2.  The script will execute, processing `app.log` using the settings from `config.yaml`.
3.  Results, including detected anomalies and their explanations, will be printed in the output of Cell 10.

## Experimentation and Configuration

The core of experimentation lies in modifying the `config.yaml` file.

1.  **Navigate to Cell 3 ("Create `config.yaml`")**.
2.  **Edit the `config.yaml` content directly within that cell**. For example, to change the anomaly detection model:
    *   Modify the line `detection_model: 'isolation_forest'` to `detection_model: 'lof'` or `detection_model: 'one_class_svm'`.
    *   You can also adjust hyperparameters for each model in their respective sections.
3.  **Re-run Cell 3** to save your changes to the `config.yaml` file in the Colab file system.
4.  **Re-run Cell 10 ("Run the Analysis Pipeline")** to see the effects of your configuration changes.

**To use your own log file:**

1.  Upload your log file to the `/content/` directory in Colab (using the "Files" sidebar on the left).
2.  In Cell 3 (`config.yaml`), if you want `main.py` to pick it up automatically without code change, ensure your uploaded file is named `app.log` or modify the `main.py` execution line in Cell 10 to specify your log file: `!python ladel/main.py --log_file your_log_file_name.log`
3.  Alternatively, edit the `main.py` script (in Cell 8) to change the default log file path. Re-run Cell 8 and then Cell 10.

## Technology Stack

-   **Google Colaboratory**: For the interactive notebook environment.
-   **Python 3.x**
-   **PyTorch**: Core deep learning framework.
-   **Hugging Face `transformers` & `sentence-transformers`**: For LLMs and embeddings.
-   **Scikit-learn**: For anomaly detection models.
-   **NumPy**, **Pandas**, **PyYAML**: For data handling and configuration.

## Future Steps

While this Colab setup is excellent for learning and experimentation:

-   Consider implementing evaluation metrics if you have labeled data.
-   For persistent storage or larger datasets, integrate with Google Drive.
-   Explore more advanced LLMs or fine-tuning for domain-specific explanations.
