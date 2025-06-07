# LADEL.ai
# Log Anomaly Detection and Explanation Layer
This repository contains a Python notebook for an end-to-end log anomaly detection pipeline. The system processes raw log files, identifies unusual or anomalous log entries using machine learning, and leverages a Large Language Model (LLM) to provide human-readable explanations for why each anomaly is considered unusual.
This project is designed to run in a Google Colab environment, making it easy to get started with no local setup required.
🚀 Key Features
File Upload Interface: Easily upload any text-based log file (.log, .txt, etc.).
Semantic Embeddings: Uses Sentence-Transformers to convert log lines into meaningful numerical vectors, capturing the semantic content of each message.
Unsupervised Anomaly Detection: Employs the IsolationForest algorithm to identify outliers in the log data without needing pre-labeled examples.
AI-Powered Explanations: Leverages a local Hugging Face model (google/flan-t5-base) to analyze each detected anomaly and explain its significance in plain English.
Self-Contained & Environment-Friendly: Runs entirely within a free Google Colab notebook, using a local LLM to avoid API key management and external dependencies.
Clear & Organized Output: Presents a final summary of normal vs. anomalous logs, followed by a detailed, structured analysis for each anomaly.
<!-- It's highly recommended to replace this with a real screenshot of your notebook's output! -->
⚙️ How It Works
The pipeline follows these steps:
Log Ingestion: The user uploads a log file through the Colab interface. The file is read and parsed into individual log lines.
Semantic Vectorization: Each log line is passed through the all-MiniLM-L6-v2 Sentence Transformer model. This model converts the text into a 384-dimensional vector that represents its meaning. Similar logs will have vectors that are close to each other in this high-dimensional space.
Anomaly Scoring: The collection of log vectors is fed into an IsolationForest model. This unsupervised learning algorithm is highly effective at identifying outliers by "isolating" data points that are different from the dense majority. It assigns a score to each log, flagging the most isolated ones as anomalies.
AI Explanation Generation: For each log flagged as an anomaly, a carefully crafted prompt is sent to a locally-run google/flan-t5-base model. The prompt provides the LLM with:
A high-quality example of a good analysis (few-shot learning).
A sample of "normal" logs from the file for context.
The specific anomalous log to analyze.
Reporting: The model's generated explanation—detailing why the log is unusual compared to the norm—is printed directly below the anomaly for immediate review.
🛠️ Technologies Used
Python 3: The core programming language.
Pandas & NumPy: For data manipulation and numerical operations.
Scikit-learn: For the IsolationForest machine learning model.
Sentence-Transformers: For generating high-quality semantic embeddings of log text.
Hugging Face transformers: To download and run the google/flan-t5-base model for natural language explanations.
PyTorch: As the backend for the transformers library.
Google Colab: For the cloud-based notebook environment with free GPU access.
🏃‍♀️ How to Run
You can run this project directly in your browser with just a few clicks.
Open in Colab: Click the "Open in Colab" badge at the top of this README.
Set Runtime to GPU: In the Colab notebook, navigate to Runtime -> Change runtime type and select T4 GPU from the "Hardware accelerator" dropdown. This is crucial for running the language model efficiently.
Run All Cells: Go to Runtime -> Run all.
Upload Your Log File: When prompted by the first code cell, click the "Choose Files" button and select a log file from your local machine.
Review the Analysis: The notebook will execute all steps, from embedding to anomaly detection to explanation. Scroll to the bottom to see the final analysis report. The first run will take a few minutes to download the models; subsequent runs will be faster.
📜 Example Log File
A sample log file (sample.log) is included in this repository to demonstrate the pipeline's functionality. It contains a mix of routine log entries and a few engineered anomalies (errors, critical failures, unusual info messages) that the system is designed to catch.
🤝 Contributing
Contributions are welcome! If you have ideas for improvements, please open an issue to discuss what you would like to change or submit a pull request.
