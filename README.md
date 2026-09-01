# Generative AI Educational Chatbot 

An interactive, AI-powered educational chatbot designed to assist users with domain-specific learning. Developed as part of a Project-Based Learning (PBL) curriculum, this project features custom dataset integration, model performance comparison, and response similarity analysis.

## 📁 Repository Structure

*   **`app.py`**: The main application entry point (e.g., Streamlit, Flask, or FastAPI) that serves the chatbot interface.
*   **`chatbot.py`**: Contains the core generative AI logic, prompt engineering, and model invocation functions.
*   **`DataSet.json`**: The custom knowledge base, QA pairs, and contextual data used to ground the chatbot's educational responses.
*   **`graph.py`**: Scripts used to calculate metrics and generate data visualizations for model evaluation.
*   **`model_comparison.png`**: Visual documentation comparing the performance, latency, or accuracy of different LLMs tested during development.
*   **`similarity_distribution.png`**: A graph illustrating the distribution of semantic similarity scores between the chatbot's generated responses and the expected ground truth.
*   **`PBL Review-2 Documetation education exp...`**: Official project documentation, methodology, and presentation materials for the academic review.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed on your system.

### Installation

1. **Activate the virtual environment:**
   * **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   * **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Usage:**
* If using standard Python/Flask/FastAPI:
    ```bash
    python app.py
    ```
* If you want to test the UI:
    ```bash
    streamlit run app.py
    ```
4. **Evaluation Metrics:**
* Model Comparison: Refer to model_comparison.png to see how the selected GenAI model stacks up against alternative architectures
* Similarity Tracking: The similarity_distribution.png chart (generated via graph.py) visualizes how closely the chatbot's answers align with the verified educational material in DataSet.json.

5. **Documentation:**
* For a deep dive into the experimental setup, educational use cases, and system architecture, please refer to the included PBL Review-2 Documentation file.
