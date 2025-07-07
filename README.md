# Malicious URL Detector with Explainable AI (XAI) Insights

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge&logo=matplotlib&logoColor=white)
![Cybersecurity](https://img.shields.io/badge/Cybersecurity-000000?style=for-the-badge&logo=security&logoColor=white)
![AI/ML](https://img.shields.io/badge/AI%2FML-FF6600?style=for-the-badge&logo=tensorflow&logoColor=white)
![XAI](https://img.shields.io/badge/AI-Explainable_AI-blue?style=for-the-badge)

## Project Overview

This project is a sophisticated web-based Python Flask application designed to detect malicious URLs (e.g., phishing, malware, spam) using Machine Learning. Its standout feature is the integration of **Explainable AI (XAI)**, which visually highlights *why* the ML model made a particular classification, providing valuable insights for security analysts. This project demonstrates the practical application of AI in cybersecurity for proactive threat detection and interpretability.

## Key Features

* **AI-Powered URL Classification:**
    * Utilizes a trained **Logistic Regression model** to classify URLs as "Benign" or "Potentially Malicious".
    * Extracts a rich set of features from URLs, including length metrics, domain characteristics (dots, hyphens, digits, IP presence), protocol (HTTPS), URL shortening, subdomains, and **Shannon Entropy** (to detect randomness).
    * Leverages **TF-IDF Vectorization** for textual features within the URL path and query.
* **Explainable AI (XAI) Insights:**
    * For each analysis, it identifies and **visually represents the top contributing features** that influenced the model's prediction.
    * Generates an interactive **bar chart (using Chart.js)** showing feature coefficients, clearly indicating which URL characteristics pushed the prediction towards "malicious" or "benign". This provides crucial interpretability.
* **Interactive Web Interface:**
    * Clean, modern, and dark-themed UI for a professional look.
    * Input field for users to submit URLs for analysis.
    * Prominently displays the prediction label (Benign/Potentially Malicious) and confidence score with an animating progress bar.
    * Lists all extracted URL features.
    * Provides clear loading, error, and info messages with visual cues.
* **Secure Access:** Implements a simple master password login system to protect access to the analysis tool.
* **Responsive Design:** The entire web interface adapts seamlessly to various screen sizes.
* **Full-Stack AI Application:** Combines a Python Flask backend (for ML model hosting, feature extraction, and prediction) with a dynamic HTML/CSS/JavaScript frontend.

## Technologies Used

* **Python 3.x:** Core language for the backend logic and ML operations.
* **Flask:** A lightweight Python web framework for serving the application and exposing the analysis API endpoint.
* **`pandas`:** For efficient data loading, manipulation, and feature engineering from URL data.
* **`scikit-learn`:** Provides key machine learning tools:
    * `TfidfVectorizer` for text feature extraction.
    * `StandardScaler` for numerical feature scaling.
    * `LogisticRegression` for the classification model.
* **`matplotlib`:** (Used implicitly by Chart.js for concepts, but also for backend plotting if needed for debugging).
* **`joblib`:** For saving and loading the trained ML model, vectorizer, and scaler.
* **HTML5, CSS3, JavaScript:** Used for developing the interactive and responsive frontend user interface.
* **Chart.js:** For creating the interactive feature importance bar chart on the frontend.
* **`fetch` API:** For asynchronous communication between the frontend and the Flask backend API.
* **Font Awesome:** Integrated for scalable vector icons.

## How to Download and Run the Project

### 1. Prerequisites

* **Python 3.x:** Ensure Python 3.x is installed on your system. Download from [python.org](https://www.python.org/downloads/).
* **`pip`:** Python's package installer.
* **Git:** Ensure Git is installed on your system. Download from [git-scm.com](https://git-scm.com/downloads/).
* **VS Code (Recommended):** For a smooth development experience.

### 2. Download the Project

1.  **Open your terminal or Git Bash.**
2.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)AtharvaMeherkar/URL-Threat-Analyzer-XAI.git
    ```
3.  **Navigate into the project directory:**
    ```bash
    cd URL-Threat-Analyzer-XAI
    ```

### 3. Setup and Installation

1.  **Open the project in VS Code:**
    ```bash
    code .
    ```
2.  **Open the Integrated Terminal in VS Code** (`Ctrl + ~`).
3.  **Create and activate a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
    You should see `(venv)` at the beginning of your terminal prompt.
4.  **Install the required Python packages:**
    ```bash
    pip install Flask pandas scikit-learn matplotlib
    ```
5.  **Dataset:** The `urls_dataset.csv` file, containing sample URLs for training, is included in the repository.

### 4. Execution

1.  **Ensure your virtual environment is active** in the VS Code terminal.
2.  **Set the Flask application environment variable:**
    ```bash
    # On Windows:
    $env:FLASK_APP = "app.py"
    # On macOS/Linux:
    export FLASK_APP=app.py
    ```
3.  **Run the Flask development server:**
    ```bash
    python -m flask run
    ```
    *(The first time you run this, the ML model will be trained and saved (`url_classifier_model.pkl`, `url_vectorizer.pkl`, `url_scaler.pkl`, `model_feature_names.pkl`). This process is relatively quick for the provided dataset. You'll see "Training ML model..." messages in your terminal.)*
    * **Default Master Password:** On the very first run, if no master password is set, the backend will automatically set it to: `analyze123`. You'll see a warning in your terminal.
4.  **Open your web browser** and go to `http://127.0.0.1:5000/login` (or `http://localhost:5000/login`).
5.  **Log in:** Enter the master password (`analyze123` by default) to access the analyzer interface.
6.  **Interact with the Analyzer:**
    * Enter URLs into the input field and click "Analyze".
    * **Test Benign URLs:** `https://www.google.com`, `https://docs.python.org/3/`, `https://secure.paypal.com/myaccount/summary`
    * **Test Malicious URLs (from `urls_dataset.csv` or similar patterns):** `http://phishing.bankofamerica.com.securelogin.xyz/login.php`, `https://free-netflix-account.xyz/verify.html`, `http://bit.ly/virusdownload`
    * Observe the prediction, confidence, extracted features, and most importantly, the **Feature Importance chart** explaining the model's decision.


## What I Learned / Challenges Faced

* **Machine Learning for Cybersecurity:** Gained deep practical experience in applying supervised learning (Logistic Regression) to a critical cybersecurity problem: malicious URL detection.
* **Explainable AI (XAI):** Implemented a core XAI concept by visually representing feature importance, allowing for interpretation of model predictions—a crucial skill for building trust in AI security systems.
* **Feature Engineering for URLs:** Developed robust methods to extract diverse and meaningful features from URLs (e.g., length metrics, domain properties, TLDs, entropy, text features) for ML model input.
* **Full-Stack AI Application:** Successfully built a complete web application, integrating a Python Flask backend (for ML model hosting and analysis) with a dynamic HTML/CSS/JavaScript frontend (for user interaction and result visualization).
* **Model Deployment & Consistency:** Managed the serialization and deserialization of ML models, vectorizers, and scalers (`joblib`) and ensured strict consistency in feature order and number between training and prediction phases—a common challenge in ML deployment.
* **Data Preprocessing Pipeline:** Gained experience in building a comprehensive data preprocessing pipeline for unstructured textual data (URLs).
* **User Interface for Security Tools:** Focused on creating an intuitive and informative interface for security analysis, providing clear results and actionable insights.
