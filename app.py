# app.py
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import joblib # For saving/loading the model
from urllib.parse import urlparse # For URL parsing

# Scikit-learn for ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # Good for interpretability
from sklearn.pipeline import Pipeline # Not directly used in this version, but good to have if refactoring
from sklearn.preprocessing import StandardScaler # For scaling numerical features

# Flask
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import hashlib # For master password hashing

# --- Configuration ---
MODEL_FILE = 'url_classifier_model.pkl'
VECTORIZER_FILE = 'url_vectorizer.pkl'
SCALER_FILE = 'url_scaler.pkl'
FEATURE_NAMES_FILE = 'model_feature_names.pkl' # New file to store feature names
REPORT_DIR = 'url_analysis_reports'

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)
if not os.path.exists(os.path.join(REPORT_DIR, 'graphs')):
    os.makedirs(os.path.join(REPORT_DIR, 'graphs'))

# --- Logging ---
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_malware_analyzer_secret_key' # FIXED KEY FOR DEMO

# --- Master Password Hashing (for tool authentication) ---
MASTER_HASH_FILE = 'analyzer_master_hash.txt'
ANALYZER_SALT = b'malware_analyzer_salt_789'

def hash_password(password: str, salt: bytes) -> str:
    return hashlib.sha256(salt + password.encode()).hexdigest()

def set_analyzer_master_password(password: str):
    hashed_pass = hash_password(password, ANALYZER_SALT)
    with open(MASTER_HASH_FILE, 'w') as f:
        f.write(hashed_pass)
    logger.info("Analyzer master password hash set/updated.")

def verify_analyzer_master_password(password: str) -> bool:
    if not os.path.exists(MASTER_HASH_FILE):
        return False
    with open(MASTER_HASH_FILE, 'r') as f:
        stored_hash = f.read().strip()
    return hash_password(password, ANALYZER_SALT) == stored_hash

if not os.path.exists(MASTER_HASH_FILE):
    logger.warning("No analyzer master password found. Setting default: 'analyze123'")
    set_analyzer_master_password('analyze123') # Default password for first run

# --- Simulated Malware Features and ML Model ---
# Suspicious keywords/API calls often found in malicious code (simplified)
SUSPICIOUS_KEYWORDS = [
    'CreateRemoteThread', 'WriteProcessMemory', 'VirtualAllocEx', 'LoadLibraryA', 'GetProcAddress',
    'ShellExecute', 'WinExec', 'system(', 'subprocess.call', 'eval(', 'exec(', 'base64.b64decode',
    'socket.connect', 'urllib.request.urlopen', 'requests.get', 'sendall', 'recv',
    'IsDebuggerPresent', 'anti-debug', 'obfuscate', 'encrypt', 'decrypt', 'xor',
    '.dll', '.exe', '.vbs', '.ps1', '.bat', 'cmd.exe', 'powershell.exe'
]

def get_url_features(url):
    """Extracts various features from a URL."""
    parsed_url = urlparse(url)
    
    features = {
        'url_length': len(url),
        'domain_length': len(parsed_url.netloc),
        'path_length': len(parsed_url.path),
        'query_length': len(parsed_url.query),
        'fragment_length': len(parsed_url.fragment),
        'num_dots_in_domain': parsed_url.netloc.count('.'),
        'num_hyphens_in_domain': parsed_url.netloc.count('-'),
        'num_digits_in_domain': sum(c.isdigit() for c in parsed_url.netloc),
        'has_ip_in_domain': bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed_url.netloc)),
        'is_https': parsed_url.scheme == 'https',
        'is_shortened': any(s in url for s in ['bit.ly', 'tinyurl.com', 'goo.gl']), # Simple check
        'num_subdomains': parsed_url.netloc.count('.') - 1 if parsed_url.netloc.count('.') > 0 else 0,
    }

    # Extract TLD (Top-Level Domain)
    tld_match = re.search(r'\.([a-zA-Z]{2,})$', parsed_url.netloc)
    features['tld'] = tld_match.group(1) if tld_match else 'unknown'

    return features

def calculate_entropy(text):
    """Calculates Shannon entropy of a string."""
    if not text:
        return 0.0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy

def train_and_save_model(df_train, numerical_features_list, text_feature_column_name, all_tld_columns_from_training):
    """Trains the ML model and saves it along with vectorizer/scaler."""
    logger.info("Training ML model...")

    # Numerical features
    X_num = df_train[numerical_features_list]

    # Text feature for TF-IDF
    text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english') # Limit features for simplicity
    X_text = text_vectorizer.fit_transform(df_train[text_feature_column_name]) # Use the correct column name

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled_num = scaler.fit_transform(X_num)

    # Prepare TLD dummy features for training data (ensure all_tld_columns_from_training are present)
    # This step ensures that the TLD columns match exactly what was observed in the full training set
    X_tld_dummies = df_train[all_tld_columns_from_training]

    # Combine all features for training
    from scipy.sparse import hstack
    X_combined = hstack([X_scaled_num, X_tld_dummies, X_text]) # Order: Numerical, TLDs, Text

    y = df_train['label'].apply(lambda x: 1 if x == 'malicious' else 0) # Convert labels to 0/1

    # Train Logistic Regression model
    model = LogisticRegression(random_state=42, solver='liblinear', C=0.5) # C is regularization strength
    model.fit(X_combined, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(text_vectorizer, VECTORIZER_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    # --- CRITICAL FIX: Store exact feature names in the correct order ---
    # 1. Numerical features (scaled)
    all_feature_names = numerical_features_list.copy()
    
    # 2. TLD dummy features (from the training data's actual columns)
    all_feature_names.extend(all_tld_columns_from_training)

    # 3. Text (TF-IDF) features
    all_feature_names.extend(text_vectorizer.get_feature_names_out().tolist())
    
    joblib.dump(all_feature_names, FEATURE_NAMES_FILE) # Save this list
    
    logger.info(f"ML model, vectorizer, scaler, and feature names saved.")
    return model, text_vectorizer, scaler, all_feature_names

def load_or_train_model(dataset_path='urls_dataset.csv'):
    """Loads model/vectorizer/scaler or trains them if not found."""
    global classifier_model, text_vectorizer_model, scaler_model, numerical_features_list, text_feature_names, all_model_feature_names, all_tld_columns_from_training_set

    # Define all possible numerical features that `get_url_features` can extract
    numerical_features_base = [
        'url_length', 'domain_length', 'path_length', 'query_length', 'fragment_length',
        'num_dots_in_domain', 'num_hyphens_in_domain', 'num_digits_in_domain',
        'has_ip_in_domain', 'is_https', 'is_shortened', 'num_subdomains', 'entropy'
    ]

    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE) and \
       os.path.exists(SCALER_FILE) and os.path.exists(FEATURE_NAMES_FILE): # Check for new feature names file
        try:
            classifier_model = joblib.load(MODEL_FILE)
            text_vectorizer_model = joblib.load(VECTORIZER_FILE)
            scaler_model = joblib.load(SCALER_FILE)
            all_model_feature_names = joblib.load(FEATURE_NAMES_FILE) # Load all feature names
            
            # Reconstruct numerical_features_list and text_feature_names from all_model_feature_names
            numerical_features_list = [f for f in all_model_feature_names if f in numerical_features_base]
            # text_feature_names are the words from TF-IDF, which are not in numerical_features_base or start with tld_
            text_feature_names = [f for f in all_model_feature_names if f not in numerical_features_list and not f.startswith('tld_')]
            # Capture TLD columns from the loaded feature names for prediction consistency
            all_tld_columns_from_training_set = [f for f in all_model_feature_names if f.startswith('tld_')]

            logger.info("ML model, vectorizer, scaler, and feature names loaded from file.")
            return True
        except Exception as e:
            logger.error(f"Error loading ML components: {e}. Retraining model.")
    
    # If loading fails or files don't exist, train new model
    try:
        df = pd.read_csv(dataset_path)
        
        # Extract features for the entire dataset
        features_list_of_dicts = [get_url_features(url) for url in df['url']]
        features_df = pd.DataFrame(features_list_of_dicts)
        
        # Add entropy
        features_df['entropy'] = df['url'].apply(calculate_entropy)

        # Identify numerical features (from base list)
        numerical_features_list = [col for col in numerical_features_base if col in features_df.columns]
        
        # Ensure all numerical features are numeric and fill NaNs
        for col in numerical_features_list:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        
        # Handle TLD one-hot encoding for training data
        tld_dummies = pd.get_dummies(features_df['tld'], prefix='tld', dummy_na=False)
        features_df = pd.concat([features_df.drop(columns=['tld']), tld_dummies], axis=1) # Drop original TLD column
        
        # Capture all TLD columns generated during training for later use in prediction
        all_tld_columns_from_training_set = [col for col in features_df.columns if col.startswith('tld_')]

        # Prepare text feature column for TF-IDF
        features_df['combined_text_for_tfidf'] = df['url'].apply(lambda u: urlparse(u).path + ' ' + urlparse(u).query + ' ' + urlparse(u).fragment)
        
        # Combine features_df with labels for training
        df_for_training = pd.concat([features_df, df[['label']]], axis=1)
        
        classifier_model, text_vectorizer_model, scaler_model, all_model_feature_names = \
            train_and_save_model(df_for_training, numerical_features_list, 'combined_text_for_tfidf', all_tld_columns_from_training_set)
        
        text_feature_names = [f for f in all_model_feature_names if f not in numerical_features_list and not f.startswith('tld_')]

        return True
    except FileNotFoundError:
        logger.error(f"Dataset not found at {dataset_path}. Cannot train model.")
        return False
    except Exception as e:
        logger.error(f"Failed to train model: {e}", exc_info=True)
        return False

# Initialize global model variables
classifier_model = None
text_vectorizer_model = None
scaler_model = None
numerical_features_list = []
text_feature_names = []
all_model_feature_names = [] # Stores the exact order of all features the model was trained on
all_tld_columns_from_training_set = [] # Stores the exact TLD columns from training

load_status = load_or_train_model()
if not load_status:
    logger.critical("Application cannot start without a trained model. Please check dataset and errors above.")

# --- Analysis Function ---
def analyze_url(url_to_analyze):
    """Performs feature extraction and prediction for a single URL."""
    if classifier_model is None or text_vectorizer_model is None or scaler_model is None:
        return {"error": "ML model not loaded. Cannot analyze."}

    # Extract features for the new URL
    new_url_features_dict = get_url_features(url_to_analyze)
    new_url_features_dict['entropy'] = calculate_entropy(url_to_analyze) # Calculate entropy for the whole URL

    # Prepare numerical features
    # Create a DataFrame row with features for the new URL, ensuring all numerical_features_list columns are present
    numerical_data_raw = pd.DataFrame([new_url_features_dict]).reindex(columns=numerical_features_list, fill_value=0)
    numerical_data_scaled = scaler_model.transform(numerical_data_raw)
    
    # Prepare text features
    parsed_url = urlparse(url_to_analyze)
    text_content = parsed_url.path + ' ' + parsed_url.query + ' ' + parsed_url.fragment
    text_vector = text_vectorizer_model.transform([text_content])

    # Handle TLD (Top-Level Domain) for prediction
    tld_of_new_url = new_url_features_dict.get('tld', 'unknown')
    
    # Create a DataFrame for the new URL's TLD, with columns matching all TLD dummies seen during training
    # Use the globally stored list of TLD columns from training
    tld_features_for_prediction = pd.DataFrame(0, index=[0], columns=all_tld_columns_from_training_set)
    
    specific_tld_col = f'tld_{tld_of_new_url}'
    if specific_tld_col in tld_features_for_prediction.columns:
        tld_features_for_prediction[specific_tld_col] = 1

    # Combine all features into a single array for prediction, maintaining exact order
    input_features_dict = {name: 0 for name in all_model_feature_names}

    # Fill numerical features
    for i, feature_name in enumerate(numerical_features_list):
        input_features_dict[feature_name] = numerical_data_scaled[0, i]

    # Fill TLD features
    for feature_name in all_tld_columns_from_training_set: # Iterate over the exact TLD columns from training
        input_features_dict[feature_name] = tld_features_for_prediction[feature_name].iloc[0]

    # Fill text (TF-IDF) features
    # Iterate over feature names from the text vectorizer and their values
    for col_idx, feature_name in enumerate(text_vectorizer_model.get_feature_names_out()):
        if feature_name in input_features_dict: # Ensure it's a feature the model expects
            input_features_dict[feature_name] = text_vector[0, col_idx]
    
    # Convert the dictionary to a list in the exact order the model expects
    X_predict_ordered = np.array([input_features_dict[name] for name in all_model_feature_names]).reshape(1, -1)

    # Make prediction
    prediction_proba = classifier_model.predict_proba(X_predict_ordered)[0]
    prediction_class = classifier_model.predict(X_predict_ordered)[0]

    result_label = "Potentially Malicious" if prediction_class == 1 else "Benign"
    confidence = round(prediction_proba[prediction_class] * 100, 2)

    # --- Explainability (XAI) ---
    feature_importances = classifier_model.coef_[0] 

    contributing_features = {}
    for i, feature_name in enumerate(all_model_feature_names):
        coef = feature_importances[i]
        is_active = False

        if feature_name in numerical_features_list:
            # Check original numerical value (before scaling) to see if it's non-zero/active
            if new_url_features_dict.get(feature_name, 0) > 0 or \
               (feature_name == 'is_https' and new_url_features_dict.get('is_https', False)) or \
               (feature_name == 'has_ip_in_domain' and new_url_features_dict.get('has_ip_in_domain', False)) or \
               (feature_name == 'is_shortened' and new_url_features_dict.get('is_shortened', False)):
                is_active = True
        elif feature_name.startswith('tld_'):
            # Check if this specific TLD dummy is active for the current URL
            if feature_name == f'tld_{tld_of_new_url}':
                is_active = True
        elif feature_name in text_vectorizer_model.vocabulary_: # Check if the word is in the vectorizer's vocabulary
            # Check if the word is actually present in the current URL's text content
            # and if its TF-IDF value is non-zero for this URL
            word_idx = text_vectorizer_model.vocabulary_[feature_name]
            if text_vector[0, word_idx] > 0:
                is_active = True
        
        # If active and coefficient is significant (positive for malicious, negative for benign)
        if is_active and (
            (prediction_class == 1 and coef > 0) or # Malicious prediction, positive coef
            (prediction_class == 0 and coef < 0)    # Benign prediction, negative coef
        ):
            contributing_features[feature_name] = coef

    # Sort by absolute coefficient value for display
    sorted_contributing_features = sorted(contributing_features.items(), key=lambda item: abs(item[1]), reverse=True)
    
    # Prepare data for graph
    graph_data = []
    for feature, coef in sorted_contributing_features[:10]: # Top 10 contributing features
        graph_data.append({'feature': feature, 'importance': coef})

    return {
        'url': url_to_analyze,
        'prediction': result_label,
        'confidence': confidence,
        'feature_importance_graph_data': graph_data,
        'extracted_features': new_url_features_dict # For display
    }

# --- Routes ---
@app.route('/')
def root_redirect():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form['password']
        if verify_analyzer_master_password(password):
            session['logged_in'] = True
            logger.info("User logged in successfully.")
            return redirect(url_for('analyzer_tool'))
        else:
            logger.warning("Failed login attempt.")
            return render_template('login.html', error="Incorrect Password")
    
    if not os.path.exists(MASTER_HASH_FILE):
        return render_template('login.html', set_password_mode=True)
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    logger.info("User logged out.")
    return redirect(url_for('login'))

@app.route('/analyzer')
def analyzer_tool():
    if 'logged_in' not in session or not session['logged_in']:
        logger.warning("Attempted to access analyzer without login. Redirecting.")
        return redirect(url_for('login'))
    return render_template('index.html')

# API to analyze URL
@app.route('/api/analyze_url', methods=['POST'])
def api_analyze_url():
    if 'logged_in' not in session or not session['logged_in']:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    url = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'No URL provided for analysis.'}), 400
    
    # Pre-check model status
    if classifier_model is None or text_vectorizer_model is None or scaler_model is None:
        return jsonify({'error': 'ML model not loaded on server. Please check server logs.'}), 500

    try:
        analysis_result = analyze_url(url)
        logger.info(f"URL '{url}' analyzed. Result: {analysis_result['prediction']} ({analysis_result['confidence']}%)")
        return jsonify(analysis_result)
    except Exception as e:
        logger.error(f"Error during URL analysis: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500

# --- Run the Flask app ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)