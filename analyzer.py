# analyzer.py
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # For scaling numerical features

# --- Configuration ---
MODEL_FILE = 'url_classifier_model.pkl'
VECTORIZER_FILE = 'url_vectorizer.pkl'
SCALER_FILE = 'url_scaler.pkl'
REPORT_DIR = 'url_analysis_reports'

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)
if not os.path.exists(os.path.join(REPORT_DIR, 'graphs')):
    os.makedirs(os.path.join(REPORT_DIR, 'graphs'))

# --- Feature Engineering Functions ---
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
        'has_suspicious_keywords': 0, # Will be set later by TF-IDF
        'entropy': 0 # Will be calculated later
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

# --- Model Training and Loading ---
def train_and_save_model(df, numerical_features, text_features):
    """Trains the ML model and saves it along with vectorizer/scaler."""
    logger.info("Training ML model...")

    # Combine numerical features and TF-IDF features
    # Numerical features
    X_num = df[numerical_features]

    # Text feature (combined path, query, fragment)
    df['text_features'] = df['path'] + ' ' + df['query'] + ' ' + df['fragment']
    
    # TF-IDF for text features
    text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english') # Limit features for simplicity
    X_text = text_vectorizer.fit_transform(df['text_features'])

    # Combine all features
    # Convert X_num to sparse matrix to concatenate with X_text
    from scipy.sparse import hstack
    X = hstack([X_num, X_text])

    y = df['label'].apply(lambda x: 1 if x == 'malicious' else 0) # Convert labels to 0/1

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled_num = scaler.fit_transform(X_num)
    X = hstack([X_scaled_num, X_text]) # Re-combine scaled numerical with text

    # Train Logistic Regression model
    model = LogisticRegression(random_state=42, solver='liblinear', C=0.5) # C is regularization strength
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(text_vectorizer, VECTORIZER_FILE)
    joblib.dump(scaler, SCALER_FILE)
    logger.info(f"ML model, vectorizer, and scaler saved to {REPORT_DIR}")
    return model, text_vectorizer, scaler, numerical_features, text_vectorizer.get_feature_names_out()

def load_or_train_model(dataset_path='urls_dataset.csv'):
    """Loads model/vectorizer/scaler or trains them if not found."""
    global classifier_model, text_vectorizer_model, scaler_model, numerical_features_list, text_feature_names

    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE) and os.path.exists(SCALER_FILE):
        try:
            classifier_model = joblib.load(MODEL_FILE)
            text_vectorizer_model = joblib.load(VECTORIZER_FILE)
            scaler_model = joblib.load(SCALER_FILE)
            
            # Reconstruct feature names for XAI
            # Need to re-process a dummy dataframe to get the exact feature names used during training
            dummy_df = pd.read_csv(dataset_path)
            dummy_df['parsed'] = dummy_df['url'].apply(urlparse)
            dummy_df['url_features'] = dummy_df['url'].apply(get_url_features)
            dummy_features_df = pd.DataFrame(dummy_df['url_features'].tolist())
            numerical_features_list = [col for col in dummy_features_df.columns if dummy_features_df[col].dtype in ['int64', 'float64', 'bool']]
            
            text_feature_names = text_vectorizer_model.get_feature_names_out()
            logger.info("ML model, vectorizer, and scaler loaded from file.")
            return True
        except Exception as e:
            logger.error(f"Error loading ML components: {e}. Retraining model.")
    
    # If loading fails or files don't exist, train new model
    try:
        df = pd.read_csv(dataset_path)
        df['parsed'] = df['url'].apply(urlparse)
        df['url_features'] = df['url'].apply(get_url_features)
        
        # Expand url_features dictionary into separate columns
        features_df = pd.DataFrame(df['url_features'].tolist())
        
        # Add entropy
        features_df['entropy'] = df['url'].apply(calculate_entropy)

        # Identify numerical and text features for the model
        numerical_features_list = [
            'url_length', 'domain_length', 'path_length', 'query_length', 'fragment_length',
            'num_dots_in_domain', 'num_hyphens_in_domain', 'num_digits_in_domain',
            'has_ip_in_domain', 'is_https', 'is_shortened', 'num_subdomains', 'entropy'
        ]
        
        # Ensure all numerical features are numeric
        for col in numerical_features_list:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
            else:
                features_df[col] = 0 # Add missing numerical features with default 0

        # One-hot encode TLD (Top-Level Domain) as a categorical feature
        tld_dummies = pd.get_dummies(features_df['tld'], prefix='tld', dummy_na=False)
        features_df = pd.concat([features_df, tld_dummies], axis=1)
        
        # Add 'has_suspicious_keywords' from text vectorizer later
        
        # Prepare data for training
        df_for_training = pd.concat([features_df, df[['label', 'url']]], axis=1) # Combine features with labels and original URL
        
        classifier_model, text_vectorizer_model, scaler_model, numerical_features_list, text_feature_names = \
            train_and_save_model(df_for_training, numerical_features_list, ['path', 'query', 'fragment']) # Pass combined text
        
        return True
    except FileNotFoundError:
        logger.error(f"Dataset not found at {dataset_path}. Cannot train model.")
        return False
    except Exception as e:
        logger.error(f"Failed to train model: {e}", exc_info=True)
        return False

# Load or train model on app startup
classifier_model = None
text_vectorizer_model = None
scaler_model = None
numerical_features_list = []
text_feature_names = []

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
    numerical_data = pd.DataFrame([new_url_features_dict])[numerical_features_list]
    numerical_data_scaled = scaler_model.transform(numerical_data)

    # Prepare text features
    parsed_url = urlparse(url_to_analyze)
    text_content = parsed_url.path + ' ' + parsed_url.query + ' ' + parsed_url.fragment
    text_vector = text_vectorizer_model.transform([text_content])

    # Handle TLD (Top-Level Domain) for prediction
    # Need to create dummy variables for the TLD of the new URL based on the training data's TLDs
    tld_of_new_url = new_url_features_dict.get('tld', 'unknown')
    
    # Create a DataFrame for the new URL's TLD, with columns matching training data's TLD dummies
    # This is a simplified way; in production, you'd store the list of TLDs seen during training.
    # For demo, we'll just check if the tld_feature exists in the model's coefficients.
    
    # Identify all TLD features from the model's coefficients
    model_tld_features = [f for f in classifier_model.feature_names_in_ if f.startswith('tld_')]
    tld_features_for_prediction = pd.DataFrame(0, index=[0], columns=model_tld_features)
    
    specific_tld_col = f'tld_{tld_of_new_url}'
    if specific_tld_col in tld_features_for_prediction.columns:
        tld_features_for_prediction[specific_tld_col] = 1

    # Combine all features for prediction
    from scipy.sparse import hstack
    
    # Ensure numerical_data_scaled is a dense array for hstack if needed
    if not isinstance(numerical_data_scaled, np.ndarray):
        numerical_data_scaled = numerical_data_scaled.toarray() # Convert sparse to dense if needed

    # Combine numerical, TLD, and text features
    # Ensure feature order matches training. This is complex with dynamic TLDs.
    # For simplicity, let's rebuild the input vector carefully.
    
    # Reconstruct the feature vector in the exact order the model expects
    # This requires knowing the exact order of features used during training.
    # For LogisticRegression, feature_names_in_ gives this order.
    
    all_input_features_flat = []
    
    # Add numerical features (scaled)
    all_input_features_flat.extend(numerical_data_scaled[0].tolist())
    
    # Add TLD one-hot encoded features
    tld_one_hot = [0] * len(model_tld_features)
    if specific_tld_col in model_tld_features:
        tld_one_hot[model_tld_features.index(specific_tld_col)] = 1
    all_input_features_flat.extend(tld_one_hot)

    # Add TF-IDF text features
    all_input_features_flat.extend(text_vector.toarray()[0].tolist()) # Convert sparse to dense

    # Convert to numpy array and reshape for prediction
    X_predict = np.array(all_input_features_flat).reshape(1, -1)

    # Make prediction
    prediction_proba = classifier_model.predict_proba(X_predict)[0]
    prediction_class = classifier_model.predict(X_predict)[0]

    result_label = "Potentially Malicious" if prediction_class == 1 else "Benign"
    confidence = round(prediction_proba[prediction_class] * 100, 2)

    # --- Explainability (XAI) ---
    # For Logistic Regression, coefficients indicate feature importance.
    # Positive coefficients increase likelihood of positive class (malicious).
    # Negative coefficients increase likelihood of negative class (benign).
    
    feature_names_in_order = numerical_features_list + model_tld_features + text_feature_names.tolist()
    
    feature_importances = classifier_model.coef_[0] # Coefficients for the positive class (malicious)

    # Map coefficients to feature names
    feature_importance_map = dict(zip(feature_names_in_order, feature_importances))

    # Identify top contributing features for malicious classification
    # Filter for features present in the current URL and having a positive coefficient
    relevant_features = {}
    for feature_name, coef in feature_importance_map.items():
        if feature_name in numerical_features_list:
            if numerical_data[feature_name].iloc[0] > 0 and coef > 0: # Check if feature is active and positive coef
                relevant_features[feature_name] = coef
        elif feature_name.startswith('tld_'):
            if tld_features_for_prediction[feature_name].iloc[0] == 1 and coef > 0:
                relevant_features[feature_name] = coef
        else: # Text features
            # For text features, check if the word is present in the URL's text content
            if feature_name in text_content.lower() and coef > 0:
                relevant_features[feature_name] = coef

    # Sort by absolute coefficient value for display
    sorted_contributing_features = sorted(relevant_features.items(), key=lambda item: abs(item[1]), reverse=True)
    
    # Prepare data for graph
    graph_data = []
    for feature, coef in sorted_contributing_features[:10]: # Top 10 contributing features
        graph_data.append({'feature': feature, 'importance': coef})

    return {
        'url': url_to_analyze,
        'prediction': result_label,
        'confidence': confidence,
        'feature_importance_graph_data': graph_data,
        'raw_features': new_url_features_dict # For debugging/display
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
    # Load or train model on startup
    # This will print status messages to the console
    app.run(debug=True, port=5000)