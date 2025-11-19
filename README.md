# Phishing-attack-detection-browser-extension
Phishing is one of the most common cyberattacks where attackers trick users into revealing sensitive information like passwords or financial data.
Traditional techniques detect phishing using either URL patterns, domain details, or content analysis—but these methods alone are not enough.

This project introduces a Hybrid Feature–Based Phishing Detection System integrated into a browser extension for real-time protection.

Data Collection

Collected phishing and legitimate URLs from Kaggle, PhishTank, etc.

Crawled web pages to collect HTML and DOM information.

Balanced the dataset to avoid model bias.

2. Feature Engineering (Hybrid Features)

🔗 URL-based features

URL length

IP address usage

Suspicious tokens

Number of subdomains

Encoded characters

Entropy metrics

🌐 Domain-based features

Domain age

WHOIS details (registrar, expiry date, etc.)

📄 HTML/DOM features

Number of forms

Hidden fields

Suspicious iframes

Login fields detection (username/password)

🧷 Rule-based signals

Blacklist checks

Keyword matching

Known phishing templates

3. Model Development

Supervised ML using Random Forest Classifier.

Followed the pipeline:
Dataset → Preprocessing → Feature Engineering → Training → Validation → Export

The final model:

📌 Accuracy: 95%
📌 ROC-AUC: 0.98
📌 Saved as: phishing_classifier.pkl

Results

Dataset distribution visualized (phishing vs. legitimate).

Confusion matrix shows high classification accuracy.

Achieved 95% accuracy and 0.98 ROC-AUC.

Successfully tested real-time predictions (e.g., facebook.com correctly classified as legitimate).

Console-based URL testing also validated predictions.

🔮 Future Implementation

Incorporate deep-learning models (CNNs, RNNs, transformers).

Improve browser extension UI and detection speed.

Add real-time domain reputation lookup.

Deploy backend API for remote model updates.

Provide user-safe browsing notifications across multiple browsers (Firefox, Edge).
