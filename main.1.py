# f1codesafe2.py
import pandas as pd, numpy as np, time, pickle, re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import tldextract
import warnings
warnings.filterwarnings("ignore")

# ---------- URL numeric features (same as before)
class URLFeatureExtractor:
    def is_ip(self, url):
        try:
            net = urlparse(url).netloc.split(':')[0]
            parts = net.split('.')
            return 1 if len(parts)==4 and all(p.isdigit() for p in parts) else 0
        except:
            return 0
    def has_at(self, url): return 1 if '@' in url else 0

    def redirect_double_slash(self, url):
        pos = url.rfind('//'); return 1 if pos > 6 else 0
    def has_dash(self, url): return 1 if '-' in urlparse(url).netloc else 0
    def many_subdomains(self, url):
        ex = tldextract.extract(url); sub = ex.subdomain
        return 1 if sub and sub.count('.')>=1 else 0
    def long_url(self, url): return 1 if len(url) >= 54 else 0
    def shortener(self, url):
        return 1 if re.search(r"bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co|is\.gd|tiny\.cc|bitly\.com|cutt\.us", url) else 0
    def has_https(self, url): return 1 if urlparse(url).scheme == 'https' else 0
    def count_digits(self, url): return sum(c.isdigit() for c in url)
    def count_specials(self, url): return sum(1 for c in url if c in "-_?=&%$+~#@")
    def host_len(self, url): return len(urlparse(url).netloc)
    def get_features(self, url):
        if not urlparse(url).scheme:
            url = "http://" + url
        return {
            'is_ip': self.is_ip(url),
            'has_at': self.has_at(url),
            'redirect_double_slash': self.redirect_double_slash(url),
            'has_dash': self.has_dash(url),
            'many_subdomains': self.many_subdomains(url),
            'long_url': self.long_url(url),
            'shortener': self.shortener(url),
            'has_https': self.has_https(url),
            'count_digits': self.count_digits(url),
            'count_specials': self.count_specials(url),
            'host_len': self.host_len(url)
        }

# ---------- Load dataset
print("Loading dataset Main_dataset.csv ...")
df = pd.read_csv("Main_dataset.csv")
label_col = 'label' if 'label' in df.columns else ('Label' if 'Label' in df.columns else None)
url_col = 'url' if 'url' in df.columns else ('domain' if 'domain' in df.columns else None)
if label_col is None or url_col is None:
    raise Exception("Dataset must contain 'url' (or 'domain') and 'label' (or 'Label') columns.")
df = df.dropna(subset=[url_col, label_col]).drop_duplicates(subset=[url_col]).reset_index(drop=True)
print("Rows:", len(df))

# ---------- Build features
fe = URLFeatureExtractor()
print("Computing numeric features...")
feat_rows = [fe.get_features(u) for u in df[url_col].astype(str)]
feat_df = pd.DataFrame(feat_rows)

X_text = df[url_col].astype(str).rename("url_text")
X_num = feat_df
y = df[label_col].astype(int)

# Combined DataFrame for convenience
X_comb = pd.concat([X_text.reset_index(drop=True), X_num.reset_index(drop=True)], axis=1)

# ---------- Preprocessing pipeline:
# 1) TF-IDF (char ngrams) -> 2) TruncatedSVD (reduce to compact dense vector)
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=3000)  # keep somewhat large then reduce
svd = TruncatedSVD(n_components=100, random_state=42)  # reduce TF-IDF to 100 dims (adjustable)

tfidf_svd_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('svd', svd)
])

# ColumnTransformer: apply tfidf+svd to url_text and passthrough numeric features
preprocessor = ColumnTransformer(transformers=[
    ('text_svd', tfidf_svd_pipeline, 'url_text'),
    ('num', 'passthrough', list(X_num.columns))
], remainder='drop')

# ---------- Classifier pipeline: after preprocessor we have a dense array (SVD produces dense) -> SMOTE works
rfc = RandomForestClassifier(n_estimators=120, class_weight='balanced', random_state=42, n_jobs=1)

imb_pipeline = ImbPipeline([
    ('pre', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', rfc)
])

# ---------- Hyperparam search (single-threaded, small)
param_dist = {
    'clf__n_estimators': [80, 120],
    'clf__max_depth': [8, 12, None],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 0.4]
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(imb_pipeline, param_dist, n_iter=6, scoring='roc_auc', cv=cv, random_state=42, n_jobs=1, verbose=2)

# ---------- Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_comb, y, test_size=0.20, stratify=y, random_state=42)
print("Training rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

# ---------- Fit (this should be memory-safe)
start = time.time()
search.fit(X_train, y_train)
print("Search done in %.1f sec" % (time.time() - start))
print("Best CV ROC AUC:", search.best_score_)
print("Best params:", search.best_params_)

# ---------- Evaluate holdout
best_pipe = search.best_estimator_
y_pred = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)[:,1]
print("\nHoldout metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, y_proba))
print("Confusion:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# --- Add plotting for label distribution and confusion matrix ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# 1) Label distribution (full dataset)
plt.figure(figsize=(8,6))
# prefer ordering: show '1' then '0' if present (like your sample)
order = sorted(list(y.unique()), reverse=True)
counts = y.value_counts().reindex(order)
sns.barplot(x=counts.index.astype(str), y=counts.values)
plt.title("Count (target)")
plt.xlabel("label")
plt.ylabel("Count")
for i, v in enumerate(counts.values):
    plt.text(i, v + max(counts.values)*0.01, str(int(v)), ha='center')
plt.tight_layout()
plt.savefig("label_distribution.png", dpi=150)
plt.show()

# 2) Confusion matrix heatmap (counts + colorbar)
cm = confusion_matrix(y_test, y_pred)  # rows: true, cols: pred
labels = [str(l) for l in sorted(np.unique(y_test))]  # ['0','1'] etc.

plt.figure(figsize=(7,6))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=labels, yticklabels=labels, cbar=True,
                 annot_kws={"size":14})
ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label", fontsize=12)
plt.title("Confusion Matrix", fontsize=13)

# optionally add normalized percentages in text (under the counts)
cm_sum = cm.sum()
if cm_sum > 0:
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i,j] / cm[i].sum() if cm[i].sum() > 0 else 0
            ax.text(j + 0.25, i + 0.6, f"{pct:.2%}", color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=10)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()


# ---------- Save artifact
artifact = {'pipeline': best_pipe, 'num_cols': list(X_num.columns)}
with open('phishing_pipeline_svd.pkl', 'wb') as f:
    pickle.dump(artifact, f)
print("Saved phishing_pipeline_svd.pkl")

# ---------- Simple predictor
def predict_url(url, model_path='phishing_pipeline_svd.pkl'):
    with open(model_path, 'rb') as f:
        art = pickle.load(f)
    pipe = art['pipeline']
    fe = URLFeatureExtractor()
    fvals = fe.get_features(url)
    row = {'url_text': url}
    row.update(fvals)
    Xsingle = pd.DataFrame([row])
    prob = float(pipe.predict_proba(Xsingle)[0][1])
    pred = int(prob >= 0.5)
    return {'label': pred, 'prob': prob, 'label_str': 'Given website is a phishing site' if pred==1 else 'Given website is a legitimate site'}

if __name__ == "__main__":
    while True:
        u = input("Enter URL (or 'quit'): ").strip()
        if u.lower() in ('quit','exit'): break
        print(predict_url(u))
