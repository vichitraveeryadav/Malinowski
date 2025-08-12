from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import re
from pathlib import Path

class DocumentClassifier:
    """This class identifies what type of document it is"""
    
    def __init__(self):
        self.model_path = Path("document_classifier.pkl")
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('nb', MultinomialNB())
        ])
        self.is_trained = False

    def prepare_training_data(self):
        """Create example data to teach the AI what each document type looks like"""
        training_data = [
            ("passport number personal details", "passport"),
            ("visa entry permit immigration", "visa"),
            ("work permit employment authorization", "permit"),
            ("birth certificate date of birth", "certificate"),
            ("driver license identification card", "identification"),
            ("travel document immigration status", "passport"),
            ("study permit student visa", "permit"),
            ("marriage certificate spouse", "certificate"),
            ("temporary resident visa", "visa"),
            ("permanent resident card", "identification")
        ]
        texts, labels = zip(*training_data)
        return list(texts), list(labels)

    def train_classifier(self):
        """Teach the AI to recognize document types"""
        texts, labels = self.prepare_training_data()
        self.classifier.fit(texts, labels)
        self.is_trained = True
        
        # Save the trained model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_classifier(self):
        """Load a previously trained model, or create a new one"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                self.is_trained = True
            else:
                self.train_classifier()
        except Exception:
            self.train_classifier()

    def classify_document(self, text):
        """Identify what type of document this text came from"""
        if not self.is_trained:
            self.load_classifier()
        
        # Clean the text
        cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        
        # Make prediction
        proba = self.classifier.predict_proba([cleaned_text])
        pred = self.classifier.classes_[proba.argmax()]
        confidence = float(proba.max())
        
        return {"document_type": pred, "confidence": confidence}
