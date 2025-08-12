import streamlit as st
import os
from PIL import Image
import json
from pathlib import Path
import re
from datetime import datetime
import shutil
import pickle

# Basic imports only - NO EasyOCR
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path.cwd()
DATABASE_URL = "sqlite:///immigration_docs.db"

DOCUMENT_TYPES = {
    "passport": ["passport", "travel document"],
    "visa": ["visa", "entry permit"], 
    "permit": ["work permit", "study permit"],
    "certificate": ["birth certificate", "marriage certificate"],
    "identification": ["driver license", "id card"]
}

# ============================================================================
# DATABASE SETUP
# ============================================================================

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    filename = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    extracted_text = Column(Text)
    structured_data = Column(Text)
    confidence_score = Column(Float)
    processed_at = Column(DateTime, default=datetime.utcnow)

def init_database():
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine

def get_db_session():
    engine = init_database()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal()

# ============================================================================
# SIMPLE OCR PROCESSOR (TESSERACT ONLY)
# ============================================================================

class OCRProcessor:
    def __init__(self):
        self.languages = "eng+hin"  # English + Hindi

    def process_document(self, image_path):
        """Extract text using only Tesseract OCR"""
        try:
            # Open image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text with Tesseract
            config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, lang=self.languages, config=config)
            
            return {
                "text": text.strip(), 
                "confidence": 0.8,
                "engine": "tesseract"
            }
            
        except Exception as e:
            return {
                "text": "", 
                "confidence": 0.0, 
                "error": str(e),
                "engine": "tesseract"
            }

# ============================================================================
# DOCUMENT CLASSIFIER
# ============================================================================

class DocumentClassifier:
    def __init__(self):
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
            ('nb', MultinomialNB())
        ])
        self.is_trained = False

    def prepare_training_data(self):
        training_data = [
            ("passport number personal details republic india", "passport"),
            ("visa entry permit immigration canada", "visa"),
            ("work permit employment authorization", "permit"),
            ("birth certificate date of birth", "certificate"),
            ("driver license identification card", "identification"),
            ("travel document immigration status", "passport"),
            ("study permit student visa education", "permit"),
            ("marriage certificate spouse husband wife", "certificate"),
            ("temporary resident visa immigration", "visa"),
            ("permanent resident card citizenship", "identification"),
            ("passport issued government travel", "passport"),
            ("visa stamp entry country", "visa")
        ]
        texts, labels = zip(*training_data)
        return list(texts), list(labels)

    def train_classifier(self):
        texts, labels = self.prepare_training_data()
        self.classifier.fit(texts, labels)
        self.is_trained = True

    def classify_document(self, text):
        if not self.is_trained:
            self.train_classifier()
        
        # Clean text
        cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        
        # Predict
        try:
            proba = self.classifier.predict_proba([cleaned_text])
            pred = self.classifier.classes_[proba.argmax()]
            confidence = float(proba.max())
        except:
            pred = "unknown"
            confidence = 0.5
        
        return {"document_type": pred, "confidence": confidence}

# ============================================================================
# DATA EXTRACTOR
# ============================================================================

class DataExtractor:
    def __init__(self):
        self.patterns = {
            "passport_number": r'\b[A-Z]{1,2}[0-9]{6,8}\b',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            "name": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            "visa_number": r'\b[A-Z0-9]{8,12}\b',
            "phone": r'\+?[0-9]{10,13}',
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        }

    def extract_structured_data(self, text, document_type):
        base = {
            "extraction_date": datetime.now().isoformat(),
            "raw_text_length": len(text),
            "document_type": document_type
        }
        
        # Find passport number
        if document_type == "passport":
            m = re.search(self.patterns['passport_number'], text, re.IGNORECASE)
            if m:
                base['passport_number'] = m.group()
        
        # Find visa number
        if document_type == "visa":
            m = re.search(self.patterns['visa_number'], text, re.IGNORECASE)
            if m:
                base['visa_number'] = m.group()
        
        # Find dates
        dates = re.findall(self.patterns['date'], text)
        if dates:
            base['dates_found'] = dates[:3]  # Limit to 3 dates
        
        # Find names
        names = re.findall(self.patterns['name'], text)
        if names:
            base['names_found'] = names[:2]  # Limit to 2 names
        
        # Find email
        email = re.search(self.patterns['email'], text, re.IGNORECASE)
        if email:
            base['email'] = email.group()
        
        return base

# ============================================================================
# FILE ORGANIZER
# ============================================================================

class FileOrganizer:
    def __init__(self, base_path="processed"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

    def organize_document(self, username, document_type, source_file, extracted_data):
        # Create user folder
        user_path = self.base_path / username
        user_path.mkdir(exist_ok=True)
        
        # Create document type folder
        doc_folder = user_path / document_type
        doc_folder.mkdir(exist_ok=True)
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_path = Path(source_file)
        new_filename = f"{document_type}_{timestamp}{source_path.suffix}"
        
        # Copy file
        destination = doc_folder / new_filename
        shutil.copy2(source_path, destination)
        
        return str(destination)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

@st.cache_resource
def get_app_services():
    ocr = OCRProcessor()
    clf = DocumentClassifier()
    extractor = DataExtractor()
    organizer = FileOrganizer()
    return ocr, clf, extractor, organizer

class ImmigrationApp:
    def __init__(self):
        self.ocr_processor, self.doc_classifier, self.data_extractor, self.file_organizer = \
            get_app_services()

    def process_uploaded_file(self, uploaded_file, username):
        temp_path = Path(f"temp_{uploaded_file.name}")
        try:
            # Save file
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Step 1: OCR
            st.write("🔍 Extracting text from document...")
            ocr_result = self.ocr_processor.process_document(temp_path)
            
            if not ocr_result.get('text'):
                st.error(f"Could not extract text: {ocr_result.get('error', 'Unknown error')}")
                return None

            # Step 2: Classification
            st.write("📋 Classifying document type...")
            classification = self.doc_classifier.classify_document(ocr_result['text'])

            # Step 3: Data extraction
            st.write("📊 Extracting structured data...")
            structured_data = self.data_extractor.extract_structured_data(
                ocr_result['text'], classification['document_type']
            )

            # Step 4: File organization
            st.write("📁 Organizing file...")
            organized_path = self.file_organizer.organize_document(
                username, classification['document_type'], temp_path, structured_data
            )

            # Step 5: Database
            try:
                db_session = get_db_session()
                
                # Ensure user exists
                user = db_session.query(User).filter_by(username=username).first()
                if not user:
                    user = User(username=username)
                    db_session.add(user)
                    db_session.commit()

                # Save document
                new_doc = Document(
                    user_id=user.id,
                    filename=uploaded_file.name,
                    document_type=classification['document_type'],
                    file_path=organized_path,
                    extracted_text=ocr_result['text'][:1000],  # Limit text length
                    structured_data=json.dumps(structured_data),
                    confidence_score=float(ocr_result.get('confidence', 0.0))
                )
                db_session.add(new_doc)
                db_session.commit()
                db_session.close()
            except Exception as e:
                st.warning(f"Database save failed: {str(e)}")

            return {
                "ocr_result": ocr_result,
                "classification": classification,
                "structured_data": structured_data,
                "organized_path": organized_path
            }
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
        finally:
            # Cleanup
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except:
                    pass

def main():
    st.set_page_config(
        page_title="Immigration Document Automation", 
        page_icon="📄", 
        layout="wide"
    )
    
    st.title("🏛️ Immigration Document Automation System")
    st.markdown("**Simple OCR-powered document processing using Tesseract**")

    # Initialize database
    try:
        init_database()
    except Exception as e:
        st.warning(f"Database initialization issue: {str(e)}")

    # Sidebar
    st.sidebar.title("Settings")
    username = st.sidebar.text_input("Username", value="user123")
    st.sidebar.info("🔧 Using Tesseract OCR (English + Hindi)")

    if not username:
        st.warning("Please enter a username to continue")
        return

    app = ImmigrationApp()

    # Main tabs
    tab1, tab2 = st.tabs(["📤 Upload Documents", "📊 View Results"])

    with tab1:
        st.header("Upload Immigration Documents")
        st.info("📸 Upload clear images (PNG/JPG) of your documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.subheader(f"📄 {uploaded_file.name}")
                
                # Show preview
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, width=400)
                except:
                    st.write("Preview not available")
                
                # Process button
                if st.button(f"🚀 Process", key=f"btn_{uploaded_file.name}"):
                    with st.spinner("Processing document..."):
                        result = app.process_uploaded_file(uploaded_file, username)
                    
                    if result:
                        st.success("✅ Document processed successfully!")
                        
                        # Results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📋 Classification")
                            st.write(f"**Type:** {result['classification']['document_type']}")
                            st.write(f"**Confidence:** {result['classification']['confidence']:.1%}")
                        
                        with col2:
                            st.subheader("📊 Extracted Info")
                            st.json(result['structured_data'])
                        
                        # Text
                        with st.expander("📄 View Extracted Text"):
                            st.text_area("", value=result['ocr_result']['text'], height=150)

    with tab2:
        st.header("Processing History")
        
        try:
            db_session = get_db_session()
            documents = db_session.query(Document).order_by(Document.processed_at.desc()).limit(10).all()
            
            if documents:
                for doc in documents:
                    with st.expander(f"📄 {doc.filename} ({doc.document_type})"):
                        st.write(f"**Processed:** {doc.processed_at}")
                        st.write(f"**Confidence:** {doc.confidence_score:.1%}")
                        if doc.structured_data:
                            try:
                                data = json.loads(doc.structured_data)
                                st.json(data)
                            except:
                                pass
            else:
                st.info("No documents processed yet.")
            
            db_session.close()
        except Exception as e:
            st.warning(f"Could not load history: {str(e)}")

if __name__ == "__main__":
    main()
