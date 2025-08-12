import streamlit as st
import os
from PIL import Image
import json
from pathlib import Path
import re
from datetime import datetime
import shutil
import pickle

# All imports at the top
import pytesseract
import easyocr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path.cwd()
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
DATABASE_URL = "sqlite:///immigration_docs.db"

DOCUMENT_TYPES = {
    "passport": ["passport", "travel document"],
    "visa": ["visa", "entry permit"], 
    "permit": ["work permit", "study permit"],
    "certificate": ["birth certificate", "marriage certificate"],
    "identification": ["driver license", "id card"]
}

OCR_LANGUAGES_TESSERACT = ["eng", "hin"]
OCR_LANGUAGES_EASYOCR = ["en", "hi"]

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
# OCR PROCESSOR
# ============================================================================

class OCRProcessor:
    def __init__(self, use_tesseract=True, use_easyocr=True):
        self.use_tesseract = use_tesseract
        self.use_easyocr = use_easyocr
        self._easyocr_reader = None

    def _get_easyocr(self):
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(OCR_LANGUAGES_EASYOCR)
        return self._easyocr_reader

    def tesseract_ocr(self, image_path):
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            lang_string = '+'.join(OCR_LANGUAGES_TESSERACT)
            config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, lang=lang_string, config=config)
            return {"text": text.strip(), "confidence": 0.8}
        except Exception as e:
            return {"text": "", "confidence": 0.0, "error": str(e)}

    def easyocr_extract(self, image_path):
        try:
            reader = self._get_easyocr()
            result = reader.readtext(str(image_path))
            
            full_text = ""
            total_confidence = 0.0
            count = 0
            
            for (_, text, conf) in result:
                full_text += text + " "
                total_confidence += float(conf)
                count += 1
            
            avg_confidence = (total_confidence / count) if count else 0.0
            return {"text": full_text.strip(), "confidence": avg_confidence}
        except Exception as e:
            return {"text": "", "confidence": 0.0, "error": str(e)}

    def process_document(self, image_path):
        results = {}
        
        if self.use_tesseract:
            results["tesseract"] = self.tesseract_ocr(image_path)
        if self.use_easyocr:
            results["easyocr"] = self.easyocr_extract(image_path)
        
        if not results:
            return {"text": "", "confidence": 0.0, "all_results": {}}
        
        best_result = max(results.values(), key=lambda x: x.get("confidence", 0.0))
        best_result["all_results"] = results
        return best_result

# ============================================================================
# DOCUMENT CLASSIFIER
# ============================================================================

class DocumentClassifier:
    def __init__(self):
        self.model_path = Path("document_classifier.pkl")
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('nb', MultinomialNB())
        ])
        self.is_trained = False

    def prepare_training_data(self):
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
        texts, labels = self.prepare_training_data()
        self.classifier.fit(texts, labels)
        self.is_trained = True
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.classifier, f)
        except:
            pass

    def load_classifier(self):
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
        if not self.is_trained:
            self.load_classifier()
        
        cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        proba = self.classifier.predict_proba([cleaned_text])
        pred = self.classifier.classes_[proba.argmax()]
        confidence = float(proba.max())
        return {"document_type": pred, "confidence": confidence}

# ============================================================================
# DATA EXTRACTOR
# ============================================================================

class DataExtractor:
    def __init__(self):
        self.patterns = {
            "passport_number": r'\b[A-Z]{1,2}[0-9]{6,8}\b',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            "name": r'\b[A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})?\b',
            "visa_number": r'\b[A-Z0-9]{8,12}\b',
            "phone": r'\+?[0-9]{10,13}',
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        }

    def extract_passport_data(self, text):
        data = {}
        m = re.search(self.patterns['passport_number'], text, re.IGNORECASE)
        if m:
            data['passport_number'] = m.group()
        
        dates = re.findall(self.patterns['date'], text)
        if dates:
            data['dates_found'] = dates
        
        names = re.findall(self.patterns['name'], text)
        if names:
            data['names_found'] = names[:3]
        return data

    def extract_visa_data(self, text):
        data = {}
        m = re.search(self.patterns['visa_number'], text, re.IGNORECASE)
        if m:
            data['visa_number'] = m.group()
        
        dates = re.findall(self.patterns['date'], text)
        if dates:
            data['validity_dates'] = dates
        return data

    def extract_structured_data(self, text, document_type):
        base = {
            "extraction_date": datetime.now().isoformat(),
            "raw_text_length": len(text),
            "document_type": document_type
        }
        
        if document_type == "passport":
            base.update(self.extract_passport_data(text))
        elif document_type == "visa":
            base.update(self.extract_visa_data(text))
        
        email = re.search(self.patterns['email'], text, re.IGNORECASE)
        if email:
            base['email'] = email.group()
        
        phone = re.search(self.patterns['phone'], text)
        if phone:
            base['phone'] = phone.group()
        
        return base

# ============================================================================
# FILE ORGANIZER
# ============================================================================

class FileOrganizer:
    def __init__(self, base_path="processed"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

    def create_user_structure(self, username):
        user_path = self.base_path / username
        user_path.mkdir(exist_ok=True)
        
        doc_types = ["passport", "visa", "permit", "certificate", "identification", "other"]
        for doc_type in doc_types:
            (user_path / doc_type).mkdir(exist_ok=True)
        
        return user_path

    def organize_document(self, username, document_type, source_file, extracted_data):
        user_path = self.create_user_structure(username)
        doc_folder = user_path / (document_type if (user_path / document_type).exists() else "other")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_path = Path(source_file)
        new_filename = f"{document_type}_{timestamp}{source_path.suffix}"
        
        destination = doc_folder / new_filename
        shutil.copy2(source_path, destination)
        
        metadata = {
            "original_filename": source_path.name,
            "processed_date": datetime.now().isoformat(),
            "document_type": document_type,
            "extracted_data": extracted_data
        }
        
        try:
            with open(doc_folder / f"{new_filename}.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except:
            pass
        
        return str(destination)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

@st.cache_resource
def get_app_services(use_tesseract=True, use_easyocr=True):
    ocr = OCRProcessor(use_tesseract=use_tesseract, use_easyocr=use_easyocr)
    clf = DocumentClassifier()
    extractor = DataExtractor()
    organizer = FileOrganizer()
    return ocr, clf, extractor, organizer

class ImmigrationApp:
    def __init__(self, use_tesseract=True, use_easyocr=True):
        self.ocr_processor, self.doc_classifier, self.data_extractor, self.file_organizer = \
            get_app_services(use_tesseract, use_easyocr)

    def process_uploaded_file(self, uploaded_file, username, user_id=1):
        temp_path = Path(f"temp_{uploaded_file.name}")
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.write("🔍 Extracting text from document...")
            ocr_result = self.ocr_processor.process_document(temp_path)
            
            if not ocr_result.get('text'):
                st.error(f"Could not extract text. Details: {ocr_result.get('error', 'no text found')}")
                return None

            st.write("📋 Classifying document type...")
            classification = self.doc_classifier.classify_document(ocr_result['text'])

            st.write("📊 Extracting structured data...")
            structured_data = self.data_extractor.extract_structured_data(
                ocr_result['text'], classification['document_type']
            )

            st.write("📁 Organizing file...")
            organized_path = self.file_organizer.organize_document(
                username, classification['document_type'], temp_path, structured_data
            )

            db_session = get_db_session()
            try:
                user = db_session.query(User).filter_by(username=username).first()
                if not user:
                    user = User(username=username, email=None)
                    db_session.add(user)
                    db_session.commit()

                new_doc = Document(
                    user_id=user.id,
                    filename=uploaded_file.name,
                    document_type=classification['document_type'],
                    file_path=organized_path,
                    extracted_text=ocr_result['text'],
                    structured_data=json.dumps(structured_data),
                    confidence_score=float(ocr_result.get('confidence', 0.0))
                )
                db_session.add(new_doc)
                db_session.commit()
            finally:
                db_session.close()

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
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

def main():
    st.set_page_config(
        page_title="Immigration Document Automation", 
        page_icon="📄", 
        layout="wide"
    )
    
    st.title("🏛️ Immigration Document Automation System")
    st.markdown("**Automate document processing with OCR and AI classification**")

    init_database()

    st.sidebar.title("Settings")
    username = st.sidebar.text_input("Username", value="user123")
    
    st.sidebar.write("Select OCR engines:")
    use_tesseract = st.sidebar.checkbox("Tesseract (eng+hin)", value=True)
    use_easyocr = st.sidebar.checkbox("EasyOCR (en+hi)", value=True)

    if not username:
        st.warning("Please enter a username to continue")
        return

    app = ImmigrationApp(use_tesseract=use_tesseract, use_easyocr=use_easyocr)

    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "📊 View Results", "📁 Manage Files"])

    with tab1:
        st.header("Upload Immigration Documents")
        st.info("Upload images only (PNG/JPG) for now.")
        
        uploaded_files = st.file_uploader(
            "Choose image files to process",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.subheader(f"Processing: {uploaded_file.name}")
                
                if uploaded_file.type.startswith('image'):
                    try:
                        image = Image.open(uploaded_file)
                        st.image(image, width=300)
                    except Exception:
                        st.write("Preview unavailable.")
                
                if st.button(f"Process {uploaded_file.name}", key=f"btn_{uploaded_file.name}"):
                    with st.spinner("Processing..."):
                        result = app.process_uploaded_file(uploaded_file, username)
                    
                    if result:
                        st.success("✅ Document processed successfully!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📋 Classification Results")
                            st.write(f"**Document Type:** {result['classification']['document_type']}")
                            st.write(f"**Confidence:** {result['classification']['confidence']:.2f}")
                        
                        with col2:
                            st.subheader("📊 Extracted Data")
                            st.json(result['structured_data'])
                        
                        st.subheader("📄 Extracted Text")
                        with st.expander("View extracted text"):
                            st.text_area("Text", value=result['ocr_result']['text'], height=200)

    with tab2:
        st.header("Processing Results")
        
        db_session = get_db_session()
        try:
            documents = db_session.query(Document).all()
            if documents:
                for doc in documents:
                    with st.expander(f"{doc.filename} - {doc.document_type}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Type:** {doc.document_type}")
                            st.write(f"**Confidence:** {doc.confidence_score:.2f}")
                            st.write(f"**Processed:** {doc.processed_at}")
                        with col2:
                            if doc.structured_data:
                                try:
                                    structured = json.loads(doc.structured_data)
                                    st.json(structured)
                                except Exception:
                                    st.write(doc.structured_data)
            else:
                st.info("No documents processed yet.")
        finally:
            db_session.close()

    with tab3:
        st.header("File Management")
        
        processed_path = Path("processed")
        if processed_path.exists():
            st.subheader("📁 Organized Files")
            for user_folder in sorted(p for p in processed_path.iterdir() if p.is_dir()):
                st.write(f"**User:** {user_folder.name}")
                for doc_type_folder in sorted(p for p in user_folder.iterdir() if p.is_dir()):
                    files = list(doc_type_folder.glob("*.*"))
                    if files:
                        st.write(f"  📂 {doc_type_folder.name}: {len(files)} items")
        else:
            st.info("No processed files directory yet.")

if __name__ == "__main__":
    main()
