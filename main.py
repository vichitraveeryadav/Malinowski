import streamlit as st
import os
from PIL import Image
import json
from pathlib import Path

# Import all our custom modules
from ocr_processor import OCRProcessor
from document_classifier import DocumentClassifier
from data_extractor import DataExtractor
from file_organizer import FileOrganizer
from database import get_db_session, init_database, User, Document

@st.cache_resource
def get_app_services(use_tesseract=True, use_easyocr=True, use_paddle=False):
    """Load all our tools once and keep them in memory"""
    ocr = OCRProcessor(use_tesseract=use_tesseract, use_easyocr=use_easyocr, use_paddle=use_paddle)
    clf = DocumentClassifier()
    extractor = DataExtractor()
    organizer = FileOrganizer()
    return ocr, clf, extractor, organizer

class ImmigrationApp:
    """The main application class"""
    
    def __init__(self, use_tesseract=True, use_easyocr=True, use_paddle=False):
        # Load all our tools
        self.ocr_processor, self.doc_classifier, self.data_extractor, self.file_organizer = \
            get_app_services(use_tesseract, use_easyocr, use_paddle)

    def process_uploaded_file(self, uploaded_file, username, user_id=1):
        """Process a single uploaded file - this is where the magic happens!"""
        temp_path = Path(f"temp_{uploaded_file.name}")
        try:
            # Step 1: Save the uploaded file temporarily
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Step 2: Extract text from the image
            st.write("🔍 Extracting text from document...")
            ocr_result = self.ocr_processor.process_document(temp_path)
            
            if not ocr_result.get('text'):
                st.error(f"Could not extract text. Details: {ocr_result.get('error', 'no text found')}")
                return None

            # Step 3: Figure out what type of document it is
            st.write("📋 Classifying document type...")
            classification = self.doc_classifier.classify_document(ocr_result['text'])

            # Step 4: Extract specific information from the text
            st.write("📊 Extracting structured data...")
            structured_data = self.data_extractor.extract_structured_data(
                ocr_result['text'], classification['document_type']
            )

            # Step 5: Organize the file into proper folders
            st.write("📁 Organizing file...")
            organized_path = self.file_organizer.organize_document(
                username, classification['document_type'], temp_path, structured_data
            )

            # Step 6: Save information to database
            db_session = get_db_session()
            try:
                # Make sure user exists in database
                user = db_session.query(User).filter_by(username=username).first()
                if not user:
                    user = User(username=username, email=None)
                    db_session.add(user)
                    db_session.commit()

                # Save document information
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

            # Return all the results
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
            # Clean up temporary file
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

def main():
    """The main function that runs our app"""
    # Set up the page
    st.set_page_config(
        page_title="Immigration Document Automation", 
        page_icon="📄", 
        layout="wide"
    )
    
    st.title("🏛️ Immigration Document Automation System")
    st.markdown("**Automate document processing with OCR and AI classification**")

    # Make sure database tables exist
    init_database()

    # Sidebar settings
    st.sidebar.title("Settings")
    username = st.sidebar.text_input("Username", value="user123")
    
    st.sidebar.write("Select OCR engines (enable one at a time if memory is low):")
    use_tesseract = st.sidebar.checkbox("Tesseract (eng+hin)", value=True)
    use_easyocr = st.sidebar.checkbox("EasyOCR (en+hi)", value=True) 
    use_paddle = st.sidebar.checkbox("PaddleOCR (en)", value=False)

    if not username:
        st.warning("Please enter a username to continue")
        return

    # Create the main app
    app = ImmigrationApp(use_tesseract=use_tesseract, use_easyocr=use_easyocr, use_paddle=use_paddle)

    # Create three tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "📊 View Results", "📁 Manage Files"])

    # TAB 1: Upload Documents
    with tab1:
        st.header("Upload Immigration Documents")
        st.info("For first deploy, upload images only (PNG/JPG). PDF support can be added later.")
        
        uploaded_files = st.file_uploader(
            "Choose image files to process",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.subheader(f"Processing: {uploaded_file.name}")
                
                # Show image preview
                if uploaded_file.type.startswith('image'):
                    try:
                        image = Image.open(uploaded_file)
                        st.image(image, width=300)
                    except Exception:
                        st.write("Preview unavailable, proceeding to OCR.")
                
                # Process button
                if st.button(f"Process {uploaded_file.name}", key=f"btn_{uploaded_file.name}"):
                    with st.spinner("Processing..."):
                        result = app.process_uploaded_file(uploaded_file, username)
                    
                    if result:
                        st.success("✅ Document processed successfully!")
                        
                        # Show results in two columns
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

    # TAB 2: View Results
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
                            st.write(f"**Path:** {doc.file_path}")
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

    # TAB 3: File Management
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
