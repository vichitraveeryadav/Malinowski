import shutil
from pathlib import Path
from datetime import datetime
import json

class FileOrganizer:
    """This class organizes files into neat folders"""
    
    def __init__(self, base_path="processed"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)  # Create the folder if it doesn't exist

    def create_user_structure(self, username):
        """Create folder structure for a specific user"""
        user_path = self.base_path / username
        user_path.mkdir(exist_ok=True)
        
        # Create subfolders for different document types
        doc_types = ["passport", "visa", "permit", "certificate", "identification", "other"]
        for doc_type in doc_types:
            (user_path / doc_type).mkdir(exist_ok=True)
        
        return user_path

    def organize_document(self, username, document_type, source_file, extracted_data):
        """Move and organize a processed document"""
        # Create user folder structure
        user_path = self.create_user_structure(username)
        
        # Pick the right folder for this document type
        doc_folder = user_path / (document_type if (user_path / document_type).exists() else "other")
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_path = Path(source_file)
        new_filename = f"{document_type}_{timestamp}{source_path.suffix}"
        
        # Copy file to organized location
        destination = doc_folder / new_filename
        shutil.copy2(source_path, destination)
        
        # Create a metadata file with information about this document
        metadata = {
            "original_filename": source_path.name,
            "processed_date": datetime.now().isoformat(),
            "document_type": document_type,
            "extracted_data": extracted_data
        }
        
        # Save metadata as JSON file
        with open(doc_folder / f"{new_filename}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        return str(destination)
