import re
from datetime import datetime

class DataExtractor:
    """This class finds specific information in document text"""
    
    def __init__(self):
        # Patterns to find specific information
        self.patterns = {
            "passport_number": r'\b[A-Z]{1,2}[0-9]{6,8}\b',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            "name": r'\b[A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})?\b',
            "visa_number": r'\b[A-Z0-9]{8,12}\b',
            "phone": r'\+?[0-9]{10,13}',
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "address": r'\d+[^,\n]*(?:street|st|road|rd|avenue|ave|lane|ln)[^,\n]*'
        }

    def extract_passport_data(self, text):
        """Find passport-specific information"""
        data = {}
        
        # Look for passport number
        m = re.search(self.patterns['passport_number'], text, re.IGNORECASE)
        if m:
            data['passport_number'] = m.group()
        
        # Look for dates
        dates = re.findall(self.patterns['date'], text)
        if dates:
            data['dates_found'] = dates
        
        # Look for names
        names = re.findall(self.patterns['name'], text)
        if names:
            data['names_found'] = names[:3]  # Keep only first 3 names
        
        return data

    def extract_visa_data(self, text):
        """Find visa-specific information"""
        data = {}
        
        # Look for visa number
        m = re.search(self.patterns['visa_number'], text, re.IGNORECASE)
        if m:
            data['visa_number'] = m.group()
        
        # Look for validity dates
        dates = re.findall(self.patterns['date'], text)
        if dates:
            data['validity_dates'] = dates
        
        return data

    def extract_structured_data(self, text, document_type):
        """Extract all relevant information based on document type"""
        # Basic information for all documents
        base = {
            "extraction_date": datetime.now().isoformat(),
            "raw_text_length": len(text),
            "document_type": document_type
        }
        
        # Add specific information based on document type
        if document_type == "passport":
            base.update(self.extract_passport_data(text))
        elif document_type == "visa":
            base.update(self.extract_visa_data(text))
        
        # Look for common information in all documents
        email = re.search(self.patterns['email'], text, re.IGNORECASE)
        if email:
            base['email'] = email.group()
        
        phone = re.search(self.patterns['phone'], text)
        if phone:
            base['phone'] = phone.group()
        
        return base
