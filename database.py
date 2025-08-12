from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from config import DATABASE_URL

Base = declarative_base()

# Table for storing user information
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Table for storing document information
class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    filename = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    extracted_text = Column(Text)
    structured_data = Column(Text)  # JSON string
    confidence_score = Column(Float)
    processed_at = Column(DateTime, default=datetime.utcnow)

def init_database():
    """Create the database and tables"""
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine

def get_db_session():
    """Get a database session to save/load data"""
    engine = init_database()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal()
