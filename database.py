from sqlalchemy import create_engine, Column, String, Integer, DECIMAL, TIMESTAMP, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv  
from bcrypt import hashpw, gensalt  

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL is not set. Check your .env file and environment variables.")

print("DATABASE_URL:", DATABASE_URL) 

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Main Suppliers Table
class MainSupplier(Base):
    __tablename__ = "main_suppliers"
    
    supplier_id = Column(String(10), primary_key=True,  autoincrement=True)
    name = Column(String(100), nullable=False)
    company_name = Column(String(100), nullable=False)
    location_name = Column(String(100), nullable=False)
    latitude = Column(DECIMAL(9, 6))
    longitude = Column(DECIMAL(9, 6))
    industry = Column(String(50))
    password_hash = Column(String(255), nullable=False)  
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    shop_keeper = relationship("ShopKeeper", back_populates="main_supplier")

# Shopkeeper Table
class ShopKeeper(Base):
    __tablename__ = "shop_keeper"
    
    shop_keeper_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    shop_name = Column(String(100), nullable=False)
    location_name = Column(String(100), nullable=False)
    latitude = Column(DECIMAL(9, 6))
    longitude = Column(DECIMAL(9, 6))
    main_supplier_id = Column(String(10), ForeignKey("main_suppliers.supplier_id"), nullable=True)  # Default NULL
    domain = Column(String(50))
    password_hash = Column(String(255), nullable=False)  
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    main_supplier = relationship("MainSupplier", back_populates="shop_keeper")

# Create Tables in Database
Base.metadata.create_all(bind=engine)
