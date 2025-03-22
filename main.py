from fastapi import FastAPI, Request, Depends, Form, HTTPException
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import SessionLocal, ShopKeeper, MainSupplier

from pydantic import BaseModel, EmailStr
from bcrypt import hashpw, gensalt

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/shopkeeper_signup")
async def signup_page(request: Request):
    return templates.TemplateResponse("shopkeeper_signup.html", {"request": request})

@app.get("/supplier_signup")
async def supplier_signup_page(request: Request):
    return templates.TemplateResponse("supplier_signup.html", {"request": request})

# Shopkeeper Signup Model
class ShopkeeperSignup(BaseModel):
    name: str
    email: EmailStr
    shop_name: str
    location_name: str
    latitude: float
    longitude: float
    domain: str
    password: str

# Supplier Signup Model
class SupplierSignup(BaseModel):
    name: str
    email: EmailStr
    company_name: str
    location_name: str
    latitude: float
    longitude: float
    password: str

@app.post("/submit_shopkeeper/")
async def submit_shopkeeper(
    name: str = Form(...),
    email: EmailStr = Form(...),
    shop_name: str = Form(...),
    location_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    domain: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    existing_shopkeeper = db.query(ShopKeeper).filter(ShopKeeper.email == email).first()
    if existing_shopkeeper:
        raise HTTPException(status_code=400, detail="Email already registered!")
    
    password_hash = hashpw(password.encode('utf-8'), gensalt()).decode('utf-8')
    new_shopkeeper = ShopKeeper(
        name=name,
        email=email,
        shop_name=shop_name,
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        domain=domain,
        password_hash=password_hash
    )
    db.add(new_shopkeeper)
    db.commit()
    db.refresh(new_shopkeeper)
    return {"message": "Shopkeeper Sign-up successful!", "shop_keeper_id": new_shopkeeper.shop_keeper_id}

@app.post("/submit_supplier/")
async def submit_supplier(
    name: str = Form(...),
    company_name: str = Form(...),
    location_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    password: str = Form(...),
    industry: str = Form(None), 
    db: Session = Depends(get_db)
):
    
    password_hash = hashpw(password.encode('utf-8'), gensalt()).decode('utf-8')
    new_supplier = MainSupplier(
        name=name,
        company_name=company_name,
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        industry=industry, 
        password_hash=password_hash
    )
    db.add(new_supplier)
    db.commit()
    db.refresh(new_supplier)
    return {"message": "Supplier Sign-up successful!", "supplier_id": new_supplier.supplier_id}

