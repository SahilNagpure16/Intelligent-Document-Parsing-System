import os
import re
import cv2
import torch
import spacy
from PIL import Image
from datetime import datetime
import pytesseract
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# Setup
RESULT_FOLDER = "results"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load models
nlp = spacy.load("en_core_web_sm")
processor = RTDetrImageProcessor.from_pretrained("HuggingPanda/docling-layout")
model = RTDetrForObjectDetection.from_pretrained("HuggingPanda/docling-layout")

# Helpers
def extract_invoice_number(text):
    match = re.search(r"invoice[\s#:]*([A-Z0-9-]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_date(text):
    match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", text)
    return match.group(0) if match else None

def extract_currency(text):
    match = re.search(r"(\$|\u20B9|EUR|INR|USD)", text)
    return match.group(1) if match else None

def extract_total(text):
    match = re.search(r"total[:\s]*([\$\u20B9]?)\s?(\d+[.,]?\d{2})", text, re.IGNORECASE)
    return match.group(1) + match.group(2) if match else None

def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+\.[\w]+", text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}", text)
    return match.group(0) if match else None

def preprocess_image_for_ocr(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    denoised = cv2.medianBlur(thresh, 3)
    return denoised

def clean_ocr_text(text):
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = ''.join(c for c in text if c.isprintable())
    return text.strip()

def process_document(extracted_data):
    invoice = {
        "invoice_id": None,
        "invoice_number": None,
        "issue_date": None,
        "due_date": None,
        "currency": None,
        "status": None,
        "issuer": {
            "company_name": None,
            "tax_id": None,
            "address": {
                "street": None, "city": None, "state": None,
                "postal_code": None, "country": None
            },
            "contact": {"name": None, "email": None, "phone": None}
        },
        "recipient": {
            "company_name": None,
            "tax_id": None,
            "address": {
                "street": None, "city": None, "state": None,
                "postal_code": None, "country": None
            },
            "contact": {"name": None, "email": None, "phone": None}
        },
        "line_items": [],
        "totals": {
            "subtotal": None, "tax_total": None,
            "total": None, "amount_due": None
        },
        "payment_terms": None,
        "notes": None,
        "metadata": {
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "source_system": "Docling-Tesseract-OCR"
        }
    }

    # Entity candidates for better mapping
    issuer_orgs, recipient_persons, issuer_cities, issuer_streets = [], [], [], []

    for box in extracted_data:
        text = box.get("text", "").strip()
        text = clean_ocr_text(text)
        # Regex extraction
        if not invoice["invoice_number"]:
            invoice_number = extract_invoice_number(text)
            if invoice_number:
                invoice["invoice_number"] = invoice_number
        date = extract_date(text)
        if date:
            if not invoice["issue_date"]:
                invoice["issue_date"] = date
            elif not invoice["due_date"] and date != invoice["issue_date"]:
                invoice["due_date"] = date
        if not invoice["currency"]:
            currency = extract_currency(text)
            if currency:
                invoice["currency"] = currency
        if not invoice["totals"]["total"]:
            total = extract_total(text)
            if total:
                invoice["totals"]["total"] = total
                invoice["totals"]["amount_due"] = total
        if not invoice["issuer"]["contact"]["email"]:
            email = extract_email(text)
            if email:
                invoice["issuer"]["contact"]["email"] = email
        if not invoice["issuer"]["contact"]["phone"]:
            phone = extract_phone(text)
            if phone:
                invoice["issuer"]["contact"]["phone"] = phone
        # NER extraction
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG" and ent.text not in issuer_orgs:
                issuer_orgs.append(ent.text)
            elif ent.label_ == "PERSON" and ent.text not in recipient_persons:
                recipient_persons.append(ent.text)
            elif ent.label_ == "GPE" and ent.text not in issuer_cities:
                issuer_cities.append(ent.text)
            elif ent.label_ == "FAC" and ent.text not in issuer_streets:
                issuer_streets.append(ent.text)
    if issuer_orgs:
        invoice["issuer"]["company_name"] = issuer_orgs[0]
        if len(issuer_orgs) > 1:
            invoice["recipient"]["company_name"] = issuer_orgs[1]
    if recipient_persons:
        invoice["recipient"]["contact"]["name"] = recipient_persons[0]
    if issuer_cities:
        invoice["issuer"]["address"]["city"] = issuer_cities[0]
    if issuer_streets:
        invoice["issuer"]["address"]["street"] = issuer_streets[0]

    return {
        "invoice": invoice,
        "extracted_blocks": extracted_data  # ✅ Added to return bbox + text too
    }

# ========================== MAIN PROCESS ==========================
def parse_layout_and_ocr(image_path: str) -> dict:
    image_pil = Image.open(image_path).convert("RGB")
    image_cv2 = cv2.imread(image_path)

    # Detect layout
    inputs = processor(images=image_pil, return_tensors="pt", size={"height": 640, "width": 640})
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([image_pil.size[::-1]]),
        threshold=0.3
    )

    extracted_info = []
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            label = model.config.id2label[label_id.item() + 1]
            box = [int(coord) for coord in box.tolist()]
            x0, y0, x1, y1 = box
            cropped = image_cv2[y0:y1, x0:x1]
            if cropped.size == 0:
                continue
            preprocessed_img = preprocess_image_for_ocr(cropped)
            ocr_text = ""
            for psm in [6, 11, 12]:
                config = f'--oem 3 --psm {psm}'
                text = pytesseract.image_to_string(preprocessed_img, config=config)
                text = clean_ocr_text(text)
                if len(text) > len(ocr_text):
                    ocr_text = text
            if ocr_text:
                extracted_info.append({
                    "label": label,
                    "bbox": box,
                    "text": ocr_text
                })
    return process_document(extracted_info)