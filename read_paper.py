from pypdf import PdfReader
import sys

try:
    reader = PdfReader(r"Research Paper Basis/2311.10554v1.pdf")
    text = ""
    print(f"Reading {len(reader.pages)} pages...")
    for i, page in enumerate(reader.pages):
        text += f"--- Page {i+1} ---\n"
        text += page.extract_text() + "\n"
        
    with open("paper_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Paper text extracted to paper_content.txt")

except Exception as e:
    print(f"Error: {e}")
