import pypdf
import sys

def extract_text(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading {pdf_path}: {e}"

if __name__ == "__main__":
    paths = sys.argv[1:]
    for path in paths:
        print(f"--- START OF FILE: {path} ---")
        print(extract_text(path))
        print(f"--- END OF FILE: {path} ---\n")
