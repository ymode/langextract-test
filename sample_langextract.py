


import langextract
import PyPDF2
import json
import glob
import traceback
import enum as _enum

from langextract.data import Extraction, ExampleData
from langextract.factory import ModelConfig

# Example for product summary extraction
example = ExampleData(
    text="SuperWidget 3000 is a high-performance widget for industrial automation, increasing efficiency and reliability.",
    extractions=[
        Extraction(
            extraction_class="product_summary",
            extraction_text="SuperWidget 3000",
            attributes={
                "product_name": "SuperWidget 3000",
                "purpose": "High-performance widget for industrial automation, increasing efficiency and reliability.",
                "key_features": [
                    "Real-time monitoring",
                    "Automated error correction",
                    "Seamless integration with existing systems"
                ]
            }
        )
    ]
)

# Model configuration for OpenAI GPT-4o
config = ModelConfig(model_id="gpt-4o", provider="openai")

# List of PDF files to process
# Discover PDF files in the current directory
pdf_files = glob.glob("*.pdf")

# Recursively convert objects to dicts for JSON serialization
def to_serializable(obj):
    """Recursively convert objects to JSON-serializable structures.

    Handles primitives, lists, dicts, Enums, objects with __dict__, _asdict (namedtuple/dataclass),
    and objects with __slots__. Falls back to str(obj) for unknown types.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, list):
        return [to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, _enum.Enum):
        # return the name for readability
        return obj.name

    # Objects that expose attributes
    # 1) normal objects with __dict__
    if hasattr(obj, "__dict__"):
        d = dict(vars(obj))
        return {k: to_serializable(v) for k, v in d.items()}

    # 2) namedtuple or dataclass-like with _asdict()
    if hasattr(obj, "_asdict") and callable(getattr(obj, "_asdict")):
        d = obj._asdict()
        return {k: to_serializable(v) for k, v in d.items()}

    # 3) objects with __slots__
    if hasattr(obj, "__slots__"):
        d = {slot: getattr(obj, slot) for slot in getattr(obj, "__slots__", [])}
        return {k: to_serializable(v) for k, v in d.items()}

    # 4) fallback to string representation
    return str(obj)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def main():
    json_results = []
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            try:
                result = langextract.extract(
                    text,
                    prompt_description="Extract the product name, purpose, and key features from this document.",
                    examples=[example],
                    config=config,
                )
                print(f"{pdf_path}:\n{result}\n")
                json_results.append({
                    "file": pdf_path,
                    "extraction": {
                        "extractions": [to_serializable(e) for e in getattr(result, "extractions", [])],
                        "text": getattr(result, "text", text)
                    }
                })
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Error processing {pdf_path}: {e}\n{tb}")
                json_results.append({
                    "file": pdf_path,
                    "extraction": None,
                    "error": str(e),
                    "traceback": tb
                })
        else:
            print(f"{pdf_path}: No text extracted.")
            json_results.append({"file": pdf_path, "extraction": None, "error": "No text extracted."})

    # Write results to a JSON file
    out_path = "langextract_output.json"
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
