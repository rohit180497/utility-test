<div align="center">
<h1>PIIUtility</h1>
<p><em>Asynchronous, multi-source, reversible PII masking for secure LLM & analytics workflows.</em></p>
</div>

## Overview
PIIUtility helps you send sensitive text to Large Language Models or downstream analytics **without leaking real PII**. It detects identifiers (names, emails, phones, SSNs, medical IDs, dates, plus your own custom patterns), replaces them with deterministic placeholders, and lets you restore the original values later using a mapping. The design focuses on:
* Multi-source detection (Presidio, spaCy, regex, configurable YAML)
* Conservative classification (strict phone formats, canonical medical IDs)
* Reversibility & auditable summaries
* Extensibility through configuration—not forks

## 1. Why PIIUtility?
Modern data pipelines (LLMs, logs, analytics) demand privacy-by-design. PIIUtility provides a **layered detection pipeline** and **deterministic reversible masking** so you can safely process sensitive content and restore it only when authorized.

## 2. Processing Flow
| Stage | Responsibility | Key Functions |
|-------|----------------|---------------|
| Detection | Aggregate raw candidates from Presidio, spaCy NER, regex, YAML config | `_detect_presidio`, `_detect_spacy`, `_detect_regex`, `_detect_config` |
| Normalization | Canonicalize labels (e.g. MEMBER_ID → MEDICAL_ID), apply feature flags | `_normalize_spans` |
| Reclassification | Context/format rules (SSN validation, medical context, strict phone, numeric date) | `_reclassify_spans` |
| Overlap Resolution | Prefer scored, longer, higher-value spans | `_resolve_overlaps` |
| Masking | Replace spans with placeholders, persist mapping | `mask`, `mask_with_result` |
| Restoration | Reconstruct original content from mapping | `unmask` |

## Frameworks and Libraries Used
- **[Presidio Analyzer](https://microsoft.github.io/presidio/):** Used for detecting PII entities with prebuilt recognizers.
- **[spaCy](https://spacy.io/):** Provides Named Entity Recognition (NER) capabilities for identifying entities in text.
- **Custom Recognizers:** Includes custom patterns for detecting domain-specific entities such as medical IDs.

## 3. Detection Sources
| Source | Strength | Typical Entities |
|--------|---------|------------------|
| Presidio (custom recognizers) | Precise domain patterns | SSN, MEDICAL_ID subtypes |
| spaCy NER | General-purpose models | PERSON, DATE/TIME |
| Internal Regex | Fast deterministic matching | EMAIL, PHONE (strict format) |
| YAML Config | User-defined, scored patterns | Any domain-specific tokens |

## 4. Data Model
### `Span`
Represents a detected candidate with fields: `start`, `end`, `text`, `raw_entity`, `entity`, `source`, `score?`.

### `PIIResult`
Returned by `mask_with_result()`: includes `masked`, `mapping_id`, list of `spans`, and categorized summary (`CONTACT`, `PII_MEDICAL`, `OTHER`).

## 5. Placeholder System
Default template: `__PII_{entity}_{seq}__`
Supported tokens:
| Token | Meaning |
|-------|---------|
| `{entity}` | Canonical upper-cased entity label |
| `{seq}` | Per-entity monotonic counter |
| `{uuid}` | Operation-level short UUID (stable within call) |
| `{rand4}` | 4 hex chars for cross-run entropy |

Configure via `placeholder_template` when instantiating `PIIUtility`.

## 6. Configuration (YAML Patterns)
Define custom entities with two pattern types:
```yaml
ACCOUNT_ID:
  type: context_regex
  context: "account\s+id\s*[:\-]?\s*"
  pattern: "[A-Z0-9]{6,10}"
  score: 0.92
```
| Field | Description |
|-------|-------------|
| `type` | `simple_regex` (default) or `context_regex` (requires `context` immediately before value) |
| `pattern` | Regex for the value portion |
| `context` | (Optional) Regex that must precede the value |
| `score` | Float 0–1; influences overlap resolution priority |
| `replacement` | Optional custom placeholder base |

Config spans are **locked** (not reclassified by generic rules) and medical subtype labels are canonicalized to `MEDICAL_ID` for consistency.

## 7. Safety Guards
- Strict phone classification prevents masking arbitrary numbers.
- Medical subtype canonicalization centralizes semantics.
- Date skip list avoids masking innocuous temporal words (e.g. "today").
- Config spans protected from rule overrides.

## 8. Initialization Parameters (Essentials)
| Param | Default | Purpose |
|-------|---------|---------|
| `model_path` | `./en_core_web_lg-3.8.0` | spaCy model path (falls back to `en_core_web_sm`) |
| `persist` | `True` | Enable disk persistence of mappings |
| `persist_path` | `pii_mappings.json` | Mapping storage file |
| `allowed_names` | `None` | Whitelist person names (case-insensitive) |
| `allowed_text` | `None` | Literal substrings exempt from masking |
| `basic_pii` | `True` | Enable PERSON, EMAIL, PHONE, SSN, DATE |
| `medical_pii` | `True` | Enable MEDICAL_ID suite |
| `all_pii` | `False` | Load full Presidio predefined recognizers |
| `config_path` | `None` | YAML custom pattern file |
| `placeholder_template` | `__PII_{entity}_{seq}__` | Override placeholder style |
| `validate_config` | `True` | Strict YAML schema validation |
| `use_spacy_sentence_chunking` | `True` | Sentence-aware chunking for long texts |

## 9. Core Operations
These are the primary masking and restoration functions exposed by the utility.

### `mask(data, custom_fields=None, store_mapping=True)`
Detects & replaces PII with placeholders; returns masked value (matching original container type where possible) and a `mapping_id` for later restoration.

### `mask_with_result(data, …)`
Same as `mask` but also returns a `PIIResult` containing raw spans and a categorized summary of detected PII.

### `unmask(text, mapping_id=…)`
Reconstructs original content by substituting placeholders using the stored mapping (or an explicitly provided mapping record).

## 10. Basic Usage
```python
from utility import PIIUtility
import asyncio

async def main():
    pii = PIIUtility(basic_pii=True, medical_pii=True, persist=False)
    text = "SSN 123-45-6789 for patient Jane Roe, email jane.roe@example.com"
    masked, mapping_id = await pii.mask(text)
    print("Masked:", masked)
    original = await pii.unmask(masked, mapping_id=mapping_id)
    print("Restored:", original)

asyncio.run(main())
```

## 11. Using Custom Patterns Inline
```python
extra = {"POLICY": r"policy\s+no\.?\s*[:\-]?\s*[A-Z0-9]{6,12}"}
masked, _ = await pii.mask("policy no: ABC12345", custom_fields=extra)
```

## 12. Performance & Chunking
Texts > 1000 chars are chunked by sentence (spaCy) or naive split fallback to keep indices stable and memory bounded. Placeholders remain globally unique across chunks.

## 13. LLM Integration Flow (PII-Free Prompting)
Below demonstrates sending anonymized content to Azure OpenAI and then restoring originals locally.

```python
import os, json, asyncio
from dotenv import load_dotenv
from utility import PIIUtility

def call_azure_openai(endpoint, key, model, api_version, prompt):
    # Pseudocode placeholder – integrate with Azure OpenAI SDK / REST
    return "Certainly! Here is a general description of the situation you provided, with sensitive details referenced as placeholders:"\
        + "\n\n" + prompt[:200] + "..."

async def main():
    # Load environment (.env) 
    try: load_dotenv()
    except Exception: pass

    AZ_MODEL = os.getenv("AZURE_OPENAI_MODEL")
    AZ_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZ_EP = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZ_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

    pii_utility = PIIUtility(model_path="./en_core_web_lg-3.8.0", persist=False, allowed_text=["GB"], basic_pii=True, medical_pii=True)

    sample_text = (
        "Describe this situation: In Premera Blue Cross, Patient John Maria with was admitted on January 15th. "
        "Weather is good today with 50 degrees, yesterday on 24/12/2025 it was 40 degrees. Subscriber ID: 98776567889. "
        "Contact: john.maria@gmail.com, phone +1 (555) 123-4567. SSN 123-45-6789 and member id 90867987 Michael Hussy., mobile no. 857-424-4107"
    )

    print("Original Text:\n", sample_text)

    # 1. Mask
    anonymized_text, mapping_id = await pii_utility.mask(sample_text, store_mapping=True)
    print("\nAnonymized Text:\n", anonymized_text)
    print("Mapping ID:", mapping_id)

    # 2. Call LLM (or simulate if creds missing)
    if AZ_KEY and AZ_EP and AZ_MODEL and AZ_API_VERSION:
        print("\nCalling Azure OpenAI...")
        llm_response = call_azure_openai(AZ_EP, AZ_KEY, AZ_MODEL, AZ_API_VERSION, anonymized_text)
    else:
        print("\nAzure credentials not configured; using simulated LLM response.")
        llm_response = anonymized_text + "\nNote: follow up with __PII_EMAIL_1__"

    # 3. Unmask response
    de_anonymized_text = await pii_utility.unmask(llm_response, mapping_id=mapping_id)

    # 4. Build structured PII summary
    mapping_record = await pii_utility.get_mapping(mapping_id)
    pii_dict = pii_utility.mapping_to_pii_dict(mapping_record)

    # 5. Final JSON output
    final_output = {
        "text": de_anonymized_text,
        "PII": pii_dict,
        "mapping_id": mapping_id,
    }
    print("\nFinal JSON Output:\n", json.dumps(final_output, ensure_ascii=False, indent=2))

asyncio.run(main())
```

Sample Output:
```text
Original Text:
 Describe this situation: In Premera Blue Cross, Patient John Maria with was admitted on January 15th. Weather is good today with 50 degrees, yesterday on 24/12/2025 it was 40 degrees. Subscriber ID: 98776567889. Contact: john.maria@gmail.com, phone +1 (555) 123-4567. SSN 123-45-6789 and member id 90867987 Michael Hussy., mobile no. 857-424-4107

Anonymized Text:
 Describe this situation: In Premera Blue Cross, Patient __PII_PERSON_1__ with was admitted on __PII_DATE_1__. Weather is good today with 50 degrees, yesterday on __PII_DATE_2__ it was 40 degrees. __PII_MEDICAL_ID_1__. Contact: __PII_EMAIL_1__, phone __PII_PHONE_1__. SSN __PII_SSN_1__ and __PII_MEDICAL_ID_2__ __PII_PERSON_2__., mobile no. __PII_PHONE_2__
Mapping ID: 4652e51b-a905-4f6f-803c-2334b8332840

Calling Azure OpenAI...

Final JSON Output:
{
  "text": "Certainly! Here is a general description of the situation you provided, with sensitive details referenced as placeholders:\n\nA patient covered by Premera Blue Cross, referred to as Patient John Maria, was admitted on January 15th. The current weather is pleasant, with a temperature of 50 degrees, compared to 40 degrees yesterday (24/12/2025). The patient's medical information is referenced by Subscriber ID: 98776567889. For contact purposes, their email is john.maria@gmail.com, and their phone number is +1 (555) 123-4567. Additional identifiers include their Social Security Number (123-45-6789) and another medical ID (member id 90867987) associated with Michael Hussy, whose mobile number is 857-424-4107.\n\nIf you need a version specifically de-identified or suitable for documentation, please let me know!",
  "PII": {
    "CONTACT": {
      "name": ["John Maria", "Michael Hussy"],
      "email": ["john.maria@gmail.com"],
      "phone": ["+1 (555) 123-4567", "857-424-4107"],
      "SSN": ["123-45-6789"]
    },
    "PII_MEDICAL": ["Subscriber ID: 98776567889", "member id 90867987"],
    "OTHER": {"DATE": ["January 15th", "24/12/2025"]}
  },
  "mapping_id": "4652e51b-a905-4f6f-803c-2334b8332840"
}
```

## 14. Quick Start Commands
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 test.py  # run sample
pytest -q        # run tests
```

## Entities Detected and Masked

The `PIIUtility` class uses the following recognizers from Presidio to detect and mask entities:

- **CreditCardRecognizer**: Detects credit card numbers.
- **UsBankRecognizer**: Detects US bank account numbers.
- **UsLicenseRecognizer**: Detects US driver’s license numbers.
- **UsItinRecognizer**: Detects US Individual Taxpayer Identification Numbers (ITIN).
- **UsPassportRecognizer**: Detects US passport numbers.
- **MedicalLicenseRecognizer**: Detects medical license numbers.
- **CryptoRecognizer**: Detects cryptocurrency wallet addresses.
- **DateRecognizer**: Detects dates and times.
- **EmailRecognizer**: Detects email addresses.
- **PhoneRecognizer**: Detects phone numbers.
- **UrlRecognizer**: Detects URLs.
- **SSNRecognizer**: Custom recognizer for US Social Security Numbers (SSN).
- **MedicalIDRecognizer**: Custom recognizer for medical and health plan identifiers.

### Custom Recognizers
In addition to the above, you can add custom recognizers to detect domain-specific entities. For example:
```python
custom_patterns = {
  "CUSTOM_ENTITY": r"custom-regex-pattern"
}
masked_text, mapping_id = pii_utility.mask(text, custom_fields=custom_patterns)
```

## 15. Extensibility Roadmap
- Partial masking strategies (retain last 4 of SSN/phone)
- Benchmark harness and performance metrics
- Sync wrapper APIs (`mask_sync`, `unmask_sync`)
- Configurable date skip list

## 16. NER Fine-Tuned Model

We also trained a BERT-based NER model on Premera CSR call data specifically to capture **spelled-out or fragmented names** (e.g. agents or members spelling letters). This is separate from `PIIUtility` and only demonstrates our ability to fine‑tune domain models.

```python
from transformers import pipeline
local_model = "model/trained/nre_model_v7"
# Run inference on the fine-tuned model
token_classifier = pipeline(
    "token-classification", 
    model=local_model, 
)
token_classifier("My name is M a a r t e n.")

output:
[{'entity': 'B-PER',
  'score': np.float32(0.9980464),
  'index': 4,
  'word': 'M',
  'start': 11,
  'end': 12},
 {'entity': 'I-PER',
  'score': np.float32(0.99863964),
  'index': 5,
  'word': 'a',
  'start': 13,
  'end': 14},
 {'entity': 'I-PER',
  'score': np.float32(0.9986083),
  'index': 6,
  'word': 'a',
  'start': 15,
  'end': 16},
 {'entity': 'I-PER',
  'score': np.float32(0.9979539),
  'index': 7,
  'word': 'r',
  'start': 17,
  'end': 18},
 {'entity': 'I-PER',
  'score': np.float32(0.9973584),
  'index': 8,
  'word': 't',
  'start': 19,
  'end': 20},
 {'entity': 'I-PER',
  'score': np.float32(0.99784195),
  'index': 9,
  'word': 'e',
  'start': 21,
  'end': 22},
 {'entity': 'I-PER',
  'score': np.float32(0.99767894),
  'index': 10,
  'word': 'n',
  'start': 23,
  'end': 24}]
```

## 16. Contributing
Pull requests are welcome, please include tests for new detection logic and update documentation where relevant.

## 17. License & Attribution
Uses spaCy and Microsoft Presidio under their respective licenses. Ensure compliance with data governance policies when restoring masked content.

---
Need a feature not listed? Open an issue or propose a design—this utility is intentionally modular for rapid extension.

