# Custom PII Configuration Guide

## Overview
The PII Utility supports custom pattern detection through YAML configuration files. This allows you to define your own entity types and detection patterns without modifying the code.

## Configuration Format

### Basic Structure
```yaml
ENTITY_NAME:
  type: context_regex | simple_regex
  context: "optional context pattern"
  pattern: "main regex pattern"
  replacement: "optional custom placeholder"
  score: 0.85  # optional confidence score (0.0-1.0)
```

### Fields Explanation

- **ENTITY_NAME** (required): Uppercase identifier for your custom entity
- **type** (optional, default: 'simple_regex'): Detection method
  - `simple_regex`: Direct pattern matching anywhere in text
  - `context_regex`: Requires preceding context pattern
- **pattern** (required): Regex pattern to match the entity value
- **context** (required for context_regex): Regex pattern that must appear immediately before the entity
- **replacement** (optional): Custom placeholder format (default: `__PII_ENTITY_NAME__`)
- **score** (optional, default: 0.85): Confidence score between 0.0 and 1.0

## Pattern Types

### 1. Simple Regex
Matches patterns anywhere in the text without requiring context.

**Example:**
```yaml
LICENSE_PLATE:
  type: simple_regex
  pattern: "\\b[A-Z]{2,3}[\\s\\-]?\\d{3,4}[A-Z]?\\b"
  replacement: "__PII_LICENSE_PLATE__"
  score: 0.75
```

**Matches:**
- ABC 1234
- AB-123X
- XYZ1234

### 2. Context Regex
Requires a specific keyword or phrase to appear before the value.

**Example:**
```yaml
SUBSCRIBER_ID:
  type: context_regex
  context: "subscriber\\s+id\\s*[:\\-]?\\s*"
  pattern: "\\b\\d{5,12}\\b"
  replacement: "__PII_SUBSCRIBER_ID__"
  score: 0.90
```

**Matches:**
- subscriber id: 12345678
- Subscriber ID 987654321
- subscriber id-123456789

**Does NOT match:**
- 12345678 (no context)
- customer id: 12345678 (wrong context)

## Usage Example

### 1. Create Configuration File
```yaml
# my_custom_patterns.yaml

POLICY_NUMBER:
  type: context_regex
  context: "policy\\s+(?:number|no\\.?|#)\\s*[:\\-]?\\s*"
  pattern: "[A-Z0-9]{6,15}"
  score: 0.90

EMPLOYEE_ID:
  type: context_regex
  context: "(?:employee|emp)\\s+(?:id|#)\\s*[:\\-]?\\s*"
  pattern: "[A-Z]{2}\\d{4,8}"
  score: 0.85
```

### 2. Initialize PIIUtility with Config
```python
from utility import PIIUtility
import asyncio

async def main():
    pii_utility = PIIUtility(
        model_path="./en_core_web_lg-3.8.0",
        persist=False,
        basic_pii=True,
        medical_pii=True,
        config_path="my_custom_patterns.yaml"  # Load custom patterns
    )
    
    text = "Policy number: ABC1234567. Employee ID: EMP-123456"
    masked, mapping_id = await pii_utility.mask(text)
    print(masked)
    # Output: Policy number: __PII_POLICY_NUMBER__. Employee ID: __PII_EMPLOYEE_ID__

asyncio.run(main())
```

## Regex Tips

### Escaping Special Characters
In YAML, backslashes must be escaped:
- Write `\\d` for digit class (not `\d`)
- Write `\\b` for word boundary (not `\b`)
- Write `\\s` for whitespace (not `\s`)

### Common Patterns

**Numbers:**
- `\\d{5,10}` - 5 to 10 digits
- `\\b\\d+\\b` - One or more digits as whole word

**Alphanumeric:**
- `[A-Z0-9]{6,12}` - 6-12 uppercase letters or digits
- `[A-Za-z0-9\\-]+` - Letters, digits, and hyphens

**Formatted IDs:**
- `[A-Z]{2}\\d{4}` - Two letters followed by 4 digits (e.g., AB1234)
- `\\d{3}-\\d{2}-\\d{4}` - SSN format (123-45-6789)

**Optional Separators:**
- `\\s*[:\\-]?\\s*` - Optional colon or dash with whitespace
- `[\\s\\-]?` - Optional space or dash

## Best Practices

1. **Be Specific with Context**: Use precise context patterns to avoid false positives
   ```yaml
   # Good - specific context
   context: "subscriber\\s+id\\s*[:\\-]?\\s*"
   
   # Bad - too generic
   context: "id\\s*"
   ```

2. **Test Your Patterns**: Validate regex patterns before deploying
   ```python
   import re
   pattern = re.compile(r"\d{5,10}")
   matches = pattern.findall("Test 12345 and 987654321")
   ```

3. **Use Appropriate Scores**: Higher scores for more specific patterns
   - 0.95-1.0: Highly specific patterns (e.g., SSN format)
   - 0.85-0.94: Context-based patterns
   - 0.70-0.84: General patterns

4. **Avoid Over-matching**: Be careful with broad patterns
   ```yaml
   # Bad - matches all numbers
   pattern: "\\d+"
   
   # Good - specific length range
   pattern: "\\b\\d{6,12}\\b"
   ```

## Troubleshooting

### Pattern Not Matching
1. Check regex escaping in YAML (use `\\` instead of `\`)
2. Verify case sensitivity (patterns are case-insensitive by default)
3. Test the pattern separately with Python's `re` module

### False Positives
1. Add more specific context requirements
2. Use word boundaries (`\\b`)
3. Increase pattern specificity

### Config Not Loading
1. Check YAML syntax (indentation, colons)
2. Verify file path is correct
3. Check logs for error messages

## Examples Library

See `pii_config.yaml` for a comprehensive collection of pre-configured patterns including:
- Subscriber IDs
- Member IDs
- Policy Numbers
- Account Numbers
- Employee IDs
- Rx Groups
- Claim Numbers
- License Plates
- Custom IDs
