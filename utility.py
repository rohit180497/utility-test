"""PII Utility Module

This module provides an asynchronous, multi-stage pipeline for detecting and masking
Personally Identifiable Information (PII) across free text, JSON structures, and other
serializable Python objects. It integrates multiple detection sources (Presidio, spaCy,
regex, and user-provided YAML configuration) and applies ordered normalization,
contextual reclassification, overlap resolution, and deterministic placeholder masking.

Design Goals:
    * Separation of concerns: discrete stages for detection, normalization, reclassification.
    * Extensibility: YAML-driven custom patterns with optional contextual preconditions.
    * Safety: conservative phone classification & medical ID canonicalization to prevent
      over-masking of benign numeric values.
    * Reversibility: Mapping persistence enabling restoration of original text.
    * Performance: Sentence-aware chunking to limit memory while maintaining natural boundaries.

"""

import re
import os
import json
import uuid
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import spacy
from presidio_analyzer import PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer import AnalyzerEngine
import aiofiles
import asyncio
import yaml


# Configure logging
logger = logging.getLogger(__name__)
# Uncomment the following lines to enable detailed logging for debugging purposes.
# logger.setLevel(logging.INFO)
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MedicalIDRecognizer(PatternRecognizer):
    """Custom Presidio recognizer for structured medical & healthcare identifiers.

    The patterns intentionally:
        * Allow flexible whitespace & separators (space/underscore) to support JSON key mirroring.
        * Restrict identifier body to 4–20 word/number characters to avoid greedy over-capture.
        * Provide individual Pattern objects so Presidio scoring remains granular.

    Supported conceptual identifiers include health plan, member/subscriber IDs, MRN, patient IDs,
    admission/encounter/visit numbers, hospital account numbers, episode numbers, BIN, RX group,
    and group IDs. All are unified downstream into the canonical entity label MEDICAL_ID.
    """

    def __init__(self):
        patterns = [
            Pattern(
                name="health_plan_1", 
                regex=r"(?i)(?:health\s+plan|plan)\s+(?:number|no\.?|num\.?|id|#)\s*[:\-]?\s*([A-Z0-9][\w\-]{3,19})\b", 
                score=0.95
            ),
            
            Pattern(
                name="member_id_1", 
                regex=r"(?i)member[\s_]+(?:id|identification|number|no\.?|num\.?|#|account)\s*(?:number|no\.?|num\.?|#)?\s*[:\-]?\s*([A-Z0-9][\w\-]{3,19})\b", 
                score=0.85
            ),
            
            Pattern(
                name="subscriber_id_1", 
                regex=r"(?i)subscriber[\s_]+(?:id|identification|number|no\.?|num\.?|#|account)\s*(?:number|no\.?|num\.?|#)?\s*[:\-]?\s*([A-Z0-9][\w\-]{3,19})\b", 
                score=0.85
            ),
            
            Pattern(
                name="mrn_1", 
                regex=r"(?i)(?:medical[\s_]+record|mrn)(?:[\s_]+(?:number|no\.?|num\.?|#))?\s*[:\-]?\s*([A-Z0-9][\w\-]{3,19})\b", 
                score=0.90
            ),
            
            Pattern(
                name="patient_id_1", 
                regex=r"(?i)patient[\s_]+(?:id|identification|number|no\.?|num\.?|#)\s*[:\-]?\s*([A-Z0-9][\w\-]{3,19})\b", 
                score=0.90
            ),
            
            Pattern(
                name="admission_1", 
                regex=r"(?i)admission\s+(?:number|no\.?|num\.?|id|#)\s*[:\-]?\s*([A-Z0-9][\w\-]{3,19})\b", 
                score=0.85
            ),
            
            Pattern(
                name="encounter_1", 
                regex=r"(?i)encounter\s+(?:number|no\.?|num\.?|#)\s*[:\-]?\s*([A-Z0-9][\w\-]{3,19})\b", 
                score=0.85
            ),
            
            Pattern(
                name="visit_1", 
                regex=r"(?i)visit\s+(?:number|no\.?|num\.?|#)\s*[:\-]?\s*([A-Z0-9][\w\-]{3,19})\b", 
                score=0.85
            ),
            
            Pattern(
                name="hospital_1", 
                regex=r"(?i)hospital\s+(?:account|record)\s*(?:number|no\.?|num\.?|#)?\s*[:\-]?\s*([A-Z0-9][\w\-]{3,19})\b", 
                score=0.85
            ),
            
            Pattern(
                name="episode_1", 
                regex=r"(?i)episode\s+(?:number|no\.?|num\.?|#)\s*[:\-]?\s*([A-Z0-9][\w\-]{3,19})\b", 
                score=0.85
            ),
            
            Pattern(
                name="bin_number", 
                regex=r"(?i)\bbin\s*(?:number|no\.?|num\.?|#)?\s*[:\-]?\s*(\d{6})\b", 
                score=0.90
            ),
            
            Pattern(
                name="rx_group_number", 
                regex=r"(?i)\b(?:rx|prescription)[\s_]+group\s*(?:number|no\.?|num\.?|#)?\s*[:\-]?\s*([A-Z0-9][\w\-]{3,9})\b", 
                score=0.90
            ),
            
            Pattern(
                name="group_id", 
                regex=r"(?i)\b(?:group|grp)(?:[\s_]+id)?\s*(?:number|no\.?|num\.?|#)?\s*[:\-]?\s*([A-Z0-9][\w\-]{3,9})\b", 
                score=0.85
            ),
        ]
        super().__init__(supported_entity="MEDICAL_ID", patterns=patterns)


class SSNRecognizer(PatternRecognizer):
    """Custom Presidio recognizer for US Social Security Numbers with rule validation.

    Validation constraints encoded in the regex follow SSA standards:
        * Area number (first 3 digits): exclude 000, 666, 900–999.
        * Group number (middle 2 digits): exclude 00.
        * Serial number (last 4 digits): exclude 0000.

    Only the canonical hyphenated format XXX-XX-XXXX is accepted, reducing false positives
    compared to naive 9-digit matching.
    """

    def __init__(self):
        patterns = [
            Pattern(
                name="ssn",
                regex=r"\b(?!000|666|9\d{2})([0-8]\d{2})-(?!00)(\d{2})-(?!0000)(\d{4})\b",
                score=0.99
            ),
        ]
        super().__init__(supported_entity="SSN", patterns=patterns)


# Default regex patterns for common PII types
DEFAULT_FIELD_PATTERNS: Dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"),
    "PHONE": re.compile(r"(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b"),
}


@dataclass
class Span:
    """Unified span representation through the detection pipeline."""
    start: int
    end: int
    text: str
    raw_entity: str  # Original detector label
    entity: str      # Mutable normalized/reclassified label
    source: str      # presidio | spacy | regex | config | final
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            **({"score": self.score} if self.score is not None else {})
        }

@dataclass
class PIIResult:
    """Wrapper for masking operation providing structured outputs."""
    masked: Any
    mapping_id: Optional[str]
    spans: List[Span]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "masked": self.masked,
            "mapping_id": self.mapping_id,
            "spans": [s.to_dict() for s in self.spans],
            "summary": self.summary,
        }


# Compiled regex & constants for reclassification and filtering
SSN_PATTERN = re.compile(r"^\d{3}-\d{2}-\d{4}$")
# Original broad phone regex retained for initial regex candidate detection (DEFAULT_FIELD_PATTERNS)
PHONE_PATTERN = re.compile(r"^[\+\(]?\d[\d\-\s\(\)]+\d$")
# Stricter phone pattern for classification to avoid over-masking numeric identifiers
STRICT_PHONE_PATTERN = re.compile(r"^(?:\+?\d{1,2}[\s\-]?)?(?:\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4})$")
DATE_PATTERN = re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$")  # Numeric date-like formats only

# Canonical medical keyword phrases (single source of truth)
MEDICAL_KEYWORDS = [
    "health plan", "plan number", "member id", "subscriber id", "patient id", "mrn",
    "medical record", "rx group", "bin", "group id", "admission", "encounter", "visit",
    "hospital", "episode"
]

def build_medical_context_pattern(keywords: List[str]) -> re.Pattern:
    # Build a permissive context pattern: any keyword optionally followed by identifier markers
    escaped = []
    for kw in keywords:
        kw_re = re.escape(kw).replace("\\ ", "\\s+")  # allow flexible whitespace
        escaped.append(kw_re + r"(?:\s+(?:number|no\.?|id|#))?")
    joined = "|".join(escaped)
    return re.compile(rf"(?:{joined})\s*[:\-]?\s*$", re.IGNORECASE)

MEDICAL_CONTEXT_PATTERN = build_medical_context_pattern(MEDICAL_KEYWORDS)

# Config-sourced medical subtype labels to canonicalize
MEDICAL_CANONICAL_LABELS = {
    "MEMBER_ID", "SUBSCRIBER_ID", "PATIENT_ID", "MRN", "HEALTH_PLAN", "GROUP_ID",
    "ADMISSION", "ENCOUNTER", "VISIT", "HOSPITAL", "EPISODE", "RX_GROUP", "BIN", "BIN_NUMBER"
}

DATE_SKIP_WORDS = {"today", "day", "tomorrow", "year", "yesterday"}  # trimmed for clarity


class PIIUtility:
    """High-level utility for detection, masking and restoration of PII.

    Key Features:
        * Multi-detector pipeline (Presidio + spaCy + regex + YAML config).
        * Canonical Span model enabling ordered transformations.
        * Deterministic placeholder generation with configurable template.
        * Optional persistence (file or in-memory) for reversible masking.
        * Medical-specific identifier consolidation & filtering safeguards.

    Typical Workflow:
        1. Instantiate with desired feature flags (basic_pii, medical_pii, etc.).
        2. Call `mask()` or `mask_with_result()` on input data.
        3. Persist or inspect placeholder mapping via returned mapping_id.
        4. Restore original content using `unmask()`.

    Args:
        model_path (str | None): Path to spaCy model. Falls back to small English model if invalid.
        analyzer (AnalyzerEngine | None): Pre-initialized Presidio analyzer; lazily constructed if omitted.
        persist_path (str): JSON file path for placeholder→original mapping storage.
        persist (bool): Whether to persist mappings to disk (otherwise kept in-memory).
        allowed_names (List[str] | None): Person-name whitelist to avoid masking (case-insensitive).
        allowed_text (List[str] | None): Literal text snippets to exempt from masking.
        basic_pii (bool): Enable core entities (PERSON, EMAIL, PHONE, SSN, DATE).
        medical_pii (bool): Enable medical identifier entities.
        all_pii (bool): Enable Presidio's full predefined recognizer set (overrides selective flags).
        config_path (str | None): YAML file path supplying custom regex/context patterns.
        placeholder_template (str): Format string supporting tokens {entity}, {seq}, {uuid}, {rand4}.
        validate_config (bool): Enforce strict validation rules on YAML schema & pattern integrity.
        use_spacy_sentence_chunking (bool): Attempt sentence-aware chunking for large texts.

    Raises:
        ValueError: If config validation is enabled and YAML schema violations are encountered.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        analyzer: Optional[AnalyzerEngine] = None,
        persist_path: str = "pii_mappings.json",
        persist: bool = True,
        allowed_names: Optional[List[str]] = None,
        allowed_text: Optional[List[str]] = None,
        basic_pii: bool = True,
        medical_pii: bool = True,
        all_pii: bool = False,
        config_path: Optional[str] = None,
        placeholder_template: str = "__PII_{entity}_{seq}__",
        validate_config: bool = True,
        use_spacy_sentence_chunking: bool = True,
    ) -> None:
        """
        Initialize the PII detection and masking utility.

        Parameters:
        - model_path: Path to spaCy model for NER. Defaults to environment variable or './en_core_web_lg-3.8.0'.
        - analyzer: Pre-configured Presidio AnalyzerEngine. Creates new instance if None.
        - persist_path: JSON file path for storing mask-to-original mappings.
        - persist: Enable/disable mapping persistence to file.
        - allowed_names: Names to exclude from masking (e.g., brand names, public figures).
        - allowed_text: Text patterns to exclude from masking (whitelist).
        - basic_pii: Enable contact PII detection (PERSON, EMAIL, PHONE, SSN).
        - medical_pii: Enable medical ID detection (member ID, subscriber ID, etc.).
        - all_pii: Enable all Presidio built-in recognizers (comprehensive mode).
        - config_path: YAML file path for custom pattern definitions with context support.
        """
        self.model_path = model_path or os.environ.get("PII_SPACY_MODEL", "./en_core_web_lg-3.8.0")
        self._nlp = None  # Lazy loading
        self.all_pii = all_pii
        self._analyzer = analyzer  # Store provided analyzer or None for lazy loading
        self.persist = persist
        self.persist_path = persist_path
        self._lock = asyncio.Lock()  # Using asyncio.Lock for async context
        self._in_memory_store = {}
        self.allowed_names = set((allowed_names or []))
        self.allowed_text = set((allowed_text or []))
        self.basic_pii = basic_pii
        self.medical_pii = medical_pii
        self.placeholder_template = placeholder_template
        self.validate_config = validate_config
        self.use_spacy_sentence_chunking = use_spacy_sentence_chunking
        # Lower-cased allowed names for case-insensitive comparison
        self.allowed_names_lower = {n.lower() for n in self.allowed_names}

        # Load custom patterns from YAML config if provided
        self.custom_patterns = self._load_config(config_path) if config_path else {}

        # Try to create mapping file if it doesn't exist
        if self.persist and not os.path.exists(self.persist_path):
            with open(self.persist_path, "w", encoding="utf-8") as fh:
                json.dump({}, fh)

    @property
    def analyzer(self) -> AnalyzerEngine:
        """Return (and lazily initialize) the Presidio AnalyzerEngine.

        The analyzer registry is rebuilt with only the requested recognizers to avoid
        unexpected overrides from Presidio defaults.

        Returns:
            AnalyzerEngine: Configured Presidio analyzer instance.
        """
        if self._analyzer is None:
            self._analyzer = self._initialize_analyzer()
        return self._analyzer
    
    def _initialize_analyzer(self) -> AnalyzerEngine:
        """Configure Presidio analyzer with selective/custom recognizers.

        Implementation Notes:
            * Default recognizer list is cleared to maintain deterministic behavior.
            * Full predefined recognizers are loaded only if `all_pii` is True.
            * Custom SSN & Medical ID recognizers always replace their built-in counterparts.

        Returns:
            AnalyzerEngine: Ready-to-use analyzer with desired recognizers registered.
        """
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": self.model_path}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())

        try:
            analyzer.registry.recognizers = []
            
            if self.all_pii:
                from presidio_analyzer.predefined_recognizers import (
                    CreditCardRecognizer,
                    UsBankRecognizer,
                    UsLicenseRecognizer,
                    UsItinRecognizer,
                    UsPassportRecognizer,
                    MedicalLicenseRecognizer,
                    CryptoRecognizer,
                    DateRecognizer,
                    EmailRecognizer,
                    PhoneRecognizer,
                    UrlRecognizer,
                )

                recognizers = [
                    CreditCardRecognizer(),
                    UsBankRecognizer(),
                    UsLicenseRecognizer(),
                    UsItinRecognizer(),
                    UsPassportRecognizer(),
                    MedicalLicenseRecognizer(),
                    CryptoRecognizer(),
                    DateRecognizer(),
                    EmailRecognizer(),
                    PhoneRecognizer(),
                    UrlRecognizer(),
                    SSNRecognizer(),  # Using the custom SSN recognizer
                    MedicalIDRecognizer(),  # Add custom Medical ID recognizer
                ]

                for recognizer in recognizers:
                    if "en" in recognizer.supported_language:
                        analyzer.registry.add_recognizer(recognizer)
            else:
                # Only add custom recognizers
                analyzer.registry.add_recognizer(SSNRecognizer())
                analyzer.registry.add_recognizer(MedicalIDRecognizer())
        except Exception:
            pass

        return analyzer
    
    @property
    def nlp(self) -> spacy.language.Language:
        """Return (and lazily load) spaCy Language model.

        Attempts the user-specified path first; falls back to `en_core_web_sm` so that
        basic PERSON/DATE detection remains available even if large model assets are missing.

        Returns:
            spacy.language.Language: Loaded spaCy NLP pipeline.
        """
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model_path)
            except Exception:
                logger.info("Local spaCy model not found; falling back to en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def _load_config(self, config_path: str) -> Dict[str, Dict[str, Any]]:
        """Load and validate user-defined pattern configuration from YAML.

        YAML Schema (per top-level key = entity name):
            type (str): 'simple_regex' (default) or 'context_regex'.
            pattern (str): Primary regex capturing the value portion.
            context (str, optional): Preceding context regex required before value (only for context_regex).
            replacement (str, optional): Custom placeholder base.
            score (float, optional): Confidence weighting (0.0–1.0) used during overlap resolution.

        Validation Strategy:
            * Mandatory 'pattern'.
            * Type restricted to known set.
            * Score coerced or rejected if non-numeric (when validate_config=True).
            * Patterns compiled early to surface syntax errors.

        Args:
            config_path (str): Path to YAML file.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of ENTITY_NAME -> compiled pattern metadata.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if not config:
                logger.warning(f"Config file {config_path} is empty")
                return {}
            
            # Validate and compile regex patterns
            validated_patterns = {}
            for entity_name, pattern_config in config.items():
                try:
                    # Validation rules
                    if 'pattern' not in pattern_config:
                        msg = f"Pattern for {entity_name} missing 'pattern' field"
                        if self.validate_config:
                            raise ValueError(msg)
                        logger.warning(msg + ", skipping")
                        continue
                    ptype = pattern_config.get('type', 'simple_regex')
                    if ptype not in ('simple_regex', 'context_regex'):
                        msg = f"Invalid type '{ptype}' for {entity_name}"
                        if self.validate_config:
                            raise ValueError(msg)
                        logger.warning(msg + "; defaulting to simple_regex")
                        ptype = 'simple_regex'
                        pattern_config['type'] = ptype
                    score_val = pattern_config.get('score', 0.85)
                    if not isinstance(score_val, (int, float)):
                        msg = f"Score for {entity_name} must be numeric"
                        if self.validate_config:
                            raise ValueError(msg)
                        logger.warning(msg + "; using default 0.85")
                        score_val = 0.85
                    
                    # Compile regex to validate
                    pattern_regex = re.compile(pattern_config['pattern'], re.IGNORECASE)
                    
                    # Compile context regex if provided
                    context_regex = None
                    if ptype == 'context_regex' and pattern_config.get('context'):
                        context_regex = re.compile(pattern_config['context'], re.IGNORECASE)
                    
                    validated_patterns[entity_name.upper()] = {
                        'type': ptype,
                        'pattern': pattern_regex,
                        'pattern_raw': pattern_config['pattern'],
                        'context': context_regex,
                        'context_raw': pattern_config.get('context'),
                        'replacement': pattern_config.get('replacement', f"__PII_{entity_name.upper()}__"),
                        'score': score_val,
                    }
                    
                    logger.info(f"Loaded custom pattern for {entity_name}")
                    
                except re.error as e:
                    logger.error(f"Invalid regex pattern for {entity_name}: {e}")
                except Exception as e:
                    logger.error(f"Error loading pattern for {entity_name}: {e}")
            
            return validated_patterns
            
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
            return {}

    def _collect_candidates(self, text: str, extra_custom: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Run full span pipeline: detect → normalize → reclassify → resolve overlaps → prioritize.

        This consolidates heterogeneous detector outputs into a stable, canonical list of span
        dictionaries (for backward compatibility with older interfaces). Internal steps:
            1. Detection: gather raw candidates from all enabled sources.
            2. Normalization: canonicalize labels & apply feature gating.
            3. Reclassification: context/format-based entity refinement (skips config spans).
            4. Overlap Resolution: greedy selection honoring custom score priority.
            5. Priority Ordering: stable sort by semantic importance for downstream usage.

        Args:
            text (str): Target input text segment.
            extra_custom (Dict[str, str] | None): Ad-hoc regex patterns (name->pattern) for one-off runs.

        Returns:
            List[Dict[str, Any]]: Final ordered list of span dictionaries (entity, start, end, text, score?).
        """
        # 1. Raw detection (no filtering)
        spans: List[Span] = []
        spans.extend(self._detect_presidio(text))
        spans.extend(self._detect_spacy(text))
        spans.extend(self._detect_regex(text, extra_custom))
        if self.custom_patterns:
            spans.extend(self._detect_config(text))

        if not spans:
            return []

        # 2. Normalize (apply flags & allowed entity sets)
        spans = self._normalize_spans(spans)
        if not spans:
            return []

        # 3. Reclassify based on context & patterns
        spans = self._reclassify_spans(spans, text)

        # 4. Resolve overlaps after reclassification (keeps highest priority)
        spans = self._resolve_overlaps(spans)

        # 5. Final priority ordering (stable sort by configured hierarchy)
        priority_order = ["SSN", "MEDICAL_ID", "PHONE", "EMAIL", "PERSON", "DATE"]
        priority_rank = {label: idx for idx, label in enumerate(priority_order)}
        spans.sort(key=lambda s: priority_rank.get(s.entity, len(priority_rank)))

        return [s.to_dict() for s in spans]

    # ---------------- Detection Stage -----------------
    def _detect_presidio(self, text: str) -> List[Span]:
        """Invoke Presidio analyzer.

        Args:
            text (str): Raw input text.

        Returns:
            List[Span]: Spans converted from Presidio RecognizerResult objects.
        """
        try:
            results = self.analyzer.analyze(text=text, language="en")
            return [
                Span(start=r.start, end=r.end, text=text[r.start:r.end], raw_entity=r.entity_type, entity=r.entity_type.upper(), source="presidio")
                for r in results
            ]
        except Exception as e:
            logger.debug("Presidio analysis failed: %s", e)
            return []

    def _detect_spacy(self, text: str) -> List[Span]:
        """Collect PERSON & DATE/TIME entities via spaCy NER.

        PERSON is filtered by a case-insensitive allow-list plus minimal length heuristics.
        DATE/TIME is restricted by skip-word set to avoid innocuous tokens (e.g., 'today').

        Args:
            text (str): Raw input text.

        Returns:
            List[Span]: spaCy-derived spans.
        """
        spans: List[Span] = []
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                label = ent.label_.upper()
                ent_text = ent.text
                if label == "PERSON":
                    # Case-insensitive allowed name filtering + length heuristic to reduce false positives
                    if ent_text.lower() in self.allowed_names_lower:
                        continue
                    if len(ent_text.strip()) <= 1:  # skip single-letter initials
                        continue
                    spans.append(Span(ent.start_char, ent.end_char, ent_text, "PERSON", "PERSON", "spacy"))
                elif label in {"DATE", "TIME"} and ent_text.lower() not in DATE_SKIP_WORDS:
                    spans.append(Span(ent.start_char, ent.end_char, ent_text, "DATE", "DATE", "spacy"))
        except Exception as e:
            logger.debug("spaCy analysis failed: %s", e)
        return spans

    def _detect_regex(self, text: str, extra_custom: Optional[Dict[str, str]]) -> List[Span]:
        """Apply built-in + ad-hoc regex patterns for fast deterministic matching.

        Args:
            text (str): Input text.
            extra_custom (Dict[str, str] | None): Additional runtime patterns to merge.

        Returns:
            List[Span]: Regex spans (EMAIL, PHONE, plus extras).
        """
        regex_patterns = dict(DEFAULT_FIELD_PATTERNS)
        if extra_custom:
            regex_patterns.update({k: re.compile(v) for k, v in extra_custom.items()})
        spans: List[Span] = []
        for name, pattern in regex_patterns.items():
            for m in pattern.finditer(text):
                spans.append(Span(m.start(), m.end(), m.group(0), name.upper(), name.upper(), "regex"))
        return spans

    def _detect_config(self, text: str) -> List[Span]:
        """Execute YAML-sourced patterns including optional context preconditions.

        For context_regex entries, the combined pattern ensures the context appears
        immediately before the value; only the value portion is captured as the span.

        Args:
            text (str): Raw input.

        Returns:
            List[Span]: Config-derived spans with associated confidence scores.
        """
        results: List[Span] = []
        for entity_name, config in self.custom_patterns.items():
            pattern_type = config['type']
            pattern_regex = config['pattern']
            context_regex = config.get('context')
            # Use raw pattern strings to avoid duplicating embedded flags/anchors
            if pattern_type == 'context_regex' and context_regex:
                context_raw = config.get('context_raw') or context_regex.pattern
                pattern_raw = config.get('pattern_raw') or pattern_regex.pattern
                # Capture only the value part; context must immediately precede the value
                combined_pattern = re.compile(f"(?:{context_raw})(?P<val>{pattern_raw})", re.IGNORECASE)
                for match in combined_pattern.finditer(text):
                    value_part = match.group('val')
                    value_start = match.start('val')
                    value_end = match.end('val')
                    results.append(Span(value_start, value_end, value_part, entity_name.upper(), entity_name.upper(), "config", config['score']))
            else:
                for match in pattern_regex.finditer(text):
                    results.append(Span(match.start(), match.end(), match.group(0), entity_name.upper(), entity_name.upper(), "config", config['score']))
        return results

    # ---------------- Normalization & Reclassification -----------------
    def _normalize_spans(self, spans: List[Span]) -> List[Span]:
        """Canonicalize entity labels & apply feature gating / filtering.

        Responsibilities:
            * Collapse medical subtype labels into `MEDICAL_ID` (see MEDICAL_CANONICAL_LABELS).
            * Filter date skip words early for efficiency.
            * Enforce strict phone format to avoid masking arbitrary numbers.
            * Respect feature flags (basic_pii, medical_pii).

        Args:
            spans (List[Span]): Raw spans from detection stage.

        Returns:
            List[Span]: Normalized and feature-gated spans.
        """
        canonical: List[Span] = []
        for s in spans:
            raw = s.entity.upper()
            # Only collapse explicitly recognized medical identifier labels.
            if raw.startswith("MEDICAL_ID") or raw == "MEDICAL_ID" or raw in MEDICAL_CANONICAL_LABELS:
                raw = "MEDICAL_ID"
            # Skip date skip words
            if raw == "DATE" and s.text.strip().lower() in DATE_SKIP_WORDS:
                continue
            # Filter phone spans that do NOT meet strict pattern (avoid masking arbitrary numbers)
            if raw == "PHONE" and not STRICT_PHONE_PATTERN.match(s.text.strip()):
                continue
            s.entity = raw
            canonical.append(s)
        # Feature gating
        gated: List[Span] = []
        for s in canonical:
            if s.entity == "MEDICAL_ID" and not self.medical_pii and not self.all_pii:
                continue
            if s.entity in {"PERSON", "EMAIL", "PHONE", "SSN", "DATE"}:
                if not (self.basic_pii or self.all_pii):
                    continue
            gated.append(s)
        return gated

    def _reclassify_spans(self, spans: List[Span], text: str) -> List[Span]:
        """Refine entity labels using ordered contextual & structural rules.

        Locked Spans:
            Spans originating from YAML config (`source == 'config'`) are preserved verbatim
            to prevent user-defined semantics being overridden by generic rules.

        Rule Order:
            1. SSN format validation.
            2. Medical context keyword proximity.
            3. Strict phone pattern validation.
            4. Numeric date pattern classification.

        Args:
            spans (List[Span]): Normalized spans.
            text (str): Original text for contextual window extraction.

        Returns:
            List[Span]: Reclassified spans (order preserved from input list).
        """
        # Do not override entities originating from YAML config
        for s in spans:
            if s.source == "config":
                s._lock_reclass = True  # sentinel attribute (dynamic) to skip rules

        def rule_ssn(span: Span, ctx: str) -> Optional[str]:
            return "SSN" if SSN_PATTERN.match(span.text) else None
        def rule_date(span: Span, ctx: str) -> Optional[str]:
            # Only reclassify as DATE if numeric pattern; do not override existing MEDICAL_ID
            if span.entity != "MEDICAL_ID" and DATE_PATTERN.match(span.text):
                return "DATE"
            return None
        def rule_medical_context(span: Span, ctx: str) -> Optional[str]:
            if span.entity not in {"SSN"} and MEDICAL_CONTEXT_PATTERN.search(ctx):
                return "MEDICAL_ID"
            return None
        def rule_phone(span: Span, ctx: str) -> Optional[str]:
            # ISSUE 3 fix: Only classify as PHONE if strict phone structure matches
            if span.entity not in {"SSN", "MEDICAL_ID"} and STRICT_PHONE_PATTERN.match(span.text.strip()):
                return "PHONE"
            return None

        rules = [rule_ssn, rule_medical_context, rule_phone, rule_date]
        for s in spans:
            if getattr(s, '_lock_reclass', False):
                continue
            context_before = text[max(0, s.start - 40):s.start]
            for rule in rules:
                new_entity = rule(s, context_before)
                if new_entity and new_entity != s.entity:
                    s.entity = new_entity
                    break
        return spans

    # ---------------- Overlap Resolution -----------------
    def _resolve_overlaps(self, spans: List[Span]) -> List[Span]:
        """Resolve overlapping spans preferring higher-confidence & longer matches.

        Selection Heuristics (sorted priority):
            1. Spans with an explicit score (custom patterns) before unscored ones.
            2. Higher score value.
            3. Longer span length (captures more context/value).
            4. Earlier start index (stable tie-breaker).

        Args:
            spans (List[Span]): Potentially overlapping spans.

        Returns:
            List[Span]: Non-overlapping spans sorted by start position.
        """
        if not spans:
            return []
        # Sort for priority: custom scored first, higher score, longer, earlier
        ordered = sorted(
            spans,
            key=lambda s: (
                0 if s.score is not None else 1,
                -(s.score or 0),
                -(s.end - s.start),
                s.start,
            ),
        )
        selected: List[Span] = []
        for span in ordered:
            overlap = False
            for kept in selected:
                if span.start < kept.end and kept.start < span.end:
                    overlap = True
                    break
            if not overlap:
                selected.append(span)
        return sorted(selected, key=lambda s: s.start)

    async def _read_mapping_file(self):
        """Load mapping JSON from persistence layer asynchronously.

        Returns:
            Dict[str, Any]: Entire persisted mapping dictionary.
        """
        async with aiofiles.open(self.persist_path, mode="r", encoding="utf-8") as fh:
            return json.loads(await fh.read())

    async def _write_mapping_file(self, data):
        """Persist full mapping dictionary to disk asynchronously.

        Args:
            data (Dict[str, Any]): Mapping content to serialize.
        """
        async with aiofiles.open(self.persist_path, mode="w", encoding="utf-8") as fh:
            await fh.write(json.dumps(data, ensure_ascii=False, indent=2))

    async def get_mapping(self, mapping_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific mapping record by its identifier.

        Args:
            mapping_id (str): Unique mapping identifier.

        Returns:
            Dict[str, Any] | None: Mapping record or None if missing/unavailable.
        """
        if self.persist:
            async with self._lock:
                try:
                    data = await self._read_mapping_file()
                    return data.get(mapping_id)
                except Exception:
                    return None
        else:
            return self._in_memory_store.get(mapping_id)

    async def unmask(self, text: str, mapping_id: Optional[str] = None, mapping_record: Optional[Dict[str, Any]] = None) -> str:
        """Restore original PII values by substituting placeholders from a mapping.

        Substitution processes longer placeholders first to avoid partial overlaps.

        Args:
            text (str): Masked text containing placeholders.
            mapping_id (str | None): Lookup key for persisted mapping (ignored if mapping_record provided).
            mapping_record (Dict[str, Any] | None): Pre-fetched mapping to bypass disk I/O.

        Returns:
            str: Fully restored text with original PII values.
        """
        record = None
        if mapping_record:
            record = mapping_record
        elif mapping_id:
            if self.persist:
                async with self._lock:
                    try:
                        data = await self._read_mapping_file()
                        record = data.get(mapping_id)
                    except Exception as e:
                        logger.exception("Failed to load mapping: %s", e)
            else:
                record = self._in_memory_store.get(mapping_id)

        if not record:
            logger.warning("No mapping found for id=%s", mapping_id)
            return text

        result = text
        # Replace longer placeholders first to avoid partial replacements (sort by length desc)
        placeholders = sorted(record.get("placeholders", []), key=lambda x: -len(x["placeholder"]))
        for p in placeholders:
            result = result.replace(p["placeholder"], p["original"])

        return result

    def _chunk_text(self, text: str, max_length: int = 1000) -> List[str]:
        """Split long text into bounded-size chunks while preserving natural boundaries.

        Strategy:
            1. Attempt spaCy sentence segmentation (if enabled & model available).
            2. Fallback to naive '. ' splitting if segmentation fails.
            3. Greedy pack sentences until `max_length` would be exceeded, then start new chunk.

        Args:
            text (str): Raw input string.
            max_length (int): Soft size limit per chunk.

        Returns:
            List[str]: Ordered list of chunk strings.
        """
        if len(text) <= max_length:
            return [text]

        sentences: List[str] = []
        if self.use_spacy_sentence_chunking:
            try:
                doc = self.nlp(text)
                sentences = [s.text for s in doc.sents if s.text.strip()]
            except Exception:
                sentences = []
        if not sentences:
            sentences = [s for s in text.split('. ') if s]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        sep = ' '
        for sent in sentences:
            # Add a period if original split removed it (heuristic)
            normalized = sent
            if not normalized.endswith('.') and '.' in text:
                # Avoid adding extra periods to abbreviations; simple heuristic: if original text contains "{sent}. " pattern
                pass
            add_len = len(normalized) + len(sep)
            if current and current_len + add_len > max_length:
                chunks.append(sep.join(current))
                current = []
                current_len = 0
            current.append(normalized)
            current_len += add_len
        if current:
            chunks.append(sep.join(current))
        return chunks

    async def mask(self, data: Any, custom_fields: Optional[Dict[str, str]] = None, store_mapping: bool = True) -> Tuple[Any, Optional[str]]:
        """Detect and mask PII across supported input container types.

        Input Handling:
            * dict/list objects are serialized to JSON for unified span indexing.
            * Non-string values are coerced with `str()`.
            * Large texts are chunked to manage memory and avoid excessive span indices.

        Args:
            data (Any): Input content (str, dict, list, or other serializable object).
            custom_fields (Dict[str, str] | None): Ad-hoc regex patterns for this invocation only.
            store_mapping (bool): Whether to persist the placeholder→original mapping.

        Returns:
            Tuple[Any, Optional[str]]: (Masked data matching original input type, mapping identifier).
        """
        if data is None:
            return None, None

        original_type = type(data)
        if isinstance(data, (dict, list)):
            text = json.dumps(data, ensure_ascii=False, indent=2)
        elif isinstance(data, str):
            text = data
        else:
            text = str(data)

        if not text:
            return text, None

        max_length = 1000
        chunks = [text] if len(text) <= max_length else self._chunk_text(text, max_length=max_length)

        masked_chunks: List[str] = []
        placeholders: List[Dict[str, Any]] = []
        counter_by_entity: Dict[str, int] = {}
        op_uuid = uuid.uuid4().hex[:8]

        for chunk in chunks:
            candidates = self._collect_candidates(chunk, extra_custom=custom_fields)
            if not candidates:
                masked_chunks.append(chunk)
                continue

            candidates_sorted = sorted(candidates, key=lambda x: x["start"])
            masked_parts: List[str] = []
            last_idx = 0
            last_masked_end = -1

            for c in candidates_sorted:
                start, end = c["start"], c["end"]
                entity = c["entity"].upper()
                if chunk[start:end] in self.allowed_text:
                    continue
                if start < last_masked_end:
                    continue
                original = chunk[start:end]
                counter_by_entity[entity] = counter_by_entity.get(entity, 0) + 1
                seq = counter_by_entity[entity]
                placeholder = self._generate_placeholder(entity, seq, op_uuid)
                masked_parts.append(chunk[last_idx:start])
                masked_parts.append(placeholder)
                placeholders.append({"placeholder": placeholder, "original": original, "entity": entity})
                last_idx = end
                last_masked_end = end

            masked_parts.append(chunk[last_idx:])
            masked_chunks.append("".join(masked_parts))

        masked_text = masked_chunks[0] if len(masked_chunks) == 1 else " ".join(masked_chunks)

        mapping_id = None
        if store_mapping:
            mapping_id = str(uuid.uuid4())
            record = {
                "id": mapping_id,
                "timestamp": time.time(),
                "original_text": text,
                "masked_text": masked_text,
                "placeholders": placeholders,
            }
            if self.persist:
                async with self._lock:
                    try:
                        file_data = await self._read_mapping_file()
                        file_data[mapping_id] = record
                        await self._write_mapping_file(file_data)
                    except Exception as e:
                        logger.exception("Failed to persist mapping: %s", e)
            else:
                async with self._lock:
                    self._in_memory_store[mapping_id] = record

        if original_type in (dict, list):
            try:
                masked_text = json.loads(masked_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse masked JSON back to original type")

        return masked_text, mapping_id

    async def mask_with_result(
        self,
        data: Any,
        custom_fields: Optional[Dict[str, str]] = None,
        store_mapping: bool = True,
        include_spans: bool = True,
    ) -> PIIResult:
        """High-level masking API returning structured span + summary metadata.

        Detection is executed before chunking to ensure span offsets reflect the original
        unified string representation. For structured inputs (dict/list) the offsets pertain
        to the JSON-rendered view, not nested object-relative indices.

        Args:
            data (Any): Input content.
            custom_fields (Dict[str, str] | None): One-off supplementary regex patterns.
            store_mapping (bool): Persist mapping for reversibility.
            include_spans (bool): Include raw span list in result (disable for performance if not needed).

        Returns:
            PIIResult: Container wrapping masked output, mapping id, spans, and a summary dict.
        """
        # Stringify early to compute spans on unified view
        original_type = type(data)
        if isinstance(data, (dict, list)):
            text_for_spans = json.dumps(data, ensure_ascii=False, indent=2)
        elif isinstance(data, str):
            text_for_spans = data
        else:
            text_for_spans = str(data)

        spans: List[Span] = []
        if include_spans and text_for_spans:
            spans_dicts = self._collect_candidates(text_for_spans, extra_custom=custom_fields)
            # Convert dicts back to Span for consistency
            for sd in spans_dicts:
                spans.append(Span(start=sd["start"], end=sd["end"], text=sd["text"], raw_entity=sd["entity"], entity=sd["entity"], source="final"))

        masked, mapping_id = await self.mask(data, custom_fields=custom_fields, store_mapping=store_mapping)

        summary = {}
        if mapping_id:
            record = None
            if self.persist:
                record = await self.get_mapping(mapping_id)
            else:
                record = self._in_memory_store.get(mapping_id)
            if record:
                summary = self.mapping_to_pii_dict(record)
        return PIIResult(masked=masked, mapping_id=mapping_id, spans=spans, summary=summary)

    def _generate_placeholder(self, entity: str, seq: int, op_uuid: str) -> str:
        """Generate deterministic placeholder using the active template.

        Supported Tokens:
            {entity}: Upper-cased canonical entity label sanitized for non-alphanumerics.
            {seq}: Monotonic per-entity counter (1-based).
            {uuid}: Operation-level short UUID (constant across placeholders for same mask run).
            {rand4}: Per-placeholder 4-hex entropy for collision avoidance across runs.

        Args:
            entity (str): Canonical entity label.
            seq (int): Sequential index for this entity type.
            op_uuid (str): Short UUID assigned once per masking operation.

        Returns:
            str: Placeholder string inserted into masked text.
        """
        safe_entity = re.sub(r"[^A-Z0-9_]+", "_", entity.upper())
        placeholder = self.placeholder_template
        placeholder = placeholder.replace('{entity}', safe_entity)
        placeholder = placeholder.replace('{seq}', str(seq))
        if '{uuid}' in placeholder:
            placeholder = placeholder.replace('{uuid}', op_uuid)
        if '{rand4}' in placeholder:
            placeholder = placeholder.replace('{rand4}', uuid.uuid4().hex[:4])
        return placeholder

    def mapping_to_pii_dict(self, mapping_record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw placeholder mapping into categorized PII summary.

        Categories:
            CONTACT: Person names, emails, phone numbers, SSNs.
            PII_MEDICAL: Consolidated medical identifiers.
            OTHER: Remaining entities (e.g., DATE) grouped under their entity label.

        Deduplication ensures each original value appears only once per category.
        Empty categories are pruned for concise output.

        Args:
            mapping_record (Dict[str, Any]): Stored mapping structure containing placeholders.

        Returns:
            Dict[str, Any]: Hierarchical PII summary dictionary.
        """
        pii_dict: Dict[str, Any] = {
            "CONTACT": {"name": [], "email": [], "phone": [], "SSN": []},
            "PII_MEDICAL": [],
            "OTHER": {},
        }

        # Deduplicate entries
        seen_entries = set()

        ssn_regex = re.compile(r"^\d{3}-\d{2}-\d{4}$")
        
        # Simple keywords to identify medical entities
        medical_keywords = [
            "MEMBER", "SUBSCRIBER", "MRN", "PATIENT", "RX", "POLICY", 
            "EMPLOYEE", "CLAIM", "HEALTH", "PLAN", "ADMISSION", "ENCOUNTER",
            "VISIT", "HOSPITAL", "EPISODE", "BIN", "GROUP", "MEDICAL"
        ]

        for p in mapping_record.get("placeholders", []):
            original = p.get("original", "")
            entity = p.get("entity", "").upper()

            if original in seen_entries:
                continue

            # SSN handling - always goes to CONTACT
            if entity == "SSN" or ssn_regex.match(original):
                pii_dict["CONTACT"]["SSN"].append(original)
            elif entity in {"PERSON", "NAME"}:
                pii_dict["CONTACT"]["name"].append(original)
            elif entity == "EMAIL":
                pii_dict["CONTACT"]["email"].append(original)
            elif entity == "PHONE":
                pii_dict["CONTACT"]["phone"].append(original)
            # Check if entity contains any medical keyword
            elif any(keyword in entity for keyword in medical_keywords):
                pii_dict["PII_MEDICAL"].append(original)
            else:
                # For anything else (dates, etc.)
                pii_dict["OTHER"].setdefault(entity, []).append(original)

            seen_entries.add(original)

        # Clean up CONTACT - remove empty lists
        pii_dict["CONTACT"] = {
            key: value for key, value in pii_dict["CONTACT"].items()
            if value  # Only keep non-empty lists
        }
        
        # Clean up OTHER - remove empty categories
        pii_dict["OTHER"] = {
            key: value for key, value in pii_dict["OTHER"].items()
            if value  # Only keep non-empty lists
        }
        
        # Remove PII_MEDICAL if empty
        if not pii_dict["PII_MEDICAL"]:
            del pii_dict["PII_MEDICAL"]

        return pii_dict



if __name__ == "__main__":
    # for testing purposes
    # Initialize the PIIUtility for masking and un-masking
    async def main():
        # Test with custom config file
        config_file = "pii_config.yaml"
        
        pii_utility = PIIUtility(
            model_path="./en_core_web_lg-3.8.0",
            persist=False,
            allowed_text=["GB"],
            basic_pii=True,
            medical_pii=True,
            all_pii=False,
            config_path=config_file if os.path.exists(config_file) else None
        )

        # Sample text containing PII
        sample_text = (
            "Describe this situation: Patient John Maria was admitted on January 15th. "
            "Weather is good today with 50 degrees. Subscriber ID: 9877656. "
            "Contact: john.maria@gmail.com, phone +1 (555) 123-4567. "
            "SSN 123-45-6789 and member id: 990867 Jack Ma. Passport no. is A12345678. "
            "my rx group 67895432. Policy number: ABC1234567. Employee ID: EMP-12345"
        )

        print("=" * 80)
        print("TEST 1: String Input")
        print("=" * 80)
        print("Original Text:\n", sample_text)
        
        if pii_utility.custom_patterns:
            print(f"\nLoaded {len(pii_utility.custom_patterns)} custom patterns from config")

        # Test 1: Anonymize regular string
        anonymized_text, mapping_id = await pii_utility.mask(sample_text, store_mapping=True)
        print("\nAnonymized Text:\n", anonymized_text)
        print("Mapping ID:", mapping_id)
        
        # Test 2: Anonymize dict
        print("\n" + "=" * 80)
        print("TEST 2: Dict Input")
        print("=" * 80)
        sample_dict = {
            "patient": "John Maria",
            "email": "john.maria@gmail.com",
            "phone": "+1 (555) 123-4567",
            "ssn": "123-45-6789",
            "subscriber_id": "9877656",
            "details": {
                "member_id": "990867",
                "rx_group": "67895432"
            }
        }
        print("Original Dict:\n", json.dumps(sample_dict, indent=2))
        
        anonymized_dict, mapping_id2 = await pii_utility.mask(sample_dict, store_mapping=True)
        print("\nAnonymized Dict:\n", json.dumps(anonymized_dict, indent=2))
        print("Mapping ID:", mapping_id2)
        
        # Test 3: Anonymize list
        print("\n" + "=" * 80)
        print("TEST 3: List Input")
        print("=" * 80)
        sample_list = [
            "Patient John Maria",
            "Email: john.maria@gmail.com",
            "SSN 123-45-6789",
            {"subscriber id": "9877656"}
        ]
        print("Original List:\n", json.dumps(sample_list, indent=2))
        
        anonymized_list, mapping_id3 = await pii_utility.mask(sample_list, store_mapping=True)
        print("\nAnonymized List:\n", json.dumps(anonymized_list, indent=2))
        print("Mapping ID:", mapping_id3)
    
    # Run the async main function
    asyncio.run(main())
