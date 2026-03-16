import json
import os
import re
import uuid
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
from ml.config import config
from ml.feedback import (
    init_feedback_db,
    log_extraction,
    get_few_shot_examples,
    format_few_shot_block,
)


@dataclass
class Chunk:
    text: str
    start_char_idx: int
    end_char_idx: int
    token_count: int
    metadata: Dict[str, Any]

# ── Structured output models (Stage 3/4) ──────────────────────────────────────

class ExtractedObject(BaseModel):
    """
    Represents a knowledge object extracted from text (objects table row).
    This is the core data unit for the Knowledge Graph.
    """
    id: str = Field(description="Unique identifier for the object (e.g., obj_001)")
    type: Literal['Idea', 'Claim', 'Assumption', 'Question', 'Task', 'Evidence', 'Definition']
    canonical_text: str = Field(description="The concise, canonical text of the object")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    context: Optional[str] = Field(default=None, description="Surrounding context from source text")
    span_start: Optional[int] = Field(default=None, description="Start char position in source text")
    span_end: Optional[int] = Field(default=None, description="End char position in source text")

    @field_validator('confidence')
    @classmethod
    def clamp_confidence(cls, v):
        """Ensures confidence scores remain within the 0.0 to 1.0 range."""
        return max(0.0, min(1.0, v))


class Link(BaseModel):
    """
    Represents a semantic relationship between two knowledge objects (links table row).
    This connects the nodes in the Knowledge Graph.
    """
    source_id: str
    target_id: str
    type: Literal['Supports', 'Contradicts', 'Refines', 'DependsOn', 'SameAs', 'Causes']
    confidence: float
    evidence_span_id: Optional[str] = Field(default=None, description="Span where this link was found")

    @field_validator('confidence')
    @classmethod
    def clamp_confidence(cls, v):
        """Ensures confidence scores remain within the 0.0 to 1.0 range."""
        return max(0.0, min(1.0, v))


class ObjectMention(BaseModel):
    """Links an extracted object back to its source span (object_mentions table row)."""
    object_id: str
    note_id: str = Field(default="note_local", description="Source note ID")
    span_id: str = Field(default="span_full", description="Source span ID")
    role: str = Field(default="primary")
    confidence: float


class ExtractionResult(BaseModel):
    """Complete extraction output containing objects, links, and provenance mentions."""
    objects: List[ExtractedObject]
    links: List[Link]
    mentions: List[ObjectMention] = []


# ── LLM Extractor ─────────────────────────────────────────────────────────────

# Per-type definitions matching ML_doc.pdf Stage 4 (page 8-9)
# "For each object type: Design LLM prompt specific to that type"
OBJECT_TYPE_DEFINITIONS = {
    'Idea': 'novel concepts, proposals, strategies, or creative thoughts',
    'Claim': 'factual assertions that can be true or false',
    'Assumption': 'unstated premises taken for granted',
    'Question': 'open questions or inquiries',
    'Task': 'action items or to-dos',
    'Evidence': 'data, observations, or citations supporting a claim',
    'Definition': 'formal definitions of terms or concepts',
}

# Per-type prompt template (exact format from ML_doc.pdf page 9)
PER_TYPE_SYSTEM_PROMPT = "You are an expert at analyzing notes and extracting structured information."

# Link types for relationship extraction pass
LINK_TYPES = {
    'Supports': 'Source provides direct evidence, data, or logical proof for the target.',
    'Contradicts': 'Source explicitly conflicts with, opposes, or refutes the target.',
    'Refines': 'Source adds detail, nuances, or clarifies the target without changing its core meaning.',
    'DependsOn': 'Source requires the target to be true or completed first (prerequisite).',
    'SameAs': 'Source and target express the same fundamental idea or fact using different wording.',
    'Causes': 'Source leads to the target as a direct consequence or result.',
}


def _attempt_json_repair(raw: str) -> Optional[dict]:
    """
    Safety Fallback: Attempts to repair malformed JSON from the LLM.
    Useful for common issues like markdown fences or trailing commas 
    that break standard json.loads().
    """
    # Strip markdown fences
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)

    # Strip non-printable control characters (except newlines/tabs)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)

    # Fix trailing commas before } or ]
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


class LLMExtractor:
    """
    Core Extraction Engine responsible for transforming raw text into a structured Knowledge Graph.
    Implements a robust 8-step process with error handling and traceablity.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, verbose: bool = False):
        """
        Initialization: Resolves API keys and sets the LLM provider (OpenAI or Groq).
        
        Args:
            api_key: Optional override for environment keys.
            model: Optional override for the default model choice.
            verbose: Enables debug printing (Raw JSON and Linker details).
        """
        # Resolve key and provider
        openai_key = os.environ.get("OPENAI_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")
        
        resolved_key = api_key or openai_key or groq_key
        
        if resolved_key and (resolved_key.startswith("sk-") or "openai" in str(api_key).lower()):
            base_url = "https://api.openai.com/v1"
            self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        else:
            base_url = "https://api.groq.com/openai/v1"
            self.model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

        if not resolved_key:
            raise ValueError("No API key provided. Check your .env file.")

        self.client = OpenAI(
            api_key=resolved_key,
            base_url=base_url,
        )

        # Simulated database tables
        self.objects_table: List[ExtractedObject] = []
        self.links_table: List[Link] = []
        self.verbose = verbose

        # HITL: Initialize feedback store and load few-shot examples
        init_feedback_db()
        self._few_shot_block = self._build_few_shot_block()

    def _split_into_chunks(self, text: str) -> List[Chunk]:
        """Split the input text into manageable chunks based on a character limit.
        This simple heuristic uses a fixed size (e.g., 3000 chars) which works well
        for most LLM token limits. Adjust as needed.
        """
        max_chars = 3000  # Roughly corresponds to typical token limits
        chunks: List[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                start_char_idx=start,
                end_char_idx=end,
                token_count=0,
                metadata={}
            ))
            start = end
        return chunks

    def _cache_path_for_chunk(self, chunk_text: str) -> Path:
        """Return a deterministic cache file path for a chunk based on its SHA256 hash.
        Includes the few-shot block so that new human corrections invalidate old cache."""
        cache_input = chunk_text + (self._few_shot_block or "")
        hash_digest = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()
        cache_dir = Path(__file__).parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{hash_digest}.json"

    def _load_cached_objects(self, cache_path: Path) -> List[ExtractedObject]:
        """Load cached objects from a JSON file and return as ExtractedObject instances."""
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            objs = []
            for item in data.get("objects", []):
                objs.append(ExtractedObject(**item))
            return objs
        except Exception:
            return []

    def _save_objects_to_cache(self, cache_path: Path, objects: List[ExtractedObject]):
        """Save a list of ExtractedObject instances to the given cache path as JSON."""
        data = {"objects": [obj.dict() for obj in objects]}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    def extract(self, text: str, note_id: str = "note_local", span_id: str = "span_full", chunks: Optional[List[Chunk]] = None) -> ExtractionResult:
        """
        The Master Orchestrator: Executes the 8-step pipeline to extract knowledge.
        
        This method is "Resilient": It can handle raw text directly, or work with 
        pre-processed chunks. If chunks are provided but missing offsets, it 
        re-calculates them locally to ensure every object is traceable.
        
        Args:
            text: Raw unstructured text.
            note_id: Source document ID for the database.
            span_id: Default span ID if chunk-based mapping fails.
            chunks: List of Chunk objects from the chunking pipeline.
        """
        # Prepare chunks (use provided or split)
        if chunks is None:
            chunks = self._split_into_chunks(text)

        all_objects: List[ExtractedObject] = []
        # Process each chunk, using cache when possible
        for chunk in chunks:
            cache_path = self._cache_path_for_chunk(chunk.text)
            if cache_path.is_file():
                cached_objs = self._load_cached_objects(cache_path)
                if cached_objs:
                    all_objects.extend(cached_objs)
                    continue
            # Not cached – run extraction on this chunk
            objs = self._extract_batch(chunk.text)
            if objs:
                self._save_objects_to_cache(cache_path, objs)
                all_objects.extend(objs)

        if not all_objects:
            print("[Extraction] ✗ No objects extracted.")
            return ExtractionResult(objects=[], links=[], mentions=[])

        # Deduplicate and re-number IDs
        all_objects = self._deduplicate_objects(all_objects)

        # Identify relationships across the full text
        links = self._extract_relationships(text, all_objects)

        # Build mentions linking objects to spans
        mentions: List[ObjectMention] = []
        for obj in all_objects:
            resolved_span_id = span_id
            if obj.span_start is not None:
                for idx, chunk in enumerate(chunks):
                    if chunk.start_char_idx <= obj.span_start <= chunk.end_char_idx:
                        resolved_span_id = f"span_{idx:03d}"
                        break
            mentions.append(ObjectMention(
                object_id=obj.id,
                note_id=note_id,
                span_id=resolved_span_id,
                role="primary",
                confidence=obj.confidence,
            ))
            self._save_to_objects_table(obj)

        for link in links:
            self._save_to_links_table(link)

        # Ensure confidence scores are within [0.0, 1.0]
        for obj in all_objects:
            obj.confidence = max(0.0, min(1.0, obj.confidence))

        # HITL: Log all extracted objects as 'pending' for human review
        log_extraction(note_id, all_objects)

        # Summary print
        type_counts = {}
        for obj in all_objects:
            type_counts[obj.type] = type_counts.get(obj.type, 0) + 1
        counts_str = ", ".join(f"{c} {t}{'s' if c != 1 else ''}" for t, c in type_counts.items())
        few_shot_info = " (with few-shot examples)" if self._few_shot_block else " (zero-shot)"
        print(f"[Extraction] ✓ Extracted {len(all_objects)} objects ({counts_str}), {len(links)} links  (model={self.model}){few_shot_info}")

        return ExtractionResult(objects=all_objects, links=links, mentions=mentions)
        """
        The Master Orchestrator: Executes the 8-step pipeline to extract knowledge.
        
        This method is "Resilient": It can handle raw text directly, or work with 
        pre-processed chunks. If chunks are provided but missing offsets, it 
        re-calculates them locally to ensure every object is traceable.

        Args:
            text: Raw unstructured text.
            note_id: Source document ID for the database.
            span_id: Default span ID if chunk-based mapping fails.
            chunks: List of Chunk objects from the chunking pipeline.
        """
        # Step 1: Note text is already loaded and passed in
        all_objects: List[ExtractedObject] = []
        obj_counter = 0

        # Step 2: Extract all object types in a single unified pass
        all_objects = self._extract_batch(text)

        if not all_objects:
            print("[Extraction] ✗ No objects extracted.")
            return ExtractionResult(objects=[], links=[], mentions=[])

        # Deduplicate and re-number IDs (Technical cleanup)
        all_objects = self._deduplicate_objects(all_objects)

        # Step 7: Identify relationships between objects
        links = self._extract_relationships(text, all_objects)

        # Step 6 & 8: Process and "Save" data
        mentions = []
        
        # Calculate chunk offsets locally if they are missing (-1)
        # This allows us to map objects back to spans without depending on a specific chunker.
        if chunks:
            current_search_pos = 0
            for chunk in chunks:
                if chunk.start_char_idx <= 0:
                    # Create a regex that allows any whitespace between the significant tokens of the chunk
                    parts = [re.escape(p) for p in chunk.text.split() if p]
                    if not parts:
                        continue
                    flexible_pattern = r"\s+".join(parts)
                    
                    match = re.search(flexible_pattern, text[current_search_pos:], re.DOTALL)
                    if match:
                        chunk.start_char_idx = current_search_pos + match.start()
                        chunk.end_char_idx = current_search_pos + match.end()
                        current_search_pos = chunk.start_char_idx + 1
                    else:
                        match_global = re.search(flexible_pattern, text, re.DOTALL)
                        if match_global:
                            chunk.start_char_idx = match_global.start()
                            chunk.end_char_idx = match_global.end()

        for obj in all_objects:
            # Step 6: Link back to source span
            resolved_span_id = span_id
            if chunks and obj.span_start is not None:
                for idx, chunk in enumerate(chunks):
                    # Check if object starts within this chunk
                    if chunk.start_char_idx != -1 and chunk.start_char_idx <= obj.span_start <= chunk.end_char_idx:
                        resolved_span_id = f"span_{idx:03d}"
                        break
            
            mentions.append(ObjectMention(
                object_id=obj.id,
                note_id=note_id,
                span_id=resolved_span_id,
                role="primary",
                confidence=obj.confidence,
            ))
            # Step 6: Insert into objects table (Simulated)
            self._save_to_objects_table(obj)

        for link in links:
            # Step 8: Insert into links table (Simulated)
            self._save_to_links_table(link)

        # Print summary
        type_counts = {}
        for obj in all_objects:
            type_counts[obj.type] = type_counts.get(obj.type, 0) + 1
        counts_str = ", ".join(f"{count} {t}{'s' if count != 1 else ''}" for t, count in type_counts.items())

        print(f"[Extraction] ✓ Extracted {len(all_objects)} objects ({counts_str}), "
              f"{len(links)} links  (model={self.model})")

        return ExtractionResult(objects=all_objects, links=links, mentions=mentions)

    def _save_to_objects_table(self, obj: ExtractedObject):
        """
        Step 6: Simulated database insertion. 
        In production, this would perform a SQL INSERT into the 'objects' table.
        """
        self.objects_table.append(obj)

    def _save_to_links_table(self, link: Link):
        """
        Step 8: Simulated database insertion.
        In production, this would perform a SQL INSERT into the 'links' table.
        """
        self.links_table.append(link)

    def _deduplicate_objects(self, all_objects: List[ExtractedObject]) -> List[ExtractedObject]:
        """
        Technical Cleanup: Removes redundant extractions based on semantic overlap.
        Orders objects by type priority to ensure the most specific classification is kept.
        """
        TYPE_PRIORITY = {'Idea': 1, 'Question': 2, 'Task': 3, 'Assumption': 4,
                         'Definition': 5, 'Evidence': 6, 'Claim': 7}
        all_objects.sort(key=lambda o: TYPE_PRIORITY.get(o.type, 99))

        deduped = []
        seen_texts = []
        for obj in all_objects:
            obj_text = obj.canonical_text.lower().strip()
            is_dup = False
            for seen in seen_texts:
                words_a = set(obj_text.split())
                words_b = set(seen.split())
                if not words_a or not words_b: continue
                overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
                if overlap > 0.8:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(obj)
                seen_texts.append(obj_text)

        # Re-number IDs for clean presentation
        for i, obj in enumerate(deduped):
            obj.id = f"obj_{i + 1:03d}"
        
        if len(deduped) < len(all_objects):
             print(f"  [Dedup] {len(all_objects)} → {len(deduped)} objects (removed {len(all_objects) - len(deduped)} duplicates)")
        return deduped

    def _build_few_shot_block(self) -> str:
        """
        HITL: Build a few-shot prompt block from human-reviewed corrections.
        Returns empty string if no feedback exists yet (graceful zero-shot fallback).
        """
        examples = get_few_shot_examples(limit=5)
        block = format_few_shot_block(examples)
        if block and self.verbose:
            print(f"[HITL] Loaded {len(examples)} few-shot examples from feedback store")
        return block

    def _extract_batch(self, text: str, is_retry: bool = False) -> List[ExtractedObject]:
        """
        Unified Pass (Steps 2-5): Extracts all knowledge objects in one LLM call.
        This optimizes for speed and cost while providing the model with full context.
        Includes built-in validation (Step 4) and regeneration (Step 5).

        HITL Enhancement: If human feedback exists, injects few-shot examples into
        the prompt to improve classification accuracy.
        """
        type_definitions_str = "\n".join([f"- {t}: {d}" for t, d in OBJECT_TYPE_DEFINITIONS.items()])

        # HITL: Inject few-shot correction examples if available
        few_shot_section = ""
        if self._few_shot_block:
            few_shot_section = f"\n{self._few_shot_block}\nNow extract objects from the following text:\n"
        
        user_prompt = f"""From the following text, extract all knowledge objects matching these definitions:

{type_definitions_str}
{few_shot_section}
Text:
\"\"\"
{text}
\"\"\"

Return JSON:
{{
  "objects": [
    {{
      "type": "Idea|Claim|Assumption|Question|Task|Evidence|Definition",
      "text": "<the verbatim text segment>",
      "context": "<surrounding context for clarity>",
      "confidence": 0.0-1.0,
      "span_start": <char position>,
      "span_end": <char position>
    }}
  ]
}}

Be precise and exhaustive. Ensure every important idea, claim, or question is captured."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": PER_TYPE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config['temperature'],
                max_tokens=config['max_tokens'],
                timeout=config['timeout'],
            )

            raw_json = response.choices[0].message.content
            
            if self.verbose:
                print(f"\n[DEBUG] Raw JSON Response (Objects):\n{raw_json}\n")

            # Step 3: Parse JSON
            parsed = None
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError:
                # Step 5: If malformed → attempt repair
                parsed = _attempt_json_repair(raw_json)

            # Step 5: If still invalid → REGENERATE (once)
            if parsed is None:
                if not is_retry:
                    print(f"  [Batch] ⚠ Malformed JSON. Attempting REGENERATION...")
                    return self._extract_batch(text, is_retry=True)
                else:
                    print(f"  [Batch] ✗ Regeneration failed.")
                    return []

            # Step 4: Validate against JSON schema
            items = parsed.get('objects', [])
            
            objects = []
            for i, item in enumerate(items):
                try:
                    obj = ExtractedObject(
                        id=f"temp_{i}", # Re-numbered by _deduplicate_objects
                        type=item.get('type'),
                        canonical_text=item.get('text', item.get('canonical_text', '')),
                        confidence=item.get('confidence', 0.8),
                        context=item.get('context'),
                        span_start=item.get('span_start'),
                        span_end=item.get('span_end'),
                    )
                    objects.append(obj)
                except Exception:
                    continue  # Skip malformed items

            # Logging summary of extraction
            type_counts = {}
            for obj in objects:
                type_counts[obj.type] = type_counts.get(obj.type, 0) + 1
            type_info = ", ".join([f"{c} {t}" for t, c in type_counts.items()])
            print(f"  [Batch] → Extracted {len(objects)} objects ({type_info})")

            return objects

        except Exception as e:
            if not is_retry:
                print(f"  [Batch] ⚠ LLM call failed ({e}). Attempting REGENERATION...")
                return self._extract_batch(text, is_retry=True)
            print(f"  [Batch] ✗ LLM call failed significantly: {e}")
            return []

    def _extract_relationships(self, text: str, objects: List[ExtractedObject], is_retry: bool = False) -> List[Link]:
        """
        Relationship Pass (Step 7): Analyzes the extracted nodes to find edges.
        Specific logic for nuanced links like 'Contradicts' or 'Causes' is applied here.
        """
        if len(objects) < 2:
            return []

        # Build a summary of all objects for the relationship prompt
        objects_summary = "\n".join(
            f"  {obj.id}: [{obj.type}] \"{obj.canonical_text}\"" for obj in objects
        )

        link_types_str = "\n".join(f"  - {k}: {v}" for k, v in LINK_TYPES.items())

        user_prompt = f"""Given the following text and extracted objects, identify ALL semantic relationships between them.
Pay close attention to complex relationships like Contradictions, Refinements, and Dependencies.

Text:
\"\"\"
{text}
\"\"\"

Extracted objects:
{objects_summary}

Relationship types:
{link_types_str}

Return JSON:
{{
  "links": [
    {{
      "source_id": "<obj_XXX>",
      "target_id": "<obj_YYY>",
      "type": "Supports|Contradicts|Refines|DependsOn|SameAs|Causes",
      "reasoning": "<brief explanation of why this link exists>",
      "confidence": 0.0-1.0
    }}
  ]
}}

Be exhaustive and highly perceptive. Look for:
1. PREREQUISITES: Does X need Y to happen first? (DependsOn)
2. CONFLICTS: Does X imply Y is not true or redundant? (Contradicts)
3. DETAILS: Does X provide a specific number, name, or date for Y? (Refines)

Only include relationships where there is a clear semantic connection supported by the text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": PER_TYPE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config['temperature'],
                max_tokens=config['max_tokens'],
                timeout=config['timeout'],
            )

            raw_json = response.choices[0].message.content

            if self.verbose:
                print(f"\n[DEBUG] Raw JSON Response (Links):\n{raw_json}\n")

            parsed = None
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError:
                parsed = _attempt_json_repair(raw_json)

            if parsed is None:
                if not is_retry:
                    print(f"  [Relationships] ⚠ Malformed JSON. Attempting REGENERATION...")
                    return self._extract_relationships(text, objects, is_retry=True)
                return []

            # Validate each link
            valid_ids = {obj.id for obj in objects}
            links = []
            for item in parsed.get('links', []):
                try:
                    link = Link(**item)
                    if link.source_id in valid_ids and link.target_id in valid_ids:
                        links.append(link)
                except Exception:
                    continue

            return links

        except Exception as e:
            if not is_retry:
                 print(f"  [Relationships] ⚠ LLM call failed ({e}). Attempting REGENERATION...")
                 return self._extract_relationships(text, objects, is_retry=True)
            print(f"  [Relationships] ✗ LLM call failed significantly: {e}")
            return []
