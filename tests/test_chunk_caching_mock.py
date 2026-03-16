# Test script for chunk caching with mocked LLM calls
import os
from ml.extraction import LLMExtractor, ExtractedObject, ExtractionResult, ObjectMention

# Mock the _extract_batch method to return deterministic objects without calling the LLM
def mock_extract_batch(self, text: str):
    # Create a dummy object for each chunk
    return [ExtractedObject(
        id="obj_1",
        type="Idea",
        canonical_text="Mocked extraction",
        confidence=0.9,
        context="",
        span_start=0,
        span_end=len(text),
    )]

# Patch the method
LLMExtractor._extract_batch = mock_extract_batch

# Ensure a dummy API key is set to bypass the key check
os.environ["GROQ_API_KEY"] = "dummy-key"

extractor = LLMExtractor()

# Create a long dummy text to trigger chunking (approx 8000 chars)
sentence = "This is a test sentence for chunking. "
long_text = sentence * 200

print("Running first extraction (should create cache files)")
result1 = extractor.extract(long_text, note_id="note_cache_test", span_id="span_cache_test")
print(f"Objects extracted: {len(result1.objects)}")

# List cache directory contents
cache_dir = os.path.join(os.path.dirname(__file__), "..", "ml", "cache")
print("Cache directory contents after first run:")
print(os.listdir(cache_dir))

print("Running second extraction (should reuse cache)")
result2 = extractor.extract(long_text, note_id="note_cache_test", span_id="span_cache_test")
print(f"Objects extracted second run: {len(result2.objects)}")
print("Cache directory contents after second run:")
print(os.listdir(cache_dir))
