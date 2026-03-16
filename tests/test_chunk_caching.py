# Test script for chunk caching
import os
from ml.extraction import LLMExtractor

# Create a long dummy text (repeat a sentence)
sentence = "This is a test sentence for chunking. "
long_text = sentence * 200  # approx 8000 chars

extractor = LLMExtractor()
print("Running first extraction (should create cache files)")
result1 = extractor.extract(long_text, note_id="note_cache_test", span_id="span_cache_test")
print(f"Objects extracted: {len(result1.objects)}")

# List cache directory
cache_dir = os.path.join(os.path.dirname(__file__), "..", "ml", "cache")
print("Cache directory contents after first run:")
print(os.listdir(cache_dir))

print("Running second extraction (should reuse cache)")
result2 = extractor.extract(long_text, note_id="note_cache_test", span_id="span_cache_test")
print(f"Objects extracted second run: {len(result2.objects)}")
print("Cache directory contents after second run:")
print(os.listdir(cache_dir))
