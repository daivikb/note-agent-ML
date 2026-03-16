-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Notes Table (Metadata)
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    title TEXT,
    content_hash TEXT,
    status TEXT DEFAULT 'created', -- 'created', 'extracted', 'chunked', 'embedded', 'structured', 'resolved'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Files Table (Physical Storage Reference)
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    note_id TEXT REFERENCES notes(id) ON DELETE CASCADE,
    uri TEXT NOT NULL,
    mime_type TEXT,
    size_bytes BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Spans Table (Chunks with Embeddings)
-- Replaces previous 'chunks' table concept
CREATE TABLE IF NOT EXISTS spans (
    id TEXT PRIMARY KEY,
    note_id TEXT REFERENCES notes(id) ON DELETE CASCADE,
    start_char INT,
    end_char INT,
    text TEXT NOT NULL,
    token_count INT,
    embedding VECTOR(384), -- Adjust dimension if model changes (e.g. 1536 for OpenAI)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create HNSW Index for vector search
CREATE INDEX IF NOT EXISTS spans_embedding_idx ON spans USING hnsw (embedding vector_cosine_ops);
-- Create Full Text Search Index
CREATE INDEX IF NOT EXISTS spans_text_idx ON spans USING GIN (to_tsvector('english', text));


-- 4. Objects Table (Extracted Entities)
CREATE TABLE IF NOT EXISTS objects (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    type TEXT NOT NULL, -- 'Idea', 'Claim', 'Assumption', 'Question', 'Task', 'Evidence', 'Definition'
    canonical_text TEXT NOT NULL,
    confidence FLOAT,
    status TEXT DEFAULT 'active', -- 'active', 'merged_into_{id}'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Object Mentions Table (Links Objects to Spans)
CREATE TABLE IF NOT EXISTS object_mentions (
    id TEXT PRIMARY KEY,
    object_id TEXT REFERENCES objects(id) ON DELETE CASCADE,
    span_id TEXT REFERENCES spans(id) ON DELETE CASCADE,
    note_id TEXT REFERENCES notes(id) ON DELETE CASCADE, -- Denormalized for query speed
    role TEXT DEFAULT 'primary', -- 'primary', 'supporting'
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. Links Table (Relationships between Objects)
CREATE TABLE IF NOT EXISTS links (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    src_object_id TEXT REFERENCES objects(id) ON DELETE CASCADE,
    dst_object_id TEXT REFERENCES objects(id) ON DELETE CASCADE,
    type TEXT NOT NULL, -- 'Supports', 'Contradicts', 'Refines', 'DependsOn', 'SameAs', 'Causes'
    confidence FLOAT,
    evidence_span_id TEXT REFERENCES spans(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. Human Reviews Table (HITL Feedback Loop)
CREATE TABLE IF NOT EXISTS human_reviews (
    id SERIAL PRIMARY KEY,
    object_id TEXT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    original_type TEXT NOT NULL,
    corrected_type TEXT,
    original_text TEXT NOT NULL,
    corrected_text TEXT,
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'accepted', 'rejected', 'corrected'
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_human_reviews_status ON human_reviews(status);
CREATE INDEX IF NOT EXISTS idx_human_reviews_note ON human_reviews(note_id);

-- 8. Insights Table (Higher Level Intelligence)
CREATE TABLE IF NOT EXISTS insights (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    type TEXT NOT NULL, -- 'contradiction', 'stale_thread', 'consolidation_opportunity'
    severity TEXT, -- 'high', 'medium', 'low'
    status TEXT DEFAULT 'new', -- 'new', 'resolved', 'dismissed'
    payload JSONB, -- Flexible JSON for diverse insight types
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
