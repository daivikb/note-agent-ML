from typing import List, Dict, Any
from .graph import KnowledgeGraph
from .extraction import ExtractedObject, Link

import uuid
from datetime import datetime, timezone

class IntelligenceLayer:
    def __init__(self, graph: KnowledgeGraph):
        """
        Initialize the Intelligence Layer.
        
        Args:
            graph: The knowledge graph instance to analyze.
        """
        self.graph = graph
        self.insights = [] # initializing list 

    def make_insight(self, insight_type, severity, payload): # helper function to standardize
        return {
            "id": f"insight_{uuid.uuid4().hex[:8]}",
            "type": insight_type,
            "severity": severity,
            "status": "new",
            "payload": payload, # DB lookup required
            "created_at": datetime.now(timezone.utc).isoformat()
        }

    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """
        Detects contradictions in the knowledge graph.
        
        Returns:
            A list of detected contradictions with severity and evidence.
        """
        # 1. Structural contradictions (explicit edges)
        explicit_contradictions = self.graph.find_contradictions()
        
        # 2. Semantic contradictions (mocked logic)
        # Real implementation would compare embeddings of all Claims
        # and use an LLM to verify if high-similarity claims are contradictory.
        
        results = []
        for c in explicit_contradictions:
            results.append({
                "type": "Explicit",
                "source_text": c['source']['canonical_text'],
                "target_text": c['target']['canonical_text'],
                "severity": "High"
            })
            src, tgt = c["source"], c["target"]
            insight = self.make_insight("contradiction", "high", {
                "claim1_id": src.get("id", ""),
                "claim1_text": src.get("canonical_text", ""),
                "claim1_note": src.get("note_id", ""),
                "claim1_date": src.get("created_at", ""),
                "claim2_id": tgt.get("id", ""),
                "claim2_text": tgt.get("canonical_text", ""),
                "claim2_note": tgt.get("note_id", ""),
                "claim2_date": tgt.get("created_at", ""),
                "confidence": c["edge"].get("confidence", 1.0),
            })
            self.insights.append(insight)
        
        return results
    
    def detect_stale_threads(self, current_date: datetime | None = None) -> List[Dict[str, Any]]:
        thresholds = {"Question": 30, "Task": 60, "Idea": 45} # days without updates
        now = current_date or datetime.now(timezone.utc) # default = today

        for node_id, data in self.graph.graph.nodes(data = True):
            obj_type = data.get("type")
            if obj_type not in thresholds: # skip if not question, or, idea
                continue

            created_raw = data.get("created_at")
            if not created_raw: # skip if no created at date
                continue

            # try block to check 
            try:
                created_date = datetime.fromisoformat(created_raw)
                if created_date.tzinfo is None:
                    created_date = created_date.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            age_days = (now - created_date).days # how many days old
            if age_days < thresholds[obj_type]:
                continue

            incoming_types = {d.get("type") for _, _, d in self.graph.graph.in_edges(node_id, data=True)}
            if incoming_types & {"Answers", "Addresses"}: # skip if already answered
                continue

            severity = "high" if age_days > thresholds[obj_type] * 2 else "medium" # considered high if 2 x threshold
            self.insights.append(self.make_insight("stale_thread", severity, {
                "object_id": node_id,
                "object_type": obj_type,
                "object_text": data.get("canonical_text", ""),
                "age_days": age_days,
                "note_id": data.get("note_id", ""),
                "suggestion": f"You have an open {obj_type} that is {age_days} days old with no documented answer.",
            }))
        return self.insights

    def generate_insights(self) -> List[Dict[str, Any]]:
        """
        Generates insights based on graph structure.
        
        Returns:
            A list of insights.
        """
        insights = []
        
        # Insight 1: Stale Threads (Questions with no answers/links)
        for node, data in self.graph.graph.nodes(data=True):
            if data.get('type') == 'Question':
                # Check if it has any outgoing edges (answers/refinements)
                if self.graph.graph.out_degree(node) == 0:
                    insights.append({
                        "type": "StaleThread",
                        "object_id": node,
                        "text": data.get('canonical_text'),
                        "message": "This question has not been addressed or linked to any answer."
                    })

        # Insight 2: High Centrality Ideas (Core themes)
        centrality = self.graph.custom_centrality()
        # Sort by centrality
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for node_id, score in top_nodes:
            if score > 0:
                node_data = self.graph.graph.nodes[node_id]
                insights.append({
                    "type": "CoreConcept",
                    "object_id": node_id,
                    "text": node_data.get('canonical_text'),
                    "score": score,
                    "message": "This concept is central to the knowledge graph."
                })
                
        return insights
