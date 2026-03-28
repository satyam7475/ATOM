import sqlite3
import json
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

from core.profiler import profile

logger = logging.getLogger("atom.brain.memory_graph")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed. Vector memory will be disabled.")

@dataclass
class MemoryNode:
    id: str
    type: str  # 'episodic', 'semantic', 'procedural'
    data: Dict[str, Any]
    relationships: List[Tuple[str, str, str]]  # e.g., ("user", "works_on", "backend_project")
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 1.0
    embedding_text: str = "" # Text representation for vector search
    
    @property
    def base_score(self) -> float:
        return self.importance + (self.access_count * 0.1)

def _memory_index_for_node_type(node_type: str) -> str:
    """Route graph node types to episodic / semantic / task vector indices."""
    if node_type == "episodic":
        return "episodic"
    if node_type in ("task", "procedural"):
        return "task"
    return "semantic"


def _route_query_type_to_index(query_type: Optional[str]) -> Optional[str]:
    """Map high-level query classes to vector indices (V6.5 routing)."""
    if not query_type:
        return None
    m = {"plan": "episodic", "knowledge": "semantic", "task": "task"}
    return m.get(query_type.strip().lower())


def _distance_to_similarity(distance: float) -> float:
    """Chroma cosine space: distance in [0, 2] → similarity in [0, 1]."""
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - d))


class VectorMemory:
    """Vector layer with episodic / semantic / task routing (single collection + metadata)."""

    def __init__(self, persist_directory: str = "atom_vector_db"):
        self.enabled = CHROMA_AVAILABLE
        if not self.enabled:
            self.collection = None  # type: ignore[assignment]
            return

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="atom_memories",
            metadata={"hnsw:space": "cosine"},
        )

    def add_memory(
        self,
        node_id: str,
        text: str,
        metadata: dict,
        memory_index: Optional[str] = None,
    ):
        if not self.enabled or not text:
            return
        try:
            idx = memory_index or _memory_index_for_node_type(str(metadata.get("type", "semantic")))
            meta = dict(metadata)
            meta["memory_index"] = idx
            self.collection.upsert(
                documents=[text],
                metadatas=[meta],
                ids=[node_id],
            )
        except Exception as e:
            logger.error(f"Failed to add to vector memory: {e}")

    def search(
        self,
        query: str,
        limit: int = 5,
        *,
        memory_index: Optional[str] = None,
        query_type: Optional[str] = None,
        similarity_threshold: float = 0.0,
        overfetch: int = 2,
    ) -> List[Dict[str, Any]]:
        """Query vectors with optional index routing and similarity pruning."""
        if not self.enabled or not query:
            return []
        try:
            routed = memory_index or _route_query_type_to_index(query_type)
            where = {"memory_index": routed} if routed else None
            n_fetch = max(limit * overfetch, limit, 1)
            kwargs = {
                "query_texts": [query],
                "n_results": n_fetch,
            }
            if where is not None:
                kwargs["where"] = where

            results = self.collection.query(**kwargs)
            # Legacy rows without memory_index: retry unfiltered if routed query is empty
            if (
                routed
                and results
                and results.get("ids")
                and len(results["ids"][0]) == 0
            ):
                results = self.collection.query(query_texts=[query], n_results=n_fetch)

            matches: List[Dict[str, Any]] = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    dist = 0.0
                    if results.get("distances") and results["distances"]:
                        dist = results["distances"][0][i]
                    sim = _distance_to_similarity(dist)
                    if sim < similarity_threshold:
                        continue
                    matches.append({
                        "id": results["ids"][0][i],
                        "distance": dist,
                        "similarity": sim,
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    })
                    if len(matches) >= limit:
                        break
            return matches
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

class MemoryGraph:
    def __init__(self, db_path: str = "atom_memory.db"):
        self.db_path = db_path
        self.vector_db = VectorMemory()
        self._query_cache: "OrderedDict[str, List[MemoryNode]]" = OrderedDict()
        self._query_cache_max = 500
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for graph storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Create nodes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT,
                    data JSON,
                    timestamp REAL,
                    access_count INTEGER,
                    importance REAL
                )
            ''')
            # Create edges table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_edges (
                    source_id TEXT,
                    relation TEXT,
                    target_id TEXT,
                    FOREIGN KEY(source_id) REFERENCES memory_nodes(id),
                    FOREIGN KEY(target_id) REFERENCES memory_nodes(id)
                )
            ''')
            # Add indexes for fast retrieval
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON memory_nodes(type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON memory_edges(source_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_target ON memory_edges(target_id)')
            conn.commit()
            
    def add_node(self, node: MemoryNode):
        """Add a memory node and its relationships to the graph."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert or replace node
            cursor.execute(
                'INSERT OR REPLACE INTO memory_nodes (id, type, data, timestamp, access_count, importance) VALUES (?, ?, ?, ?, ?, ?)',
                (node.id, node.type, json.dumps(node.data), node.timestamp, node.access_count, node.importance)
            )
            
            # Insert relationships
            for source, relation, target in node.relationships:
                cursor.execute(
                    'INSERT INTO memory_edges (source_id, relation, target_id) VALUES (?, ?, ?)',
                    (source, relation, target)
                )
            conn.commit()

        self.invalidate_query_cache()

        # Add to vector memory if text is provided
        if node.embedding_text:
            self.vector_db.add_memory(
                node_id=node.id,
                text=node.embedding_text,
                metadata={"type": node.type, "importance": node.importance, "timestamp": node.timestamp},
                memory_index=_memory_index_for_node_type(node.type),
            )
            
    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a specific node by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, type, data, timestamp, access_count, importance FROM memory_nodes WHERE id = ?', (node_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
                
            # Update access count
            cursor.execute('UPDATE memory_nodes SET access_count = access_count + 1 WHERE id = ?', (node_id,))
            conn.commit()
                
            # Get relationships
            cursor.execute('SELECT source_id, relation, target_id FROM memory_edges WHERE source_id = ? OR target_id = ?', (node_id, node_id))
            edges = cursor.fetchall()
            
            return MemoryNode(
                id=row[0],
                type=row[1],
                data=json.loads(row[2]),
                relationships=edges,
                timestamp=row[3] if row[3] is not None else time.time(),
                access_count=(row[4] or 0) + 1,
                importance=row[5] if row[5] is not None else 1.0
            )
            
    def _decay(self, timestamp: float) -> float:
        """Calculate memory decay based on time elapsed."""
        age_hours = (time.time() - timestamp) / 3600.0
        # Half-life of 24 hours
        return 0.5 ** (age_hours / 24.0)
        
    def _calculate_score(self, node: MemoryNode, semantic_score: float = 0.0) -> float:
        """Calculate dynamic relevance score based on 3-layer ranking."""
        recency_weight = self._decay(node.timestamp)
        importance_weight = node.importance
        
        # Final Score = (semantic_similarity * 0.5) + (recency_weight * 0.3) + (importance_score * 0.2)
        return (semantic_score * 0.5) + (recency_weight * 0.3) + (importance_weight * 0.2)

    def invalidate_query_cache(self) -> None:
        self._query_cache.clear()

    def _query_cache_key(self, query_params: Dict[str, Any], context: Optional[Dict[str, Any]], limit: int) -> str:
        return json.dumps(
            {"q": query_params, "ctx": context or {}, "limit": limit},
            sort_keys=True,
            default=str,
        )

    @profile("memory")
    def query(self, query_params: Dict[str, Any], context: Dict[str, Any] = None, limit: int = 10) -> List[MemoryNode]:
        """Retrieve memory nodes using Hybrid Retrieval (Graph + Vector)."""
        qp = dict(query_params)
        query_type = qp.pop("query_type", None)
        top_k = int(qp.pop("top_k", limit))
        similarity_threshold = float(qp.pop("similarity_threshold", 0.7))
        effective_limit = max(1, min(top_k, 64))

        cache_key = self._query_cache_key(
            {"qp": qp, "query_type": query_type, "top_k": effective_limit, "sim": similarity_threshold},
            context,
            effective_limit,
        )
        if cache_key in self._query_cache:
            self._query_cache.move_to_end(cache_key)
            return self._query_cache[cache_key]

        nodes_dict = {}
        semantic_scores = {}
        
        # 1. Semantic Search (Vector Layer) — routed index + top_k + threshold
        if "text" in qp and self.vector_db.enabled:
            vector_results = self.vector_db.search(
                qp["text"],
                limit=effective_limit * 2,
                query_type=query_type,
                similarity_threshold=similarity_threshold,
            )
            for res in vector_results:
                sim_score = float(res.get("similarity", max(0.0, 1.0 - (res["distance"] / 2.0))))
                semantic_scores[res["id"]] = sim_score
                node = self.get_node(res["id"])
                if node:
                    nodes_dict[node.id] = node
                    
        # 2. Graph Search (Relational Layer)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if "type" in qp:
                cursor.execute('SELECT id FROM memory_nodes WHERE type = ?', (qp["type"],))
                for row in cursor.fetchall():
                    if row[0] not in nodes_dict:
                        node = self.get_node(row[0])
                        if node:
                            nodes_dict[node.id] = node
                            
            if "target" in qp:
                target = qp["target"]
                cursor.execute('SELECT source_id FROM memory_edges WHERE target_id = ?', (target,))
                for row in cursor.fetchall():
                    if row[0] not in nodes_dict:
                        node = self.get_node(row[0])
                        if node:
                            nodes_dict[node.id] = node
                            
        # 3. Hybrid Ranking
        scored_nodes = []
        for node_id, node in nodes_dict.items():
            sem_score = semantic_scores.get(node_id, 0.0)
            final_score = self._calculate_score(node, semantic_score=sem_score)
            scored_nodes.append((final_score, node))
            
        # Sort by relevance score
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        result = [n[1] for n in scored_nodes[:effective_limit]]
        if len(self._query_cache) >= self._query_cache_max:
            self._query_cache.popitem(last=False)
        self._query_cache[cache_key] = result
        return result
        
    def compress_memories(self):
        """Periodically group raw episodic logs into semantic summaries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Find old episodic memories
            cutoff_time = time.time() - (24 * 3600) # Older than 24 hours
            cursor.execute('SELECT id, data FROM memory_nodes WHERE type = "episodic" AND timestamp < ?', (cutoff_time,))
            rows = cursor.fetchall()
            
            if not rows:
                return
                
            # Very simple compression logic for demonstration
            # In a real system, this would use an LLM to generate the summary
            app_usage = {}
            ids_to_delete = []
            for row in rows:
                node_id, data_str = row[0], row[1]
                data = json.loads(data_str)
                ids_to_delete.append(node_id)
                
                if "app" in data:
                    app = data["app"]
                    duration = data.get("duration", 0)
                    app_usage[app] = app_usage.get(app, 0) + duration
                    
            if app_usage:
                summary_text = "User worked on: " + ", ".join([f"{app} for {dur}s" for app, dur in app_usage.items()])
                summary_node = MemoryNode(
                    id=f"summary_{int(time.time())}",
                    type="semantic",
                    data={"summary": summary_text, "compressed_from": len(ids_to_delete)},
                    relationships=[]
                )
                self.add_node(summary_node)
                
                # Delete compressed nodes
                for node_id in ids_to_delete:
                    cursor.execute('DELETE FROM memory_nodes WHERE id = ?', (node_id,))
                    cursor.execute('DELETE FROM memory_edges WHERE source_id = ? OR target_id = ?', (node_id, node_id))
                
            conn.commit()

    # ---------------- Memory Evolution Utilities ----------------
    def index_experience(self, experience: dict) -> None:
        """Index an experience for similarity search (plan_execution episodic nodes)."""
        plan_steps = experience.get("plan") or experience.get("plan_steps") or []
        text = " ".join(plan_steps) if isinstance(plan_steps, list) else str(plan_steps)
        node = MemoryNode(
            id=f"exp_{int(time.time())}",
            type="episodic",
            data=dict(experience, kind="plan_execution"),
            relationships=[],
            embedding_text=text,
        )
        node.importance = 2.0 if float(experience.get("success_score", 0.0)) >= 0.9 else 1.0
        self.add_node(node)

    def reinforce_memory(self, node_id: str, weight: float = 0.1) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE memory_nodes SET importance = importance + ? WHERE id = ?",
                (weight, node_id),
            )
            conn.commit()

    def decay_memories(self, half_life_hours: float = 24.0) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, importance FROM memory_nodes")
            rows = cursor.fetchall()
            now = time.time()
            for row in rows:
                node_id, ts, imp = row[0], row[1], row[2] or 1.0
                age_hours = (now - (ts or now)) / 3600.0
                decay_factor = 0.5 ** (age_hours / half_life_hours)
                new_imp = max(0.01, (imp or 1.0) * decay_factor)
                cursor.execute(
                    "UPDATE memory_nodes SET importance = ? WHERE id = ?",
                    (new_imp, node_id),
                )
            conn.commit()

    def promote_to_long_term(self, node_id: str) -> None:
        node = self.get_node(node_id)
        if not node:
            return
        node.type = "semantic"
        node.importance = max(node.importance, 2.0)
        self.add_node(node)

    def get_last_active_project(self) -> Optional[str]:
        """Best-effort active project name from recent task / semantic nodes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data FROM memory_nodes ORDER BY timestamp DESC LIMIT 400",
            )
            for (data_str,) in cursor.fetchall():
                try:
                    data = json.loads(data_str)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                for key in ("active_project", "project", "project_name", "repo"):
                    v = data.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()[:500]
        return None

    def get_recent_entities(self, limit: int = 12) -> List[Dict[str, Any]]:
        """Recent nodes as lightweight dicts for predictor / planner hints."""
        out: List[Dict[str, Any]] = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, type, data, timestamp
                FROM memory_nodes
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 64)),),
            )
            for row in cursor.fetchall():
                nid, ntype, data_str, ts = row[0], row[1], row[2], row[3]
                label = nid
                try:
                    data = json.loads(data_str) if data_str else {}
                    if isinstance(data, dict):
                        label = (
                            data.get("title")
                            or data.get("name")
                            or data.get("summary")
                            or data.get("label")
                            or nid
                        )
                except Exception:
                    data = {}
                out.append({
                    "id": nid,
                    "type": ntype,
                    "label": str(label)[:300],
                    "timestamp": ts,
                })
        return out
