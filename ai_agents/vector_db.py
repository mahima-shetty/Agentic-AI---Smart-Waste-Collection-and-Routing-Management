"""
Vector Database Manager - ChromaDB integration for semantic search and contextual memory
Provides embedding generation, storage, and retrieval for AI agents
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import json
import hashlib

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """
    Vector database manager using ChromaDB for semantic search and contextual memory
    Provides embedding generation, storage, and retrieval capabilities for AI agents
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_prefix: str = "waste_management"
    ):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.collection_prefix = collection_prefix
        
        # Initialize ChromaDB client
        self.client = None
        self.embedding_function = None
        self.collections: Dict[str, chromadb.Collection] = {}
        
        # Collection schemas
        self.collection_schemas = {
            "agent_conversations": {
                "description": "Agent conversation history and context",
                "metadata_fields": ["agent_id", "conversation_id", "timestamp", "message_type"]
            },
            "route_patterns": {
                "description": "Historical route optimization patterns and solutions",
                "metadata_fields": ["ward_id", "vehicle_count", "optimization_score", "timestamp"]
            },
            "alert_patterns": {
                "description": "Alert generation patterns and historical responses",
                "metadata_fields": ["bin_id", "alert_type", "severity", "resolution_time", "timestamp"]
            },
            "analytics_insights": {
                "description": "Generated analytics insights and patterns",
                "metadata_fields": ["analysis_type", "data_sources", "insight_type", "timestamp"]
            },
            "bin_behaviors": {
                "description": "Bin fill patterns and behavioral data",
                "metadata_fields": ["bin_id", "ward_id", "pattern_type", "seasonal_factor", "timestamp"]
            },
            "system_knowledge": {
                "description": "General system knowledge and learned patterns",
                "metadata_fields": ["knowledge_type", "source_agent", "confidence_score", "timestamp"]
            }
        }
        
        # Statistics
        self.stats = {
            "embeddings_generated": 0,
            "documents_stored": 0,
            "searches_performed": 0,
            "collections_created": 0
        }
        
        logger.info("🗄️ Vector Database Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the ChromaDB client and collections"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            
            # Create or get collections
            await self._initialize_collections()
            
            logger.info("✅ Vector Database Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vector Database Manager: {e}")
            return False
    
    async def _initialize_collections(self):
        """Initialize all predefined collections"""
        for collection_name, schema in self.collection_schemas.items():
            try:
                full_name = f"{self.collection_prefix}_{collection_name}"
                
                # Get or create collection
                collection = self.client.get_or_create_collection(
                    name=full_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": schema["description"]}
                )
                
                self.collections[collection_name] = collection
                self.stats["collections_created"] += 1
                
                logger.info(f"📚 Collection initialized: {collection_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize collection {collection_name}: {e}")
    
    async def store_document(
        self,
        collection_name: str,
        document_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> bool:
        """Store a document with its embedding in the specified collection"""
        try:
            if collection_name not in self.collections:
                logger.error(f"❌ Collection not found: {collection_name}")
                return False
            
            collection = self.collections[collection_name]
            
            # Generate embedding if not provided
            if embedding is None:
                embedding = await self._generate_embedding(content)
            
            # Add timestamp to metadata
            metadata["stored_at"] = datetime.now().isoformat()
            metadata["content_hash"] = hashlib.md5(content.encode()).hexdigest()
            
            # Store document
            collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[document_id]
            )
            
            self.stats["documents_stored"] += 1
            logger.info(f"📄 Document stored: {document_id} in {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store document {document_id}: {e}")
            return False
    
    async def store_conversation(
        self,
        agent_id: str,
        conversation_id: str,
        message: str,
        message_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store agent conversation for contextual memory"""
        
        document_id = f"{agent_id}_{conversation_id}_{datetime.now().timestamp()}"
        
        metadata = {
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "message_type": message_type,
            "timestamp": datetime.now().isoformat()
        }
        
        if context:
            metadata.update(context)
        
        return await self.store_document(
            collection_name="agent_conversations",
            document_id=document_id,
            content=message,
            metadata=metadata
        )
    
    async def store_route_pattern(
        self,
        ward_id: int,
        route_data: Dict[str, Any],
        optimization_score: float,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store route optimization pattern for future reference"""
        
        document_id = f"route_{ward_id}_{datetime.now().timestamp()}"
        content = json.dumps(route_data, indent=2)
        
        metadata = {
            "ward_id": ward_id,
            "vehicle_count": len(route_data.get("vehicles", [])),
            "optimization_score": optimization_score,
            "timestamp": datetime.now().isoformat(),
            "pattern_type": "route_optimization"
        }
        
        if context:
            metadata.update(context)
        
        return await self.store_document(
            collection_name="route_patterns",
            document_id=document_id,
            content=content,
            metadata=metadata
        )
    
    async def store_alert_pattern(
        self,
        bin_id: str,
        alert_data: Dict[str, Any],
        resolution_time: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store alert pattern for learning and improvement"""
        
        document_id = f"alert_{bin_id}_{datetime.now().timestamp()}"
        content = json.dumps(alert_data, indent=2)
        
        metadata = {
            "bin_id": bin_id,
            "alert_type": alert_data.get("alert_type", "unknown"),
            "severity": alert_data.get("severity", "medium"),
            "timestamp": datetime.now().isoformat(),
            "pattern_type": "alert_generation"
        }
        
        if resolution_time:
            metadata["resolution_time"] = resolution_time
        
        if context:
            metadata.update(context)
        
        return await self.store_document(
            collection_name="alert_patterns",
            document_id=document_id,
            content=content,
            metadata=metadata
        )
    
    async def store_analytics_insight(
        self,
        analysis_type: str,
        insight_data: Dict[str, Any],
        data_sources: List[str],
        confidence_score: float = 0.8,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store analytics insight for future reference"""
        
        document_id = f"insight_{analysis_type}_{datetime.now().timestamp()}"
        content = json.dumps(insight_data, indent=2)
        
        metadata = {
            "analysis_type": analysis_type,
            "data_sources": ",".join(data_sources),
            "insight_type": insight_data.get("type", "general"),
            "confidence_score": confidence_score,
            "timestamp": datetime.now().isoformat()
        }
        
        if context:
            metadata.update(context)
        
        return await self.store_document(
            collection_name="analytics_insights",
            document_id=document_id,
            content=content,
            metadata=metadata
        )
    
    async def semantic_search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include_distances: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform semantic search in the specified collection"""
        try:
            if collection_name not in self.collections:
                logger.error(f"❌ Collection not found: {collection_name}")
                return []
            
            collection = self.collections[collection_name]
            
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"] if include_distances else ["documents", "metadatas"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                }
                
                if include_distances and "distances" in results:
                    result["distance"] = results["distances"][0][i]
                    result["similarity"] = 1 - results["distances"][0][i]  # Convert distance to similarity
                
                formatted_results.append(result)
            
            self.stats["searches_performed"] += 1
            logger.info(f"🔍 Semantic search completed: {len(formatted_results)} results for '{query[:50]}...'")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    async def find_similar_routes(
        self,
        ward_id: int,
        vehicle_count: int,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Find similar route optimization patterns"""
        
        query = f"route optimization ward {ward_id} vehicles {vehicle_count}"
        
        where_filter = {
            "ward_id": ward_id,
            "vehicle_count": vehicle_count
        }
        
        return await self.semantic_search(
            collection_name="route_patterns",
            query=query,
            n_results=n_results,
            where=where_filter
        )
    
    async def find_similar_alerts(
        self,
        bin_id: str,
        alert_type: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar alert patterns for learning"""
        
        query = f"alert {alert_type} bin {bin_id}"
        
        where_filter = {
            "alert_type": alert_type
        }
        
        return await self.semantic_search(
            collection_name="alert_patterns",
            query=query,
            n_results=n_results,
            where=where_filter
        )
    
    async def find_relevant_insights(
        self,
        analysis_type: str,
        query_context: str,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Find relevant analytics insights"""
        
        query = f"{analysis_type} analysis {query_context}"
        
        where_filter = {
            "analysis_type": analysis_type
        }
        
        return await self.semantic_search(
            collection_name="analytics_insights",
            query=query,
            n_results=n_results,
            where=where_filter
        )
    
    async def get_agent_context(
        self,
        agent_id: str,
        conversation_id: Optional[str] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Get contextual conversation history for an agent"""
        
        where_filter = {"agent_id": agent_id}
        if conversation_id:
            where_filter["conversation_id"] = conversation_id
        
        query = f"agent {agent_id} conversation history"
        
        return await self.semantic_search(
            collection_name="agent_conversations",
            query=query,
            n_results=n_results,
            where=where_filter
        )
    
    async def store_system_knowledge(
        self,
        knowledge_type: str,
        content: str,
        source_agent: str,
        confidence_score: float = 0.8,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store general system knowledge"""
        
        document_id = f"knowledge_{knowledge_type}_{datetime.now().timestamp()}"
        
        metadata = {
            "knowledge_type": knowledge_type,
            "source_agent": source_agent,
            "confidence_score": confidence_score,
            "timestamp": datetime.now().isoformat()
        }
        
        if context:
            metadata.update(context)
        
        return await self.store_document(
            collection_name="system_knowledge",
            document_id=document_id,
            content=content,
            metadata=metadata
        )
    
    async def query_system_knowledge(
        self,
        query: str,
        knowledge_type: Optional[str] = None,
        min_confidence: float = 0.5,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Query system knowledge base"""
        
        where_filter = {
            "confidence_score": {"$gte": min_confidence}
        }
        
        if knowledge_type:
            where_filter["knowledge_type"] = knowledge_type
        
        return await self.semantic_search(
            collection_name="system_knowledge",
            query=query,
            n_results=n_results,
            where=where_filter
        )
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured model"""
        try:
            # Use ChromaDB's embedding function
            embeddings = self.embedding_function([text])
            self.stats["embeddings_generated"] += 1
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
    
    async def update_document(
        self,
        collection_name: str,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """Update an existing document"""
        try:
            if collection_name not in self.collections:
                logger.error(f"❌ Collection not found: {collection_name}")
                return False
            
            collection = self.collections[collection_name]
            
            update_data = {"ids": [document_id]}
            
            if content is not None:
                update_data["documents"] = [content]
                if embedding is None:
                    embedding = await self._generate_embedding(content)
            
            if embedding is not None:
                update_data["embeddings"] = [embedding]
            
            if metadata is not None:
                metadata["updated_at"] = datetime.now().isoformat()
                update_data["metadatas"] = [metadata]
            
            collection.update(**update_data)
            
            logger.info(f"📝 Document updated: {document_id} in {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update document {document_id}: {e}")
            return False
    
    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete a document from the collection"""
        try:
            if collection_name not in self.collections:
                logger.error(f"❌ Collection not found: {collection_name}")
                return False
            
            collection = self.collections[collection_name]
            collection.delete(ids=[document_id])
            
            logger.info(f"🗑️ Document deleted: {document_id} from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document {document_id}: {e}")
            return False
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a specific collection"""
        try:
            if collection_name not in self.collections:
                return {"error": f"Collection not found: {collection_name}"}
            
            collection = self.collections[collection_name]
            count = collection.count()
            
            return {
                "collection_name": collection_name,
                "document_count": count,
                "description": self.collection_schemas.get(collection_name, {}).get("description", ""),
                "metadata_fields": self.collection_schemas.get(collection_name, {}).get("metadata_fields", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics"""
        
        collection_stats = {}
        total_documents = 0
        
        for collection_name in self.collections:
            stats = await self.get_collection_stats(collection_name)
            collection_stats[collection_name] = stats
            if "document_count" in stats:
                total_documents += stats["document_count"]
        
        return {
            "total_collections": len(self.collections),
            "total_documents": total_documents,
            "embedding_model": self.embedding_model,
            "persist_directory": self.persist_directory,
            "collection_stats": collection_stats,
            "operation_stats": self.stats,
            "timestamp": datetime.now().isoformat()
        }
    
    async def cleanup_old_documents(
        self,
        collection_name: str,
        days_old: int = 30,
        max_documents: Optional[int] = None
    ) -> int:
        """Clean up old documents from a collection"""
        try:
            if collection_name not in self.collections:
                logger.error(f"❌ Collection not found: {collection_name}")
                return 0
            
            collection = self.collections[collection_name]
            
            # Get all documents with metadata
            results = collection.get(include=["metadatas"])
            
            # Find documents to delete
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            documents_to_delete = []
            
            for i, metadata in enumerate(results["metadatas"]):
                stored_at = metadata.get("stored_at")
                if stored_at:
                    try:
                        doc_timestamp = datetime.fromisoformat(stored_at).timestamp()
                        if doc_timestamp < cutoff_date:
                            documents_to_delete.append(results["ids"][i])
                    except:
                        continue
            
            # Limit by max_documents if specified
            if max_documents and len(documents_to_delete) > max_documents:
                documents_to_delete = documents_to_delete[:max_documents]
            
            # Delete documents
            if documents_to_delete:
                collection.delete(ids=documents_to_delete)
                logger.info(f"🧹 Cleaned up {len(documents_to_delete)} old documents from {collection_name}")
            
            return len(documents_to_delete)
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup collection {collection_name}: {e}")
            return 0
    
    async def backup_collection(self, collection_name: str, backup_path: str) -> bool:
        """Backup a collection to a file"""
        try:
            if collection_name not in self.collections:
                logger.error(f"❌ Collection not found: {collection_name}")
                return False
            
            collection = self.collections[collection_name]
            
            # Get all data from collection
            results = collection.get(include=["documents", "metadatas", "embeddings"])
            
            backup_data = {
                "collection_name": collection_name,
                "backup_timestamp": datetime.now().isoformat(),
                "document_count": len(results["ids"]),
                "data": results
            }
            
            # Save to file
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"💾 Collection backed up: {collection_name} to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to backup collection {collection_name}: {e}")
            return False

logger.info("🗄️ Vector Database Manager loaded successfully")