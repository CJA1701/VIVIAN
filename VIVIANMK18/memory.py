import json
import os
from datetime import datetime
import pytz
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manage conversation memory with context awareness"""
    
    def __init__(self, config):
        self.config = config
        self.memory_file = config.files['memory_file']
        self.memory_limit = config.files['memory_limit']
        self.timezone = pytz.timezone(config.timezone)
        
        # In-memory conversation context (for current session)
        self.session_context = []
        self.max_session_context = 5  # Keep last 5 exchanges in active context
    
    def add_interaction(self, user_prompt: str, reply: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Add a conversation interaction to memory
        
        Args:
            user_prompt: What the user said
            reply: What VIVIAN responded
            metadata: Optional dict with extra info (function_called, music_playing, etc.)
        
        Returns:
            Updated memory list
        """
        timestamp = datetime.now(self.timezone)
        
        # Create structured entry
        entry = {
            "timestamp": timestamp.isoformat(),
            "user": user_prompt,
            "assistant": reply,
            "metadata": metadata or {}
        }
        
        # Add to session context (for immediate use)
        self.session_context.append(entry)
        if len(self.session_context) > self.max_session_context:
            self.session_context.pop(0)
        
        # Load existing memory
        memory = self._load_memory_structured()
        memory.append(entry)
        
        # Keep only last N entries
        memory = memory[-self.memory_limit:]
        
        # Save to file
        self._save_memory_structured(memory)
        
        logger.info(f"Memory updated: {len(memory)} total entries, {len(self.session_context)} in session")
        return memory
    
    def get_session_context(self) -> List[Dict]:
        """Get recent conversation context for GPT (last 5 exchanges)"""
        return self.session_context
    
    def get_recent_memory_summary(self, count: int = 10) -> str:
        """
        Get a concise summary of recent conversations
        More efficient than dumping all 30 conversations
        """
        memory = self._load_memory_structured()
        recent = memory[-count:] if len(memory) > count else memory
        
        if not recent:
            return "No previous conversations."
        
        # Format as concise summary
        summary_lines = []
        for entry in recent:
            # Parse timestamp for relative time. Entries converted from the
            # old string format carry non-ISO timestamps — show them as-is.
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                time_str = ts.strftime("%I:%M %p")
            except (ValueError, TypeError):
                time_str = str(entry.get("timestamp", "?"))[:20]
            
            # Truncate long messages
            user_msg = entry["user"][:50] + "..." if len(entry["user"]) > 50 else entry["user"]
            asst_msg = entry["assistant"][:50] + "..." if len(entry["assistant"]) > 50 else entry["assistant"]
            
            summary_lines.append(f"[{time_str}] User: {user_msg} | VIVIAN: {asst_msg}")
        
        return "\n".join(summary_lines)
    
    def get_relevant_context(self, current_query: str, max_items: int = 3) -> List[Dict]:
        """
        Get contextually relevant past conversations (simple keyword matching)
        More sophisticated than dumping all memory
        """
        memory = self._load_memory_structured()
        
        if not memory:
            return []
        
        # Simple keyword relevance scoring
        query_keywords = set(current_query.lower().split())
        scored_memories = []
        
        for entry in memory[-20:]:  # Only check recent 20
            # Check both user and assistant messages
            text = (entry["user"] + " " + entry["assistant"]).lower()
            matching_keywords = query_keywords.intersection(set(text.split()))
            score = len(matching_keywords)
            
            if score > 0:
                scored_memories.append((score, entry))
        
        # Sort by relevance and return top N
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [entry for score, entry in scored_memories[:max_items]]
    
    def _load_memory_structured(self) -> List[Dict]:
        """Load memory as structured data"""
        if not os.path.exists(self.memory_file):
            return []
        
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                
            # Handle both old format (strings) and new format (dicts)
            if data and isinstance(data[0], str):
                # Old format - convert to new format
                logger.info("Converting old memory format to new structured format")
                return self._convert_old_memory(data)
            
            return data
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading memory: {e}")
            return []
    
    def _save_memory_structured(self, memory: List[Dict]) -> None:
        """Save memory as structured JSON.

        Written to a temp sibling then atomically replaced: opening the real
        file 'w' truncates it first, and the car cuts Pi power at ignition-off,
        so a mid-write shutoff left half a JSON file behind — which makes
        _load_memory_structured() throw and silently drop ALL history.
        """
        tmp_file = self.memory_file + ".tmp"
        try:
            with open(tmp_file, 'w') as f:
                json.dump(memory, f, indent=2)
            os.replace(tmp_file, self.memory_file)
        except IOError as e:
            logger.error(f"Error saving memory: {e}")
    
    def _convert_old_memory(self, old_memory: List[str]) -> List[Dict]:
        """Convert old string-based memory to new structured format"""
        new_memory = []
        
        for entry in old_memory:
            try:
                # Parse old format: "2026-01-14 21:58:57 EST - User: ... VIVIAN: ..."
                parts = entry.split(" - ", 1)
                if len(parts) != 2:
                    continue
                
                timestamp_str = parts[0]
                conversation = parts[1]
                
                # Split user and assistant
                if " VIVIAN: " in conversation:
                    user_part, asst_part = conversation.split(" VIVIAN: ", 1)
                    user_msg = user_part.replace("User: ", "").strip()
                    asst_msg = asst_part.strip()
                    
                    new_memory.append({
                        "timestamp": timestamp_str,
                        "user": user_msg,
                        "assistant": asst_msg,
                        "metadata": {}
                    })
            except Exception as e:
                logger.warning(f"Could not convert old memory entry: {e}")
                continue
        
        return new_memory
    
    def get_music_history(self, count: int = 5) -> List[Dict]:
        """Get recent music-related interactions"""
        memory = self._load_memory_structured()
        music_keywords = ["play", "song", "music", "album", "artist", "skip", "pause"]
        
        music_entries = []
        for entry in reversed(memory):
            text = (entry["user"] + " " + entry["assistant"]).lower()
            if any(keyword in text for keyword in music_keywords):
                music_entries.append(entry)
                if len(music_entries) >= count:
                    break
        
        return list(reversed(music_entries))
    
    def clear_session_context(self):
        """Clear the current session context (e.g., after user says goodbye)"""
        self.session_context = []
        logger.info("Session context cleared")
    
    def clear_memory(self) -> None:
        """Clear all conversation history"""
        self._save_memory_structured([])
        self.session_context = []
        logger.info("All memory cleared")
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about memory usage"""
        memory = self._load_memory_structured()
        
        if not memory:
            return {
                "total_entries": 0,
                "session_entries": 0,
                "oldest_entry": None,
                "newest_entry": None
            }
        
        return {
            "total_entries": len(memory),
            "session_entries": len(self.session_context),
            "oldest_entry": memory[0]["timestamp"],
            "newest_entry": memory[-1]["timestamp"],
            "storage_size_kb": os.path.getsize(self.memory_file) / 1024 if os.path.exists(self.memory_file) else 0
        }
