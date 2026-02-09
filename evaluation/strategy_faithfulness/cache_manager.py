"""
Cache Manager for Strategy Faithfulness Evaluation

Manages caching of intermediate results to avoid redundant computations
when evaluating different tool configurations.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


class CacheManager:
    """
    Manages caching of tool configuration evaluation results.
    
    Cache key: tuple of tool mask (e.g., (0, 1, 1))
    Cache value: dict with strategy, explanation, faithfulness_score
    """
    
    def __init__(self, cache_dir: str):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for current session
        self.memory_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        
        # Track cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0
        }
    
    def _get_cache_key(self, config: Tuple[int, ...], question_id: str) -> str:
        """Generate unique cache key from config and question"""
        config_str = "_".join(map(str, config))
        return f"{question_id}_{config_str}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache entry"""
        return self.cache_dir / f"cache_{cache_key}.json"
    
    def get(
        self, 
        config: Tuple[int, ...], 
        question_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a tool configuration.
        
        Args:
            config: Tool mask tuple (e.g., (0, 1, 1))
            question_id: Question identifier
            
        Returns:
            Cached result dict or None if not found
        """
        cache_key = self._get_cache_key(config, question_id)
        
        # Check memory cache first
        if config in self.memory_cache:
            self.stats["hits"] += 1
            return self.memory_cache[config]
        
        # Check file cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cache_entry = json.load(f)
                # Extract the actual result from cache entry
                # Cache entry format: {"config": ..., "question_id": ..., "timestamp": ..., "result": actual_result}
                actual_result = cache_entry.get('result', cache_entry)
                # Store in memory cache for faster access
                self.memory_cache[config] = actual_result
                self.stats["hits"] += 1
                return actual_result
            except (json.JSONDecodeError, IOError):
                pass
        
        self.stats["misses"] += 1
        return None
    
    def set(
        self, 
        config: Tuple[int, ...], 
        question_id: str,
        result: Dict[str, Any],
        persist: bool = True
    ):
        """
        Cache result for a tool configuration.
        
        Args:
            config: Tool mask tuple
            question_id: Question identifier
            result: Result dict to cache
            persist: Whether to persist to disk
        """
        cache_key = self._get_cache_key(config, question_id)
        
        # Store in memory cache
        self.memory_cache[config] = result
        
        # Persist to disk if requested
        if persist:
            cache_path = self._get_cache_path(cache_key)
            try:
                # Add metadata
                cache_entry = {
                    "config": list(config),
                    "question_id": question_id,
                    "timestamp": datetime.now().isoformat(),
                    "result": result
                }
                with open(cache_path, 'w') as f:
                    json.dump(cache_entry, f, indent=2, default=str)
                self.stats["saves"] += 1
            except IOError as e:
                print(f"Warning: Failed to persist cache: {e}")
    
    def has(self, config: Tuple[int, ...], question_id: str) -> bool:
        """Check if config is cached"""
        return self.get(config, question_id) is not None
    
    def clear_memory(self):
        """Clear in-memory cache"""
        self.memory_cache.clear()
    
    def clear_all(self):
        """Clear both memory and disk cache"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("cache_*.json"):
            try:
                cache_file.unlink()
            except IOError:
                pass
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            **self.stats,
            "memory_size": len(self.memory_cache),
            "disk_size": len(list(self.cache_dir.glob("cache_*.json")))
        }
    
    def get_all_cached_configs(self, question_id: str) -> List[Tuple[int, ...]]:
        """Get all cached configurations for a question"""
        configs = []
        prefix = f"cache_{question_id}_"
        
        for cache_file in self.cache_dir.glob(f"{prefix}*.json"):
            # Extract config from filename
            filename = cache_file.stem
            config_str = filename.replace(prefix.replace("cache_", ""), "")
            if config_str:
                try:
                    config = tuple(map(int, config_str.split("_")))
                    configs.append(config)
                except ValueError:
                    pass
        
        return configs
