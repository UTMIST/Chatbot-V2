from typing import List, Optional, Set
from llama_index.core.schema import NodeWithScore
from dataclasses import dataclass

"""NOTE: Class is not used and is incomplete ATM."""

@dataclass
class FilterConfig:
    """Configuration for metadata filtering."""
    case_sensitive: bool = False
    metadata_fields: Optional[List[str]] = None  # If None, search all metadata fields
    require_all_words: bool = False  # If True, all words must match instead of any

class Filter:
    """Filter nodes based on metadata content matching."""
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize the filter with configuration.
        
        Args:
            config (Optional[FilterConfig]): Configuration for filtering
        """
        self.config = config or FilterConfig()

    def filter_nodes(
        self,
        nodes: List[NodeWithScore],
        words: List[str]
    ) -> List[NodeWithScore]:
        """
        Filter nodes based on whether their metadata contains any of the given words.
        
        Args:
            nodes (List[NodeWithScore]): List of nodes to filter
            words (List[str]): List of words to match against
            
        Returns:
            List[NodeWithScore]: Filtered list of nodes
        """
        if not words:
            return nodes

        if not self.config.case_sensitive:
            words = [word.lower() for word in words]
        words_set = set(words)
        
        filtered_nodes = []
        
        for node in nodes:
            metadata = node.node.metadata
            
            if self.config.metadata_fields:
                metadata = {
                    k: v for k, v in metadata.items() 
                    if k in self.config.metadata_fields
                }
            
            metadata_text = " ".join(str(value) for value in metadata.values())
            if not self.config.case_sensitive:
                metadata_text = metadata_text.lower()
            
            # Check if any/all words are in the metadata
            if self.config.require_all_words:
                if all(word in metadata_text for word in words_set):
                    filtered_nodes.append(node)
            else:
                if any(word in metadata_text for word in words_set):
                    filtered_nodes.append(node)
                    
        return filtered_nodes

    def get_matching_metadata_fields(
        self,
        node: NodeWithScore,
        words: List[str]
    ) -> Set[str]:
        """
        Get the metadata fields that contain matches for the given words.
        
        Args:
            node (NodeWithScore): Node to check
            words (List[str]): List of words to match against
            
        Returns:
            Set[str]: Set of metadata field names containing matches
        """
        if not self.config.case_sensitive:
            words = [word.lower() for word in words]
        words_set = set(words)
        
        matching_fields = set()
        metadata = node.node.metadata
        
        for field, value in metadata.items():
            if self.config.metadata_fields and field not in self.config.metadata_fields:
                continue
                
            field_text = str(value)
            if not self.config.case_sensitive:
                field_text = field_text.lower()
                
            if any(word in field_text for word in words_set):
                matching_fields.add(field)
                
        return matching_fields

