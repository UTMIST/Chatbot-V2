from qdrant_client import QdrantClient
from typing import Optional

def get_qdrant_collection_stats(
    client: QdrantClient,
    collection_name: str,
) -> Optional[dict]:
    """
    Display statistics for a Qdrant collection.
    """
    try:
        collection_info = client.get_collection(collection_name)
        
        if collection_info is None:
            print(f"Collection {collection_name} not found")
            return None
            
        print(f"Collection: {collection_name}")
        print(f"Vectors count: {collection_info.vectors_count}")
        
        return collection_info.model_dump()
        
    except Exception as e:
        print(f"Error getting collection stats: {str(e)}")
        return None
