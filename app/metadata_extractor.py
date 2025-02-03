"""Metadata extraction functionality."""
import logging
from typing import Dict, Optional
from datetime import datetime

from app.keboola_client import KeboolaClient

def extract_metadata(client: KeboolaClient, force_full: bool = False) -> Dict:
    """Extract metadata from Keboola using the provided client.
    
    Args:
        client: Initialized KeboolaClient instance
        force_full: Whether to force a full extraction instead of incremental
        
    Returns:
        Dictionary containing extracted metadata with enriched information:
        - Table statistics and quality metrics
        - Column data profiles and statistics
        - Transformation dependencies and lineage
        - Access patterns and usage metrics
        - Data quality scores and metrics
        - Relationship mappings
    """
    logging.info("Starting metadata extraction (force_full=%s)", force_full)
    
    try:
        # Extract base metadata
        metadata = client.extract_metadata(force_full=force_full)
        
        # Initialize relationship tracking
        relationships = {
            "column_to_column": [],
            "table_to_table": [],
            "config_dependencies": []
        }
        
        # Enrich table metadata with quality metrics and statistics
        for bucket_id, tables in metadata["tables"].items():
            for table in tables:
                table_id = table.get("id")
                if table_id and table_id in metadata["table_details"]:
                    table_detail = metadata["table_details"][table_id]
                    
                    # Add enhanced table statistics
                    table_detail["statistics"] = {
                        "row_count": table_detail.get("rowsCount", 0),
                        "size_bytes": table_detail.get("dataSizeBytes", 0),
                        "last_import_date": table_detail.get("lastImportDate"),
                        "last_change_date": table_detail.get("lastChangeDate"),
                        "avg_import_duration": table_detail.get("avgImportDurationSeconds", 0),
                        "last_import_status": table_detail.get("lastImportStatus"),
                        "data_retention_days": table_detail.get("dataRetentionDays", 0),
                        "last_access_date": table_detail.get("lastAccessDate")
                    }
                    
                    # Calculate data freshness score (0-100)
                    if table_detail["statistics"]["last_change_date"]:
                        last_change = datetime.fromisoformat(table_detail["statistics"]["last_change_date"].replace("Z", "+00:00"))
                        days_since_change = (datetime.now() - last_change).days
                        freshness_score = max(0, 100 - (days_since_change * 2))  # Decrease score by 2 points per day
                        table_detail["statistics"]["freshness_score"] = freshness_score
                    
                    # Add enhanced column statistics
                    if "columns" in table_detail:
                        for column in table_detail["columns"]:
                            column_stats = {
                                "null_count": column.get("nullCount", 0),
                                "unique_count": column.get("uniqueCount", 0),
                                "data_type": column.get("type", "unknown"),
                                "format": column.get("format", ""),
                                "constraints": column.get("constraints", []),
                                "min_value": column.get("minValue"),
                                "max_value": column.get("maxValue"),
                                "avg_value": column.get("avgValue"),
                                "median_value": column.get("medianValue"),
                                "sample_values": column.get("sampleValues", []),
                                "pattern_match_rate": column.get("patternMatchRate", 0),
                                "distinct_count": column.get("distinctCount", 0)
                            }
                            
                            # Calculate column quality score (0-100)
                            if column_stats["null_count"] is not None and table_detail["statistics"]["row_count"] > 0:
                                null_rate = column_stats["null_count"] / table_detail["statistics"]["row_count"]
                                unique_rate = column_stats["unique_count"] / table_detail["statistics"]["row_count"] if column_stats["unique_count"] else 0
                                pattern_match_rate = column_stats["pattern_match_rate"] / 100 if column_stats["pattern_match_rate"] else 1
                                
                                quality_score = (
                                    (1 - null_rate) * 40 +  # Penalize null values
                                    pattern_match_rate * 40 +  # Reward pattern matches
                                    min(unique_rate * 100, 20)  # Reward uniqueness up to 20 points
                                )
                                column_stats["quality_score"] = max(0, min(100, quality_score))
                            
                            column["statistics"] = column_stats
                            
                            # Track potential column relationships
                            if column.get("name") and column.get("type"):
                                for other_bucket_id, other_tables in metadata["tables"].items():
                                    for other_table in other_tables:
                                        other_table_id = other_table.get("id")
                                        if other_table_id == table_id:
                                            continue
                                            
                                        if other_table_id in metadata["table_details"]:
                                            other_table_detail = metadata["table_details"][other_table_id]
                                            for other_column in other_table_detail.get("columns", []):
                                                if (
                                                    other_column.get("name") == column.get("name")
                                                    and other_column.get("type") == column.get("type")
                                                ):
                                                    relationships["column_to_column"].append({
                                                        "source_table": table_id,
                                                        "source_column": column["name"],
                                                        "target_table": other_table_id,
                                                        "target_column": other_column["name"],
                                                        "relationship_type": "same_name_and_type",
                                                        "confidence": 0.8
                                                    })
                    
                    # Add enhanced transformation dependencies
                    table_detail["dependencies"] = {
                        "source_tables": table_detail.get("sourceTableIds", []),
                        "target_tables": table_detail.get("targetTableIds", []),
                        "transformations": table_detail.get("transformationIds", []),
                        "upstream_tables": [],  # Tables this table depends on
                        "downstream_tables": [],  # Tables that depend on this table
                        "transformation_flow": []  # Sequence of transformations
                    }
                    
                    # Track table-to-table relationships
                    for source_id in table_detail["dependencies"]["source_tables"]:
                        relationships["table_to_table"].append({
                            "source_table": source_id,
                            "target_table": table_id,
                            "relationship_type": "data_flow",
                            "confidence": 1.0
                        })
                    
                    # Update the enriched metadata
                    metadata["table_details"][table_id] = table_detail
        
        # Add relationships to metadata
        metadata["relationships"] = relationships
        
        return metadata
        
    except Exception as e:
        logging.error(f"Error during metadata extraction: {str(e)}")
        # Return empty metadata structure on error
        return {
            "buckets": [],
            "tables": {},
            "table_details": {},
            "configurations": {},
            "config_rows": {},
            "columns": {},
            "relationships": {
                "column_to_column": [],
                "table_to_table": [],
                "config_dependencies": []
            }
        } 