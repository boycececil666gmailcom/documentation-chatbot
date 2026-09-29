# region Driver
import json
import re
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from .llm_client import llm


@lru_cache(maxsize=1)
def get_driver():
    """Lazily initializes and caches Neo4j GraphDatabase driver."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception:
        return None


# endregion


# region Helpers
def sanitize_rel_type(rel_type: str) -> str:
    """Sanitizes relationship type to conform to Cypher naming rules."""
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", rel_type).upper().strip("_")
    return sanitized or "RELATED_TO"


def clean_json_response(content: Any) -> str:
    """Strips markdown code fences from LLM response text."""
    if isinstance(content, list):
        content = "".join(
            part if isinstance(part, str) else part.get("text", "") for part in content
        )
    text = str(content).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# endregion


# region Graph Extractors
def extract_entities_and_relations(text: str) -> dict[str, list[dict[str, Any]]]:
    """Extracts entities and relationships from text using LLM."""
    system_prompt = (
        "You are an expert knowledge graph extractor. Extract entities and relationships "
        "from the provided text. Output ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "entities": [{"name": "Entity Name", "type": "Entity Type", "description": "Brief description"}],\n'
        '  "relationships": [{"source": "Source Name", "type": "REL_TYPE", "target": "Target Name"}]\n'
        "}\n"
        "Do not include markdown fences. Output raw JSON only."
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Text to extract from:\n{text}"),
            ]
        )
        data = json.loads(clean_json_response(response.content))
        return {
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }
    except Exception:
        return {"entities": [], "relationships": []}


def extract_query_entities(query: str) -> list[str]:
    """Identifies entity names mentioned in a user search query."""
    system_prompt = (
        "You are an entity extractor. Identify and list the main entities in the user query. "
        'Output ONLY a valid JSON array of strings, e.g. ["Supernova", "Premium Plan"]. '
        "Output raw JSON only."
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: '{query}'"),
            ]
        )
        entities = json.loads(clean_json_response(response.content))
        return [str(item) for item in entities] if isinstance(entities, list) else []
    except Exception:
        return []


# endregion


# region Cypher Operations
def add_graph_relations(
    entities: list[dict[str, Any]], relationships: list[dict[str, Any]]
) -> None:
    """Saves extracted entities and relationships into Neo4j."""
    driver = get_driver()
    if not driver:
        return

    try:
        with driver.session() as session:
            for ent in entities:
                name = ent.get("name")
                if name:
                    session.run(
                        "MERGE (e:Entity {name: $name}) "
                        "ON CREATE SET e.type = $type, e.description = $description "
                        "ON MATCH SET e.description = CASE WHEN e.description CONTAINS $description "
                        "THEN e.description ELSE e.description + '; ' + $description END",
                        name=name,
                        type=ent.get("type", "Unknown"),
                        description=ent.get("description", ""),
                    )

            for rel in relationships:
                source, target = rel.get("source"), rel.get("target")
                rel_type = sanitize_rel_type(rel.get("type", "RELATED_TO"))
                if source and target:
                    session.run("MERGE (:Entity {name: $name})", name=source)
                    session.run("MERGE (:Entity {name: $name})", name=target)
                    session.run(
                        f"MATCH (s:Entity {{name: $source}}), (t:Entity {{name: $target}}) "
                        f"MERGE (s)-[:{rel_type}]->(t)",
                        source=source,
                        target=target,
                    )
    except Exception:
        pass


def retrieve_graph_relations(query_entities: list[str]) -> str:
    """Queries Neo4j for listed entities and their 1-hop relationships."""
    driver = get_driver()
    if not driver or not query_entities:
        return ""

    context_parts = []
    try:
        with driver.session() as session:
            for entity_name in query_entities:
                result = session.run(
                    "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($name) OR toLower($name) CONTAINS toLower(e.name) "
                    "OPTIONAL MATCH (e)-[r]-(neighbor:Entity) "
                    "RETURN e.name as name, e.type as type, e.description as description, "
                    "       type(r) as rel_type, neighbor.name as neighbor_name, "
                    "       startNode(r) = e as is_source",
                    name=entity_name,
                )
                records = list(result)
                if not records:
                    continue

                first = records[0]
                relations_info = [
                    f"- ({first['name']}) -[{r['rel_type']}]-> ({r['neighbor_name']})"
                    if r.get("is_source")
                    else f"- ({r['neighbor_name']}) -[{r['rel_type']}]-> ({first['name']})"
                    for r in records
                    if r.get("rel_type") and r.get("neighbor_name")
                ]

                part = f"Entity: {first['name']} (Type: {first['type']}, Description: {first['description']})"
                if relations_info:
                    part += "\nRelationships:\n" + "\n".join(relations_info)
                context_parts.append(part)

        return "\n\n".join(context_parts)
    except Exception:
        return ""


def ingest_graph_document(text: str) -> int:
    """Extracts entities/relations from text and persists to Neo4j."""
    extracted = extract_entities_and_relations(text)
    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])
    if entities or relationships:
        add_graph_relations(entities, relationships)
        return len(entities) + len(relationships)
    return 0


# endregion
