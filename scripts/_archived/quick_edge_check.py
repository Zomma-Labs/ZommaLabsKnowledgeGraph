#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.util.services import get_services

neo4j = get_services().neo4j

# Check total edges with fact_id
total = neo4j.query('''
    MATCH ()-[r]->()
    WHERE r.fact_id IS NOT NULL
    RETURN count(*) as cnt
''')
print(f'Total edges with fact_id: {total[0]["cnt"]}')

# Check Subject->Chunk edges
subject_edges = neo4j.query('''
    MATCH (e)-[r]->(c:EpisodicNode {group_id: "default"})
    WHERE (e:EntityNode OR e:TopicNode)
      AND r.fact_id IS NOT NULL
    RETURN count(*) as cnt
''')
print(f'Subject->Chunk edges: {subject_edges[0]["cnt"]}')

# Check Chunk->Object edges (_TARGET)
object_edges = neo4j.query('''
    MATCH (c:EpisodicNode {group_id: "default"})-[r]->(e)
    WHERE (e:EntityNode OR e:TopicNode)
      AND r.fact_id IS NOT NULL
    RETURN count(*) as cnt
''')
print(f'Chunk->Object edges: {object_edges[0]["cnt"]}')

# Sample a random complete pattern
sample = neo4j.query('''
    MATCH (subj)-[r1]->(c:EpisodicNode {group_id: "default"})-[r2]->(obj)
    WHERE r1.fact_id IS NOT NULL
      AND r1.fact_id = r2.fact_id
      AND (subj:EntityNode OR subj:TopicNode)
      AND (obj:EntityNode OR obj:TopicNode)
    RETURN subj.name as subject, type(r1) as rel, obj.name as object, r1.fact_id as fact_id
    LIMIT 5
''')
print(f'\nSample complete patterns:')
for s in sample:
    print(f'  {s["subject"]} -[{s["rel"]}]-> {s["object"]}')
    print(f'    fact_id: {s["fact_id"][:20]}...')

# Check facts vs edges ratio
facts_count = neo4j.query('MATCH (f:FactNode {group_id: "default"}) RETURN count(*) as cnt')[0]["cnt"]
print(f'\nTotal FactNodes: {facts_count}')
print(f'Complete patterns: {min(subject_edges[0]["cnt"], object_edges[0]["cnt"])}')
print(f'Coverage: {min(subject_edges[0]["cnt"], object_edges[0]["cnt"]) / facts_count * 100:.1f}%')
