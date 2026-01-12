#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services
neo4j = get_services().neo4j
facts = neo4j.query('MATCH (f:FactNode {group_id: "default"}) RETURN count(*) as cnt')[0]['cnt']
print(f'Total facts in DB: {facts}')