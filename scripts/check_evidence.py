"""Check evidence for specific questions."""
import asyncio
import json
from src.querying_system.v6 import V6Pipeline

QUESTIONS = [
    (16, "List the major subsidiaries of Alphabet mentioned in the article."),
    (26, "Why did Larry Page say they chose the name 'Alphabet'?"),
    (37, "What lawsuit did Alphabet file against Uber in 2017?"),
    (39, "What was the settlement amount for the Google+ privacy bug class action lawsuit?"),
]

async def main():
    pipeline = V6Pipeline(group_id="default")

    for qid, question in QUESTIONS:
        print(f"\n{'='*80}")
        print(f"Q{qid}: {question}")
        print('='*80)

        result = await pipeline.query(question)

        print(f"\nEVIDENCE ({len(result.evidence)} facts):")
        print("-"*40)
        for i, ev in enumerate(result.evidence[:30]):  # Show up to 30
            print(f"[{i+1}] score={ev.score:.3f}")
            print(f"    {ev.content[:200]}...")
            print()

if __name__ == "__main__":
    asyncio.run(main())
