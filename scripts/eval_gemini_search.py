"""
Evaluate Gemini + Google Search Grounding with Batched Judging
==============================================================

This script optimizes OpenAI limits by batching the Judge calls.
It first runs all Gemini queries (concurrently), then sends batches of Q&A pairs to the Judge.

Usage:
    python scripts/eval_gemini_search.py --limit 10
    python scripts/eval_gemini_search.py --model gemini-2.0-flash
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from src.util.llm_client import get_critique_llm

# =============================================================================
# Data Models
# =============================================================================

class JudgeVerdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    ABSTAIN = "abstain"

class SingleJudgeResult(BaseModel):
    question_id: int = Field(..., description="The ID of the question being evaluated")
    verdict: JudgeVerdict
    reasoning: str

class BatchJudgeOutput(BaseModel):
    results: List[SingleJudgeResult]

# =============================================================================
# Agent Logic (Gemini)
# =============================================================================

def get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)

async def generate_answer(client, question: str, q_id: int, model_name: str) -> dict:
    """Generate answer using Gemini with Search Grounding."""
    start_time = time.time()
    answer = ""
    grounding_metadata = None
    
    # Prompt engineering
    prompt = f"""Question: {question}

Please answer this question using the Federal Reserve Beige Book from October 2025 as the primary source. 
Search for "Federal Reserve Beige Book October 2025" to confirm details.
Be precise and comprehensive. If you cannot find the information, please admit it."""

    MAX_RETRIES = 5
    base_delay = 2

    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}]
                }
            )
            answer = response.text
            if response.candidates and response.candidates[0].grounding_metadata:
                 grounding_metadata = response.candidates[0].grounding_metadata.search_entry_point.rendered_content
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < MAX_RETRIES - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    print(f"  !! Rate limit hit for Q{q_id}. Retrying in {sleep_time}s...")
                    await asyncio.sleep(sleep_time)
                    continue
            answer = f"Error: {e}"
            break

    return {
        "question_id": q_id,
        "question": question,
        "answer": answer,
        "agent_time_sec": time.time() - start_time
    }

# =============================================================================
# Judge Logic (Batched)
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating a BATCH of AI agent answers.
For each question, compare the Agent's Answer to the Expected Answer.

## Verdict Options:
- **correct**: Contains ALL key facts. Extra detail is allowed.
- **partially_correct**: Contains SOME key facts but missing others.
- **incorrect**: Factually WRONG or CONTRADICTS expected answer.
- **abstain**: Refused to answer (e.g. "I don't know").

## Output Format:
Return a JSON object with a 'results' list, containing one object per question with:
- question_id (must match input)
- verdict
- reasoning
"""

async def evaluate_batch(judge_llm, batch_data: List[dict]):
    """Evaluate a batch of questions."""
    
    # Construct Prompts
    user_content = "Evaluate the following questions:\n\n"
    for item in batch_data:
        user_content += f"""### Question ID: {item['question_id']}
Question: {item['question']}
Expected: {item['expected']}
Agent Answer: {item['answer']}
--------------------------------------------------
"""

    start_time = time.time()
    try:
        structured_llm = judge_llm.with_structured_output(BatchJudgeOutput)
        batch_result = await structured_llm.ainvoke([
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_content)
        ])
        
        # Map results back by ID
        result_map = {res.question_id: res for res in batch_result.results}
        
        timed_results = []
        judge_duration = (time.time() - start_time) / len(batch_data) # Average time per item
        
        for item in batch_data:
            q_id = item["question_id"]
            if q_id in result_map:
                res = result_map[q_id]
                timed_results.append({
                    **item,
                    "verdict": res.verdict.value,
                    "reasoning": res.reasoning,
                    "judge_time_sec": judge_duration
                })
            else:
                # Fallback if LLM missed an ID
                timed_results.append({
                    **item,
                    "verdict": "abstain",
                    "reasoning": "Judge failed to return a verdict for this ID in the batch.",
                    "judge_time_sec": judge_duration
                })
        return timed_results

    except Exception as e:
        print(f"Batch Judge Failed: {e}")
        # Return error results for whole batch
        return [{
            **item,
            "verdict": "abstain",
            "reasoning": f"Batch Judge Error: {e}",
            "judge_time_sec": 0
        } for item in batch_data]

# =============================================================================
# Main Loop
# =============================================================================

async def run_batch_evaluation(qa_pairs: list, model_name: str, max_concurrent: int = 5, batch_size: int = 10):
    total = len(qa_pairs)
    print(f"\n{'='*70}")
    print(f"GEMINI BATCH EVALUATION")
    print(f"Questions: {total} | Concurrency: {max_concurrent} | Batch Size: {batch_size}")
    print(f"{'='*70}\n")

    client = get_gemini_client()
    judge_llm = get_critique_llm()
    
    # 1. Generate All Answers
    print(">>> Phase 1: Generating Answers with Gemini...")
    agent_results = [None] * total
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = [0]

    async def generate_with_sem(i, pair):
        async with semaphore:
            q_id = pair.get("id", i + 1)
            result = await generate_answer(client, pair["question"], q_id, model_name)
            
            # Combine with expected answer for judging
            full_record = {
                "question_id": q_id,
                "question": pair["question"],
                "expected": pair["answer"],
                "answer": result["answer"],
                "agent_time_sec": result["agent_time_sec"]
            }
            agent_results[i] = full_record
            completed[0] += 1
            print(f"  [{completed[0]}/{total}] Generated Q{q_id}")

    await asyncio.gather(*[generate_with_sem(i, pair) for i, pair in enumerate(qa_pairs)])
    
    # 2. Batch Judge
    print("\n>>> Phase 2: Batch Judging with GPT-5.1...")
    final_results = []
    
    # Create chunks
    chunks = [agent_results[i:i + batch_size] for i in range(0, len(agent_results), batch_size)]
    
    for i, chunk in enumerate(chunks):
        print(f"  Judging Batch {i+1}/{len(chunks)} ({len(chunk)} items)...")
        batch_out = await evaluate_batch(judge_llm, chunk)
        final_results.extend(batch_out)
        
    return final_results

def print_summary(results: list, model_name: str):
    total = len(results)
    if total == 0: return
    
    correct = sum(1 for r in results if r["verdict"] == "correct")
    partial = sum(1 for r in results if r["verdict"] == "partially_correct")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")
    abstain = sum(1 for r in results if r["verdict"] == "abstain")

    avg_time = sum(r["agent_time_sec"] for r in results) / total
    avg_judge = sum(r["judge_time_sec"] for r in results) / total

    print(f"\n{'='*70}")
    print(f"BATCH EVAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Total: {total}")
    print(f"  + Correct:           {correct:3d} ({correct/total*100:5.1f}%)")
    print(f"  ~ Partially Correct: {partial:3d} ({partial/total*100:5.1f}%)")
    print(f"  - Incorrect:         {incorrect:3d} ({incorrect/total*100:5.1f}%)")
    print(f"  ? Abstain:           {abstain:3d} ({abstain/total*100:5.1f}%)")
    print(f"\n  Strict Acc: {correct/total*100:.1f}%")
    print(f"  Lenient Acc: {(correct+partial)/total*100:.1f}%")
    print(f"\n  Avg Agent Time: {avg_time:.1f}s")
    print(f"  Avg Judge Time: {avg_judge:.1f}s ( amortized )")
    print(f"{'='*70}")

def save_results(results: list, output_path: str, model_name: str):
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    partial = sum(1 for r in results if r["verdict"] == "partially_correct")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")
    abstain = sum(1 for r in results if r["verdict"] == "abstain")
    avg_time = sum(r["agent_time_sec"] for r in results) / total if total > 0 else 0

    data = {
        "timestamp": datetime.now().isoformat(),
        "system": f"gemini_batch_{model_name}",
        "total_questions": total,
        "summary": {
            "correct": correct,
            "partially_correct": partial,
            "incorrect": incorrect,
            "abstain": abstain,
            "strict_accuracy": correct / total if total > 0 else 0,
            "lenient_accuracy": (correct + partial) / total if total > 0 else 0,
            "avg_time_sec": avg_time,
        },
        "results": results
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file", default="Biege_OA.json")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--concurrency", type=int, default=2, help="Max concurrent agents")
    parser.add_argument("--batch-size", type=int, default=10, help="Judge batch size")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    qa_file = args.qa_file if os.path.isabs(args.qa_file) else project_root / "eval" / args.qa_file

    with open(qa_file) as f:
        qa_data = json.load(f)
    qa_pairs = qa_data.get("qa_pairs", qa_data)

    if args.limit:
        qa_pairs = qa_pairs[:args.limit]

    results = []
    try:
        results = await run_batch_evaluation(qa_pairs, args.model, args.concurrency, args.batch_size)
    except KeyboardInterrupt:
        print("\n\n!! Interrupted !!")
    except Exception as e:
        print(f"\n\n!! Error: {e} !!")
        import traceback
        traceback.print_exc()
    finally:
        if results:
            print_summary(results, args.model)
            output = args.output or f"eval/eval_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_results(results, output, args.model)

if __name__ == "__main__":
    asyncio.run(main())
