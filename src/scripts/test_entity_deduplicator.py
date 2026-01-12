#!/usr/bin/env python3
"""
Test script for entity deduplicator.

Creates 300 diverse financial entities with known duplicates,
then tests deduplication accuracy at batch sizes 100 and 200.
"""

import asyncio
import random
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from deprecated.agents.entity_deduplicator import dedupe_entities_with_llm_async

# =============================================================================
# Test Data: Groups of entities that should be deduplicated
# =============================================================================

# Each group = same entity with different names
ENTITY_GROUPS = [
    # Tech Companies
    ["Apple Inc.", "Apple", "AAPL", "Apple Computer"],
    ["Microsoft Corporation", "Microsoft", "MSFT", "Microsoft Corp"],
    ["Alphabet Inc.", "Alphabet", "GOOGL", "Google", "Google LLC"],
    ["Amazon.com Inc.", "Amazon", "AMZN", "Amazon.com"],
    ["Meta Platforms Inc.", "Meta", "META", "Facebook", "Facebook Inc."],
    ["Tesla Inc.", "Tesla", "TSLA", "Tesla Motors"],
    ["NVIDIA Corporation", "NVIDIA", "NVDA", "Nvidia"],

    # Banks & Financial
    ["JPMorgan Chase & Co.", "JPMorgan", "JPM", "JP Morgan", "Chase Bank"],
    ["Goldman Sachs Group Inc.", "Goldman Sachs", "GS", "Goldman"],
    ["Morgan Stanley", "MS", "Morgan Stanley & Co."],
    ["Bank of America Corporation", "Bank of America", "BAC", "BofA"],
    ["Citigroup Inc.", "Citigroup", "C", "Citi", "Citibank"],
    ["BlackRock Inc.", "BlackRock", "BLK"],

    # =========================================================================
    # FEDERAL RESERVE SYSTEM - All separate entities!
    # =========================================================================
    # Main Federal Reserve (these ARE the same)
    ["Federal Reserve System", "Federal Reserve", "The Fed", "Fed", "U.S. Federal Reserve", "Federal Reserve Board"],

    # Regional Federal Reserve Banks - each is SEPARATE from the main Fed and each other!
    ["Federal Reserve Bank of New York"],  # Unique - most important regional
    ["Federal Reserve Bank of San Francisco"],  # Unique
    ["Federal Reserve Bank of Chicago"],  # Unique
    ["Federal Reserve Bank of Boston"],  # Unique
    ["Federal Reserve Bank of Philadelphia"],  # Unique
    ["Federal Reserve Bank of Cleveland"],  # Unique
    ["Federal Reserve Bank of Richmond"],  # Unique
    ["Federal Reserve Bank of Atlanta"],  # Unique
    ["Federal Reserve Bank of St. Louis"],  # Unique
    ["Federal Reserve Bank of Minneapolis"],  # Unique
    ["Federal Reserve Bank of Kansas City"],  # Unique
    ["Federal Reserve Bank of Dallas"],  # Unique

    # FOMC - separate from Fed itself
    ["Federal Open Market Committee"],  # Unique - FOMC

    # Other Central Banks
    ["European Central Bank", "ECB", "The ECB"],
    ["Bank of England", "BoE", "The Bank of England"],
    ["Bank of Japan", "BoJ", "BOJ"],
    ["People's Bank of China", "PBOC", "PBoC", "The People's Bank of China"],
    ["Reserve Bank of India", "RBI", "The RBI"],
    ["Swiss National Bank", "SNB", "The SNB"],
    ["Reserve Bank of Australia", "RBA", "The RBA"],
    ["Bank of Canada", "BoC", "The Bank of Canada"],

    # International Organizations
    ["International Monetary Fund", "IMF", "The IMF"],
    ["World Bank Group", "World Bank", "The World Bank", "IBRD"],
    ["World Trade Organization", "WTO", "The WTO"],
    ["Bank for International Settlements", "BIS", "The BIS"],

    # Government Agencies
    ["Securities and Exchange Commission", "SEC", "The SEC", "U.S. SEC"],
    ["Commodity Futures Trading Commission", "CFTC", "The CFTC"],
    ["Federal Deposit Insurance Corporation", "FDIC", "The FDIC"],
    ["U.S. Department of the Treasury", "Treasury Department", "U.S. Treasury", "The Treasury"],

    # Stock Indices
    ["S&P 500 Index", "S&P 500", "SPX", "Standard & Poor's 500"],
    ["Dow Jones Industrial Average", "Dow Jones", "DJIA", "The Dow", "Dow 30"],
    ["NASDAQ Composite", "NASDAQ", "Nasdaq", "The Nasdaq"],
    ["Russell 2000 Index", "Russell 2000", "RUT"],
    ["FTSE 100 Index", "FTSE 100", "FTSE", "Footsie"],
    ["VIX Index", "VIX", "CBOE Volatility Index", "Fear Index"],

    # Currencies
    ["United States Dollar", "U.S. Dollar", "USD", "US Dollar", "The Dollar"],
    ["Euro", "EUR", "The Euro"],
    ["Japanese Yen", "JPY", "Yen", "The Yen"],
    ["British Pound Sterling", "British Pound", "GBP", "Pound Sterling"],
    ["Chinese Yuan", "Yuan", "CNY", "Renminbi", "RMB"],

    # Commodities
    ["West Texas Intermediate", "WTI", "WTI Crude", "WTI Oil"],
    ["Brent Crude Oil", "Brent", "Brent Crude", "North Sea Brent"],
    ["COMEX Gold", "Gold", "XAU", "Gold Futures"],

    # Countries (as economic entities)
    ["United States of America", "United States", "USA", "U.S.", "America"],
    ["People's Republic of China", "China", "PRC", "Mainland China"],
    ["United Kingdom", "UK", "Britain", "Great Britain"],
    ["European Union", "EU", "The EU"],

    # Rating Agencies
    ["S&P Global Ratings", "Standard & Poor's Ratings", "S&P Ratings"],
    ["Moody's Investors Service", "Moody's", "Moodys"],
    ["Fitch Ratings", "Fitch"],

    # Stock Exchanges
    ["New York Stock Exchange", "NYSE", "The NYSE"],
    ["Nasdaq Stock Market", "Nasdaq Exchange"],
    ["London Stock Exchange", "LSE", "The LSE"],

    # Big Four Auditors
    ["Deloitte Touche Tohmatsu Limited", "Deloitte", "Deloitte & Touche"],
    ["PricewaterhouseCoopers", "PwC", "PWC"],
    ["Ernst & Young", "EY", "E&Y"],
    ["KPMG International", "KPMG"],

    # Private Equity & Hedge Funds
    ["Blackstone Inc.", "Blackstone", "BX", "The Blackstone Group"],
    ["KKR & Co. Inc.", "KKR", "Kohlberg Kravis Roberts"],
    ["Bridgewater Associates", "Bridgewater"],
    ["Citadel LLC", "Citadel"],

    # Crypto
    ["Bitcoin", "BTC", "XBT"],
    ["Ethereum", "ETH", "Ether"],
    ["Coinbase Global Inc.", "Coinbase", "COIN"],

    # ETFs
    ["SPDR S&P 500 ETF Trust", "SPY", "S&P 500 ETF"],
    ["Invesco QQQ Trust", "QQQ", "Nasdaq-100 ETF"],

    # Economic Indicators (each unique)
    ["Consumer Price Index"],
    ["Gross Domestic Product"],
    ["Unemployment Rate"],
    ["Federal Funds Rate"],
    ["10-Year Treasury Yield"],

    # People
    ["Tim Cook", "Timothy Cook", "Apple CEO Tim Cook"],
    ["Satya Nadella", "Microsoft CEO Satya Nadella"],
    ["Elon Musk", "Tesla CEO Elon Musk", "SpaceX CEO Elon Musk"],
    ["Warren Buffett", "Warren E. Buffett", "The Oracle of Omaha"],
    ["Jamie Dimon", "James Dimon", "JPMorgan CEO Jamie Dimon"],
    ["Jerome Powell", "Jerome H. Powell", "Jay Powell", "Fed Chair Jerome Powell", "Chair Powell"],
    ["Janet Yellen", "Janet L. Yellen", "Treasury Secretary Janet Yellen"],
    ["Christine Lagarde", "ECB President Christine Lagarde"],
    ["Larry Fink", "Laurence Fink", "BlackRock CEO Larry Fink"],

    # Subsidiaries - should NOT merge with parent!
    ["Amazon Web Services"],  # AWS - NOT Amazon
    ["Google Cloud Platform"],  # GCP - NOT Google
    ["Microsoft Azure"],  # Azure - NOT Microsoft
    ["Instagram"],  # NOT Meta
    ["WhatsApp"],  # NOT Meta
    ["YouTube"],  # NOT Google
    ["LinkedIn"],  # NOT Microsoft

    # Similar names, different entities
    ["General Electric Company"],  # GE
    ["General Motors Company"],  # GM - different!
]


# Entity descriptions - helps LLM distinguish similar names
ENTITY_SUMMARIES = {
    # Tech Companies
    "Apple Inc.": "Technology company that makes iPhones, Macs, and software",
    "Apple": "Technology company known for consumer electronics",
    "AAPL": "Stock ticker symbol for Apple Inc. on NASDAQ",
    "Apple Computer": "Original name of Apple Inc. before 2007",
    "Microsoft Corporation": "Software company known for Windows and Office",
    "Microsoft": "Tech company that makes Windows operating system",
    "MSFT": "Stock ticker for Microsoft Corporation",
    "Microsoft Corp": "Abbreviated name for Microsoft Corporation",
    "Alphabet Inc.": "Parent company of Google, traded on NASDAQ",
    "Alphabet": "Holding company for Google and other subsidiaries",
    "GOOGL": "Stock ticker for Alphabet Inc. Class A shares",
    "Google": "Search engine and technology company owned by Alphabet",
    "Google LLC": "Subsidiary of Alphabet providing search and cloud services",
    "Google Cloud Platform": "Cloud computing services division of Google",
    "YouTube": "Video sharing platform owned by Alphabet/Google",
    "Amazon.com Inc.": "E-commerce and cloud computing company",
    "Amazon": "Online retail and cloud services company",
    "AMZN": "Stock ticker for Amazon.com Inc.",
    "Amazon.com": "E-commerce website operated by Amazon",
    "Amazon Web Services": "Cloud computing division of Amazon",
    "Meta Platforms Inc.": "Social media company formerly known as Facebook",
    "Meta": "Parent company of Facebook, Instagram, WhatsApp",
    "META": "Stock ticker for Meta Platforms Inc.",
    "Facebook": "Social networking platform owned by Meta",
    "Facebook Inc.": "Former name of Meta Platforms Inc.",
    "Instagram": "Photo sharing app owned by Meta",
    "WhatsApp": "Messaging app owned by Meta",
    "Tesla Inc.": "Electric vehicle and clean energy company",
    "Tesla": "EV maker founded by Elon Musk",
    "TSLA": "Stock ticker for Tesla Inc.",
    "Tesla Motors": "Original name of Tesla Inc.",
    "NVIDIA Corporation": "Semiconductor company known for GPUs",
    "NVIDIA": "Graphics chip maker for gaming and AI",
    "NVDA": "Stock ticker for NVIDIA Corporation",
    "Nvidia": "GPU manufacturer for gaming and data centers",

    # Banks
    "JPMorgan Chase & Co.": "Largest US bank by assets",
    "JPMorgan": "Investment banking division of JPMorgan Chase",
    "JPM": "Stock ticker for JPMorgan Chase",
    "JP Morgan": "Alternative spelling of JPMorgan",
    "Chase Bank": "Retail banking brand of JPMorgan Chase",
    "Goldman Sachs Group Inc.": "Investment bank and financial services",
    "Goldman Sachs": "Wall Street investment bank",
    "GS": "Stock ticker for Goldman Sachs",
    "Goldman": "Short name for Goldman Sachs",
    "Morgan Stanley": "Investment bank and wealth management firm",
    "MS": "Stock ticker for Morgan Stanley",
    "Morgan Stanley & Co.": "Formal name of Morgan Stanley",
    "Bank of America Corporation": "Multinational banking corporation",
    "Bank of America": "Major US retail and commercial bank",
    "BAC": "Stock ticker for Bank of America",
    "BofA": "Abbreviation for Bank of America",
    "Citigroup Inc.": "Multinational investment bank",
    "Citigroup": "Financial services company",
    "C": "Stock ticker for Citigroup",
    "Citi": "Brand name for Citigroup's consumer banking",
    "Citibank": "Consumer banking division of Citigroup",
    "BlackRock Inc.": "World's largest asset manager",
    "BlackRock": "Investment management corporation",
    "BLK": "Stock ticker for BlackRock Inc.",
    "BX": "Blackstone Inc.",

    # Federal Reserve System
    "Federal Reserve System": "Central banking system of the United States",
    "Federal Reserve": "US central bank that sets monetary policy",
    "The Fed": "Nickname for the Federal Reserve System",
    "Fed": "Short name for Federal Reserve",
    "U.S. Federal Reserve": "American central bank",
    "Federal Reserve Board": "Governing body of the Federal Reserve System",
    "Federal Reserve Bank of New York": "Regional Fed bank in New York, handles open market operations",
    "Federal Reserve Bank of San Francisco": "Regional Fed bank covering the western US",
    "Federal Reserve Bank of Chicago": "Regional Fed bank covering the midwest",
    "Federal Reserve Bank of Boston": "Regional Fed bank in New England",
    "Federal Reserve Bank of Philadelphia": "Regional Fed bank in the mid-Atlantic",
    "Federal Reserve Bank of Cleveland": "Regional Fed bank in Ohio area",
    "Federal Reserve Bank of Richmond": "Regional Fed bank in the southeast",
    "Federal Reserve Bank of Atlanta": "Regional Fed bank in the deep south",
    "Federal Reserve Bank of St. Louis": "Regional Fed bank in the central US",
    "Federal Reserve Bank of Minneapolis": "Regional Fed bank in the upper midwest",
    "Federal Reserve Bank of Kansas City": "Regional Fed bank in the plains states",
    "Federal Reserve Bank of Dallas": "Regional Fed bank in Texas region",
    "Federal Open Market Committee": "Fed committee that sets interest rate policy",

    # Central Banks
    "European Central Bank": "Central bank of the eurozone",
    "ECB": "Abbreviation for European Central Bank",
    "The ECB": "European Central Bank",
    "Bank of England": "Central bank of the United Kingdom",
    "BoE": "Abbreviation for Bank of England",
    "The Bank of England": "UK's central bank",
    "Bank of Japan": "Central bank of Japan",
    "BoJ": "Abbreviation for Bank of Japan",
    "BOJ": "Stock-style abbreviation for Bank of Japan",
    "People's Bank of China": "Central bank of China",
    "PBOC": "Abbreviation for People's Bank of China",
    "PBoC": "Alternative abbreviation for People's Bank of China",
    "The People's Bank of China": "China's central bank",
    "Reserve Bank of India": "Central bank of India",
    "RBI": "Abbreviation for Reserve Bank of India",
    "The RBI": "India's central bank",
    "Swiss National Bank": "Central bank of Switzerland",
    "SNB": "Abbreviation for Swiss National Bank",
    "The SNB": "Switzerland's central bank",
    "Reserve Bank of Australia": "Central bank of Australia",
    "RBA": "Abbreviation for Reserve Bank of Australia",
    "The RBA": "Australia's central bank",
    "Bank of Canada": "Central bank of Canada",
    "BoC": "Abbreviation for Bank of Canada",
    "The Bank of Canada": "Canada's central bank",

    # Crypto
    "Bitcoin": "First and largest cryptocurrency by market cap",
    "BTC": "Ticker symbol for Bitcoin",
    "XBT": "Alternative ticker for Bitcoin used on some exchanges",
    "Ethereum": "Blockchain platform with smart contract functionality",
    "ETH": "Ticker symbol for Ethereum's native token Ether",
    "Ether": "Native cryptocurrency of the Ethereum network",
    "Coinbase Global Inc.": "Cryptocurrency exchange company",
    "Coinbase": "US-based crypto trading platform",
    "COIN": "Stock ticker for Coinbase",

    # Indices
    "S&P 500 Index": "Stock index of 500 largest US companies",
    "S&P 500": "Benchmark US stock market index",
    "SPX": "Ticker symbol for S&P 500 index",
    "Standard & Poor's 500": "Full name of S&P 500 index",
    "Dow Jones Industrial Average": "Index of 30 major US stocks",
    "Dow Jones": "Shortened name for the Dow index",
    "DJIA": "Abbreviation for Dow Jones Industrial Average",
    "The Dow": "Nickname for Dow Jones index",
    "Dow 30": "Reference to the 30 stocks in the Dow",
    "NASDAQ Composite": "Index of all stocks on NASDAQ exchange",
    "NASDAQ": "Stock exchange and its composite index",
    "Nasdaq": "Alternative capitalization for NASDAQ",
    "The Nasdaq": "Reference to NASDAQ exchange or index",
    "Russell 2000 Index": "Index of 2000 small-cap US stocks",
    "Russell 2000": "Small-cap stock index",
    "RUT": "Ticker for Russell 2000",
    "FTSE 100 Index": "Index of 100 largest UK companies",
    "FTSE 100": "UK's main stock index",
    "FTSE": "UK stock index (Financial Times Stock Exchange)",
    "Footsie": "Nickname for FTSE index",
    "VIX Index": "Volatility index measuring market fear",
    "VIX": "CBOE Volatility Index ticker",
    "CBOE Volatility Index": "Measures expected S&P 500 volatility",
    "Fear Index": "Nickname for VIX",

    # ETFs
    "SPDR S&P 500 ETF Trust": "ETF that tracks the S&P 500 index",
    "SPY": "Ticker for SPDR S&P 500 ETF",
    "S&P 500 ETF": "Exchange-traded fund tracking S&P 500",
    "Invesco QQQ Trust": "ETF tracking the Nasdaq-100",
    "QQQ": "Ticker for Invesco QQQ Trust",
    "Nasdaq-100 ETF": "ETF tracking top 100 Nasdaq stocks",

    # Currencies
    "United States Dollar": "Official currency of the United States",
    "U.S. Dollar": "American currency",
    "USD": "Currency code for US Dollar",
    "US Dollar": "The dollar used in America",
    "The Dollar": "Informal name for US Dollar",
    "Euro": "Official currency of the eurozone",
    "EUR": "Currency code for Euro",
    "The Euro": "European common currency",
    "Japanese Yen": "Official currency of Japan",
    "JPY": "Currency code for Japanese Yen",
    "Yen": "Japanese currency",
    "The Yen": "Japan's currency",
    "British Pound Sterling": "Official currency of the UK",
    "British Pound": "UK currency",
    "GBP": "Currency code for British Pound",
    "Pound Sterling": "Full name of UK currency",
    "Chinese Yuan": "Official currency of China",
    "Yuan": "Chinese currency",
    "CNY": "Currency code for Chinese Yuan",
    "Renminbi": "Official name of Chinese currency",
    "RMB": "Abbreviation for Renminbi",

    # Commodities
    "West Texas Intermediate": "US benchmark crude oil price",
    "WTI": "Abbreviation for West Texas Intermediate oil",
    "WTI Crude": "US crude oil benchmark",
    "WTI Oil": "American crude oil standard",
    "Brent Crude Oil": "International oil price benchmark",
    "Brent": "North Sea oil benchmark",
    "Brent Crude": "Global crude oil price standard",
    "North Sea Brent": "Oil from North Sea used as benchmark",
    "COMEX Gold": "Gold futures traded on COMEX exchange",
    "Gold": "Precious metal commodity",
    "XAU": "Currency code for gold",
    "Gold Futures": "Derivatives contracts for gold",

    # People
    "Tim Cook": "CEO of Apple Inc. since 2011",
    "Timothy Cook": "Full name of Apple CEO Tim Cook",
    "Apple CEO Tim Cook": "Tim Cook in his role at Apple",
    "Satya Nadella": "CEO of Microsoft since 2014",
    "Microsoft CEO Satya Nadella": "Satya Nadella in his role at Microsoft",
    "Elon Musk": "CEO of Tesla and SpaceX",
    "Tesla CEO Elon Musk": "Elon Musk in his role at Tesla",
    "SpaceX CEO Elon Musk": "Elon Musk in his role at SpaceX",
    "Warren Buffett": "CEO of Berkshire Hathaway, legendary investor",
    "Warren E. Buffett": "Full name of Warren Buffett",
    "The Oracle of Omaha": "Nickname for Warren Buffett",
    "Jamie Dimon": "CEO of JPMorgan Chase",
    "James Dimon": "Full name of Jamie Dimon",
    "JPMorgan CEO Jamie Dimon": "Jamie Dimon in his role at JPMorgan",
    "Jerome Powell": "Chair of the Federal Reserve",
    "Jerome H. Powell": "Full name of Fed Chair Jerome Powell",
    "Jay Powell": "Nickname for Jerome Powell",
    "Fed Chair Jerome Powell": "Jerome Powell in his Fed role",
    "Chair Powell": "Jerome Powell as Fed Chair",
    "Janet Yellen": "US Treasury Secretary, former Fed Chair",
    "Janet L. Yellen": "Full name of Janet Yellen",
    "Treasury Secretary Janet Yellen": "Janet Yellen in her Treasury role",
    "Christine Lagarde": "President of the European Central Bank",
    "ECB President Christine Lagarde": "Christine Lagarde in her ECB role",
    "Larry Fink": "CEO of BlackRock",
    "Laurence Fink": "Full name of Larry Fink",
    "BlackRock CEO Larry Fink": "Larry Fink in his role at BlackRock",

    # Other
    "LinkedIn": "Professional networking platform owned by Microsoft",
    "Microsoft Azure": "Cloud computing platform by Microsoft",
    "General Electric Company": "Industrial conglomerate, makes jet engines and power equipment",
    "General Motors Company": "Automobile manufacturer, makes Chevrolet and Cadillac",
}


def create_test_entities(num_entities: int = 300) -> tuple[list[dict], dict[str, str]]:
    """Create a randomized list of entities with known ground truth."""
    all_entities = []
    ground_truth = {}

    for group in ENTITY_GROUPS:
        canonical = group[0]
        for name in group:
            summary = ENTITY_SUMMARIES.get(name, "")
            all_entities.append({"name": name, "summary": summary})
            ground_truth[name] = canonical

    random.shuffle(all_entities)

    if len(all_entities) > num_entities:
        all_entities = all_entities[:num_entities]
        ground_truth = {k: v for k, v in ground_truth.items() if k in [e["name"] for e in all_entities]}

    print(f"  Created {len(all_entities)} entities from {len(ENTITY_GROUPS)} groups")
    return all_entities, ground_truth


def evaluate_dedup(result: dict[str, str], ground_truth: dict[str, str]) -> dict[str, float]:
    """Evaluate deduplication accuracy."""
    result_groups = {}
    for entity, canonical in result.items():
        result_groups.setdefault(canonical, []).append(entity)

    truth_groups = {}
    for entity, canonical in ground_truth.items():
        truth_groups.setdefault(canonical, []).append(entity)

    correct_pairs = 0
    total_result_pairs = 0
    total_truth_pairs = 0

    for canonical, entities in result_groups.items():
        # Only count pairs where BOTH entities are in ground_truth
        entities_in_truth = [e for e in entities if e in ground_truth]
        n = len(entities_in_truth)
        total_result_pairs += n * (n - 1) // 2
        for i, e1 in enumerate(entities_in_truth):
            for e2 in entities_in_truth[i+1:]:
                if ground_truth[e1] == ground_truth[e2]:
                    correct_pairs += 1

    for canonical, entities in truth_groups.items():
        entities_in_result = [e for e in entities if e in result]
        n = len(entities_in_result)
        total_truth_pairs += n * (n - 1) // 2

    precision = correct_pairs / max(total_result_pairs, 1)
    recall = correct_pairs / max(total_truth_pairs, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    wrong_merges = total_result_pairs - correct_pairs

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_pairs": correct_pairs,
        "wrong_merges": wrong_merges,
        "total_entities": len(result),
        "unique_canonicals": len(set(result.values())),
        "expected_canonicals": len(set(v for k, v in ground_truth.items() if k in result))
    }


async def test_batch_size(entities: list[dict], ground_truth: dict[str, str], batch_size: int, llm):
    """Test deduplication with a specific batch size."""
    print(f"\n{'='*60}")
    print(f"Testing batch_size={batch_size}")
    print(f"{'='*60}")

    start = time.time()
    result = await dedupe_entities_with_llm_async(entities, llm, batch_size=batch_size)
    elapsed = time.time() - start

    metrics = evaluate_dedup(result, ground_truth)

    print(f"Time: {elapsed:.1f}s")
    print(f"Entities: {metrics['total_entities']}")
    print(f"Unique canonicals: {metrics['unique_canonicals']} (expected: {metrics['expected_canonicals']})")
    print(f"Correct pairs: {metrics['correct_pairs']}, Wrong merges: {metrics['wrong_merges']}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"F1 Score: {metrics['f1']:.1%}")

    # Show merged groups
    result_groups = {}
    for entity, canonical in result.items():
        result_groups.setdefault(canonical, []).append(entity)

    merged_groups = [(c, e) for c, e in result_groups.items() if len(e) > 1]
    print(f"\nMerged groups ({len(merged_groups)}):")
    for canonical, entities in sorted(merged_groups, key=lambda x: -len(x[1]))[:10]:
        # Check if all entities in this group SHOULD be together
        # (i.e., they all map to the same ground truth canonical)
        entities_in_truth = [e for e in entities if e in ground_truth]
        if entities_in_truth:
            truth_canonicals = set(ground_truth[e] for e in entities_in_truth)
            all_correct = len(truth_canonicals) == 1  # All map to same canonical
        else:
            all_correct = True  # No ground truth to compare
        status = "✓" if all_correct else "✗"
        print(f"  {status} {canonical}: {entities}")

    # Check for wrong merges (entities that should NOT be together)
    print(f"\nWrong merges:")
    wrong_examples = []
    for canonical, entities in result_groups.items():
        if len(entities) > 1:
            # Only consider entities that are in ground_truth
            entities_in_truth = [e for e in entities if e in ground_truth]
            if len(entities_in_truth) > 1:
                truth_canonicals = set(ground_truth[e] for e in entities_in_truth)
                if len(truth_canonicals) > 1:
                    wrong_examples.append((canonical, entities_in_truth, truth_canonicals))

    if wrong_examples:
        for canonical, entities, truths in wrong_examples[:5]:
            print(f"  ✗ Merged: {entities}")
            print(f"    Should be separate: {truths}")
    else:
        print("  None!")

    # Check Fed handling specifically
    print(f"\nFed-related entities:")
    fed_entities = [e for e in result if "Fed" in e or "Federal" in e]
    for e in fed_entities:
        mapped_to = result.get(e, e)
        expected = ground_truth.get(e, e)
        status = "✓" if mapped_to == expected or ground_truth.get(mapped_to) == expected else "✗"
        if mapped_to != e:
            print(f"  {status} '{e}' -> '{mapped_to}'")

    return {"batch_size": batch_size, "time": elapsed, **metrics}


async def main():
    print("="*60)
    print("ENTITY DEDUPLICATOR TEST")
    print("="*60)

    # Start smaller - 100 entities first
    print("\nCreating test entities...")
    entities, ground_truth = create_test_entities(100)
    print(f"  {len(set(ground_truth.values()))} expected canonical groups")

    print("\nInitializing LLM (Claude Sonnet 4.5)...")
    # llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    llm = ChatAnthropic(model="claude-opus-4-5", temperature=0)
    # llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0)
    print("  Ready")

    # Just test batch size 100 first (single batch = fastest)
    results = []
    for batch_size in [50]:
        result = await test_batch_size(entities, ground_truth, batch_size, llm)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Batch':<8} {'Time':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Canonicals':<12} {'Wrong':<8}")
    print("-"*60)
    for r in results:
        print(f"{r['batch_size']:<8} {r['time']:.1f}s{'':<3} {r['precision']:.1%}{'':<2} {r['recall']:.1%}{'':<2} {r['f1']:.1%}{'':<2} {r['unique_canonicals']}/{r['expected_canonicals']:<6} {r['wrong_merges']:<8}")


if __name__ == "__main__":
    asyncio.run(main())
