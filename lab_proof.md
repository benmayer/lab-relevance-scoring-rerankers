# Lab Proof: First match ≠ right match

Notebook: [relevance_scoring_rerankers.ipynb](relevance_scoring_rerankers.ipynb)

## Reviewer summary

**What I built:** a RAG pipeline over two sources (the Trustworthy AI podcast transcript and the EU AI Act PDF). There are 784 chunks with metadata, embedded with `text-embedding-3-small` and stored in Pinecone. On top of baseline vector search I added an LLM relevance reranker (`gpt-4o-mini`), a local cross-encoder reranker, metadata filtering by source, section and article, and a citation-grounded answer chain.

**The traced query:** "What fines apply for using a prohibited AI practice?"
- The baseline ranks Article 4 (AI literacy) first, which says nothing about fines. See [Step 2](#step-2-embeddings--baseline-retrieval).
- The LLM reranker moves it to rank 4 and puts penalty passages in the top three. See [Step 3](#step-3-llm-relevance-scoring).
- Final cited answer: up to EUR 35,000,000 or 7% of worldwide annual turnover, citing EU AI Act p.115. See [Step 6](#step-6-citation-grounded-rag), which has the claim-to-chunk trace.

**The failure:** my metadata filter `article == "Article 5"` returns the Article 4 chunk at rank 1, because a chunk ending at the Article 5 heading is labelled Article 5 as a whole. See [Failure or limitation](#failure-or-limitation).

**Not yet captured:** the Step 4 cross-encoder output, the Step 7 evaluation table, and confirmation that the fine amounts are inside the cited p.115 chunk. Each is marked where it occurs.

## Step 1: Setup and Data Preparation

Reflection questions:

### 1. What metadata is missing?

The chunks currently have `source_type`, `source_name`, `chunk_id`, and for the EU AI Act, `page`. They're missing fields like `section` (recital, article, or annex) or `article` (for example, Article 5). So I can filter by source type, but I can't reliably filter for Article 5 or for recitals. I would need to extract those labels from the regulation text, add them to each chunk's metadata, and rebuild the vector index before filtering on them.

### 2. How can a reviewer locate a podcast chunk?

The podcast has no page numbers, timestamps, or speaker labels in the provided text. Its generated `chunk_id` identifies a chunk in the processed collection, but it is not a human-readable location in the transcript. A reviewer can use the chunk ID together with a short quoted snippet to search `data/podcast.txt`. For stronger traceability, preserve character offsets (or timestamps and speaker labels if available) in each chunk's metadata and cite those with the chunk ID.

## Step 2: Embeddings + Baseline Retrieval

### Question: For which query does the **top result** look wrong or only half-relevant? That's your *first match ≠ right match* case.

Here are the results: 

``` 
=== Baseline | Query: What AI practices are prohibited under the EU AI Act?
#1  score=0.698  [regulation | p.51.0]  'Article 4\nAI literacy\nProviders and deployers of AI systems shall take measures to ensure, to their best extent, a suffi'
#2  score=0.684  [regulation | p.12.0]  '(45) Practices that are prohibited by Union law, including data protection law, non-discrimination law, consumer \nprotec'
#3  score=0.681  [regulation | p.9.0]  'intention to distort behaviour where the distortion results from factors external to the AI system which are outside \nth'
#4  score=0.673  [regulation | p.45.0]  '(b) prohibitions of certain AI practices;\n(c) specific requirements for high-risk AI systems and obligations for operato'
#5  score=0.660  [regulation | p.3.0]  'operators involved in the AI value chain under this Regulation should apply without prejudice to national law, in \ncompl'

=== Baseline | Query: What are the seven requirements for trustworthy AI?
#1  score=0.674  [podcast | podcast-2]  'make AI powerful to how do we make AI worthy of trust? And our mission for this deep dive is to figure out how to operat'
#2  score=0.611  [regulation | p.8.0]  '(27) While the risk-based approach is the basis for a proportionate and effective set of binding rules, it is important '
#3  score=0.587  [regulation | p.22.0]  '(75) Technical robustness is a key requirement for high-risk AI systems. They should be resilient in relation to harmful'
#4  score=0.584  [regulation | p.61.0]  'The requirement for a separate verification by at least two natural persons shall not apply to high-risk AI systems used'
#5  score=0.576  [regulation | p.8.0]  'coherent, trustworthy and human-centric AI, in line with the Charter and with the values on which the Union is \nfounded.'

=== Baseline | Query: What is the difference between human-in-the-loop and human-on-the-loop?
#1  score=0.531  [podcast | podcast-8]  "human in command. It's a gradient of control. Let's break that down. Human in the loop. That's the tightest control. The"
#2  score=0.402  [regulation | p.21.0]  'functioning, ensure that they are used as intended and that their impacts are addressed over the system’s lifecycle. To '
#3  score=0.383  [podcast | podcast-7]  "crime data is biased, which it almost always is, you're violating fairness by targeting specific neighborhoods over and "
#4  score=0.347  [podcast | podcast-18]  'this all together, we started with a bridge. We trust the bridge because of physics and laws. Can we ever get there with'
#5  score=0.345  [regulation | p.60.0]  '(e) the computational and hardware resources needed, the expected lifetime of the high-risk AI system and any necessary '

=== Baseline | Query: What fines apply for using a prohibited AI practice?
#1  score=0.633  [regulation | p.51.0]  'Article 4\nAI literacy\nProviders and deployers of AI systems shall take measures to ensure, to their best extent, a suffi'
#2  score=0.617  [regulation | p.118.0]  '(d) failed to make available to the Commission access to the general-purpose AI model or general-purpose AI model with \n'
#3  score=0.613  [regulation | p.115.0]  'including start-ups, and their economic viability.\n2. The Member States shall, without delay and at the latest by the da'
#4  score=0.610  [regulation | p.117.0]  'with regard to the same subject matter;\n(e) any similar previous infringements by the Union institution, body, office or'
#5  score=0.584  [regulation | p.117.0]  'undertakings in the protection of their personal data or business secrets.\n6. Funds collected by imposition of fines in '

=== Baseline | Query: Is social scoring of citizens allowed?
#1  score=0.588  [regulation | p.9.0]  'multiple data points related to their social behaviour in multiple contexts or known, inferred or predicted personal \nor'
#2  score=0.497  [regulation | p.9.0]  'face or fingerprint, to deduce or infer an individuals’ political opinions, trade union membership, religious or \nphilos'
#3  score=0.497  [regulation | p.51.0]  '(c) the placing on the market, the putting into service or the use of AI systems for the evaluation or classification of'
#4  score=0.485  [podcast | podcast-16]  'learning for every child. The potential is incredible, but the expert group lists some specific critical concerns. These'
#5  score=0.421  [regulation | p.16.0]  'should be granted, denied, reduced, revoked or reclaimed by authorities, including whether beneficiaries are \nlegitimate'
``` 

To me it seems like the chunk of the first result is only ever partially relevant and wouldn't be able to answer the question fully.

The clearest examples are **'What fines apply for using a prohibited AI practice?'** and **'What AI practices are prohibited under the EU AI Act?'** Both rank Article 4 on AI literacy (page 51) first, even though that text does not directly answer either query. More relevant passages appear lower in the results. 

## Step 3: LLM Relevance Scoring

Output for one question here:

``` 
LLM relevance calls: 10

=== Baseline top 5 | Query: What fines apply for using a prohibited AI practice?
#1  score=0.633  [regulation | p.51.0]  'Article 4\nAI literacy\nProviders and deployers of AI systems shall take measures to ensure, to their best extent, a suffi'
#2  score=0.617  [regulation | p.118.0]  '(d) failed to make available to the Commission access to the general-purpose AI model or general-purpose AI model with \n'
#3  score=0.613  [regulation | p.115.0]  'including start-ups, and their economic viability.\n2. The Member States shall, without delay and at the latest by the da'
#4  score=0.610  [regulation | p.117.0]  'with regard to the same subject matter;\n(e) any similar previous infringements by the Union institution, body, office or'
#5  score=0.584  [regulation | p.117.0]  'undertakings in the protection of their personal data or business secrets.\n6. Funds collected by imposition of fines in '

=== LLM reranked top 5 (combined score) | Query: What fines apply for using a prohibited AI practice?
#1  score=0.884  [regulation | p.115.0]  'including start-ups, and their economic viability.\n2. The Member States shall, without delay and at the latest by the da'
#2  score=0.883  [regulation | p.117.0]  'with regard to the same subject matter;\n(e) any similar previous infringements by the Union institution, body, office or'
#3  score=0.875  [regulation | p.117.0]  'undertakings in the protection of their personal data or business secrets.\n6. Funds collected by imposition of fines in '
#4  score=0.190  [regulation | p.51.0]  'Article 4\nAI literacy\nProviders and deployers of AI systems shall take measures to ensure, to their best extent, a suffi'
#5  score=0.185  [regulation | p.118.0]  '(d) failed to make available to the Commission access to the general-purpose AI model or general-purpose AI model with \n'

``` 

The ranking changed, but by itself it would not be able to answer the question. The chunks are too small imo to really contain all the relevant information to answer the question by itself.

Observation on the scores: the combined scores split into two groups, around 0.88 for ranks 1–3 and around 0.19 for ranks 4–5. With weights 0.3 similarity / 0.7 LLM, a score near 0.19 means `gpt-4o-mini` scored the chunk about 0. So it judged only 3 of the 10 candidates relevant, and the "top 5" is padded with two chunks it considered irrelevant.

## Step 4: Cross-encoder reranker

Implemented in the notebook (`crossencoder_rerank`, model `cross-encoder/ms-marco-MiniLM-L-6-v2`, reranking the same 10 baseline candidates, with a side-by-side against baseline and the LLM reranker including timing).

**Not yet captured.** Paste the output here and say whether it picked the same top 3 as the LLM reranker, and how its runtime compares to the 10 API calls.

## Step 5: Metadata Filtering

Enrichment found 764 EU AI Act chunks (784 total): 313 recital, 362 article and 89 annex chunks, with 6 chunks labelled Article 5. Metadata was patched in place on the 764 existing Pinecone vectors using their stable chunk IDs, without re-embedding.

Query: "What AI practices are prohibited under the EU AI Act?"

```
=== EU AI Act articles only | Query: What AI practices are prohibited under the EU AI Act?
#1  score=0.699  [regulation | p.51.0]  'Article 4\nAI literacy\nProviders and deployers of AI systems shall take measures to ensure, to their best extent, a suffi'
#2  score=0.673  [regulation | p.45.0]  '(b) prohibitions of certain AI practices;\n(c) specific requirements for high-risk AI systems and obligations for operato'
#3  score=0.643  [regulation | p.119.0]  'rules on artificial intelligence and amending Regulations (EC) No 300/2008, (EU) No 167/2013, (EU) No 168/2013, \n(EU) 20'
#4  score=0.636  [regulation | p.46.0]  '6. This Regulation does not apply to AI systems or AI models, including their output, specifically developed and put int'
#5  score=0.622  [regulation | p.121.0]  '(EU) 2018/858, (EU) 2018/1139 and (EU) 2019/2144 and Directives 2014/90/EU, (EU) 2016/797 and (EU) \n2020/1828 (Artificia'

=== EU AI Act Article 5 only | Query: What AI practices are prohibited under the EU AI Act?
#1  score=0.699  [regulation | p.51.0]  'Article 4\nAI literacy\nProviders and deployers of AI systems shall take measures to ensure, to their best extent, a suffi'
#2  score=0.605  [regulation | p.51.0]  'assessments of natural persons in order to assess or predict the risk of a natural person committing a criminal offence,'
#3  score=0.603  [regulation | p.51.0]  '(c) the placing on the market, the putting into service or the use of AI systems for the evaluation or classification of'
#4  score=0.568  [regulation | p.51.0]  'materially distorting the behaviour of a person or a group of persons by appreciably impairing their ability to make an '
#5  score=0.556  [regulation | p.51.0]  'of a natural person in the areas of workplace and education institutions, except where the use of the AI system is \n'
```

The Article 5 filter returns only regulation chunks labelled Article 5, and ranks 2 to 5 are genuine prohibition text (criminal-offence risk assessment, social scoring, behavioural distortion, workplace and education emotion recognition). Rank 1 is the Article 4 chunk, mislabelled; see [Failure or limitation](#failure-or-limitation). The source-only comparison (regulation vs podcast) was not captured.

## Step 6: Citation-grounded RAG

The notebook formats each retrieved chunk as `[source, p.N]` (or `[source, chunk_id]` for the podcast) and instructs the answer model to use only that context or say "I don't know based on the sources." The fines query runs through it on the LLM-reranked chunks from Step 3 (the five chunks in the second table of the Step 3 output).

**Final answer:**

> Non-compliance with the prohibition of the AI practices referred to in Article 5 shall be subject to administrative fines of up to EUR 35,000,000 or, if the offender is an undertaking, up to 7% of its total worldwide annual turnover for the preceding financial year, whichever is higher [EU AI Act (Reg. 2024/1689), p.115].

**Trace:**

| Claim in the answer | Cited chunk |
|---|---|
| Fines apply to non-compliance with the Article 5 prohibitions | regulation p.115 (rank 1 after reranking) |
| Up to EUR 35,000,000 | regulation p.115 |
| Or up to 7% of total worldwide annual turnover, whichever is higher | regulation p.115 |

**Caveat:** the printed 120-character snippet of the p.115 chunk starts with "including start-ups…" and does not show the fine amounts. The answer cites p.115, but I have not yet printed the full chunk text to confirm that the EUR 35,000,000 and 7% wording is inside that exact chunk. To close this, print `rag_evidence[0][0].page_content` and paste the relevant lines here.

## Step 7: Evaluation

The notebook's `evaluate_queries(...)` helper is defined but has not been called yet, so the table below is only partly filled in. Fill it in by reading the retrieved evidence.

| Query | Method | Top-1 chunk relevant? (Y/N) | Answer correct? (Y/Partial/N) | Grounded in cited chunk? (Y/N) | Notes |
|---|---|---|---|---|---|
| Q4 (fines) | Baseline | N (Article 4, AI literacy) | | | |
| Q4 (fines) | LLM reranked | | | | |
| Q4 (fines) | Cross-encoder | | | | |
| Q1, Q2, Q3, Q5 | | | | | not yet run |

What is known already:
- **Order changed:** for the fines query, reranking moved Article 4 from rank 1 to rank 4 and put three penalty-related passages (p.115, p.117, p.117) in the top three.
- **Cost:** the LLM reranker made 10 relevance-scoring calls plus 1 answer call for this query. Elapsed time and money were not recorded.

To answer after the table is filled in:
- Did reranking change the answer, or only the order?
- What did it cost in extra calls and seconds per query, and would a client pay for that?

## Failure or limitation

Metadata filtering can return plausible but wrong results. I filtered the query "What AI practices are prohibited under the EU AI Act?" to `article == "Article 5"` (Step 5). The #1 result (score 0.699) is the **Article 4 AI literacy chunk on p.51**, the same wrong chunk the unfiltered baseline returned. My enrichment heuristic assigns each chunk the *last* `Article N` heading found inside it, so a chunk that begins with Article 4 and ends with the Article 5 heading is labelled Article 5 as a whole. The filter therefore looks precise but admits an unrelated chunk, and an answer built from it would cite Article 4 text as if it were a prohibition.

Further sources of unsupported answers:
- Chunks are about 1000 characters. A fine, its conditions and its cross-references can sit in different chunks, so a reranked top chunk may mention fines without stating the amount (see the Step 6 caveat).
- The LLM relevance scores are coarse, close to 0 or 1 (see the Step 3 observation), so the ordering inside each group comes mostly from the similarity score.
- The EU AI Act chunk labels depend on a heuristic (recitals assumed to end on p.44, article numbers read from line-start text), and I have not checked it against the PDF beyond the Article 5 count (6 chunks).
- The podcast chunks have no page, timestamp or speaker label, so a reviewer can only locate them by `chunk_id` plus a quoted snippet.
