# Lab Proof: First match ≠ right match

Notebook: [relevance_scoring_rerankers.ipynb](relevance_scoring_rerankers.ipynb)

## Reviewer summary

**What I built:** a RAG pipeline over two sources (the Trustworthy AI podcast transcript and the EU AI Act PDF). There are 784 chunks with metadata, embedded with `text-embedding-3-small` and stored in Pinecone. On top of baseline vector search I added an LLM relevance reranker (`gpt-4o-mini`), a local cross-encoder reranker, metadata filtering by source, section and article, and a citation-grounded answer chain.

**The traced query:** "What fines apply for using a prohibited AI practice?"
- The baseline ranks Article 4 (AI literacy) first, which says nothing about fines. See [Step 2](#step-2-embeddings--baseline-retrieval).
- The LLM reranker and the cross-encoder both move it down and put the penalty passages first. See [Step 3](#step-3-llm-relevance-scoring) and [Step 4](#step-4-cross-encoder-reranker).
- Final cited answer: up to EUR 35,000,000 or 7% of worldwide annual turnover, citing EU AI Act p.115. The full text of that chunk contains both figures. See [Step 6](#step-6-citation-grounded-rag), which has the claim-to-chunk trace.

**Baseline vs two rerankers (five queries, [Step 7](#step-7-evaluation)):** reranking changed the order of the evidence much more often than it changed the answer. The answers were the same in 3 of 5 queries (fines, seven requirements, human-in/on-the-loop) and similar in content in a fourth (social scoring). The LLM reranker took about 2.2x the time of the baseline and added 50 API calls; the local cross-encoder took about 1.2x.

**The failure:** the prohibited-practices query returns a plausible but incomplete answer. Every method lists only Article 5(1)(a) (subliminal manipulation) plus recital text, because the chunks holding the other prohibitions in Article 5 never reach the top 5. The metadata filter shows the same weakness from another side. See [Failure or limitation](#failure-or-limitation).

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

*Update after Step 7 (added later, my original answer above is unchanged):* printing the full text of the baseline's #1 chunk shows it is not only Article 4. After the AI-literacy article it continues with `CHAPTER II / PROHIBITED AI PRACTICES / Article 5 / 1. The following AI practices shall be prohibited: (a) the placing on the market ... subliminal techniques ...`. So for the prohibited-practices query this chunk is partly relevant, which fits the observation above, but it does contain the start of Article 5. For the fines query it is still unrelated.

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

*Update after Step 7:* for the fines query the chunk that the answer cites (p.115) does contain the full penalty text (EUR 35 000 000 and 7 %; see Step 6). So chunk size was not what blocked the answer here. The printed 120-character snippets just make the chunks look too small to answer from.

Observation on the scores: the combined scores split into two groups, around 0.88 for ranks 1–3 and around 0.19 for ranks 4–5. With weights 0.3 similarity / 0.7 LLM, a score near 0.19 means `gpt-4o-mini` scored the chunk about 0. So it judged only 3 of the 10 candidates relevant, and the "top 5" is padded with two chunks it considered irrelevant.

## Step 4: Cross-encoder reranker

Implemented in the notebook (`crossencoder_rerank`, model `cross-encoder/ms-marco-MiniLM-L-6-v2`, reranking the same 10 baseline candidates, with a side-by-side against baseline and the LLM reranker including timing).

Output (top 3 per method, same 10 baseline candidates; cross-encoder scores are raw logits, not 0–1):

```
=== Baseline top 3 | Query: What fines apply for using a prohibited AI practice?
#1  score=0.634  [regulation | p.51.0]  'Article 4\nAI literacy\nProviders and deployers of AI systems shall take measures to ensure, to their best extent, a suffi'
#2  score=0.617  [regulation | p.118.0]  '(d) failed to make available to the Commission access to the general-purpose AI model or general-purpose AI model with \n'
#3  score=0.613  [regulation | p.115.0]  'including start-ups, and their economic viability.\n2. The Member States shall, without delay and at the latest by the da'

=== LLM reranked top 3 (12.7s) | Query: What fines apply for using a prohibited AI practice?
#1  score=0.884  [regulation | p.115.0]  'including start-ups, and their economic viability.\n2. The Member States shall, without delay and at the latest by the da'
#2  score=0.883  [regulation | p.117.0]  'with regard to the same subject matter;\n(e) any similar previous infringements by the Union institution, body, office or'
#3  score=0.875  [regulation | p.117.0]  'undertakings in the protection of their personal data or business secrets.\n6. Funds collected by imposition of fines in '

=== Cross-encoder top 3 (4.0s; scores are logits) | Query: What fines apply for using a prohibited AI practice?
#1  score=4.259  [regulation | p.115.0]  'including start-ups, and their economic viability.\n2. The Member States shall, without delay and at the latest by the da'
#2  score=4.098  [regulation | p.117.0]  'with regard to the same subject matter;\n(e) any similar previous infringements by the Union institution, body, office or'
#3  score=2.504  [regulation | p.108.0]  'establishes that the AI system was misclassified by the provider as non-high-risk in order to circumvent the application'

=== Baseline top 3 | Query: What AI practices are prohibited under the EU AI Act?
#1  score=0.699  [regulation | p.51.0]  'Article 4\nAI literacy\nProviders and deployers of AI systems shall take measures to ensure, to their best extent, a suffi'
#2  score=0.684  [regulation | p.12.0]  '(45) Practices that are prohibited by Union law, including data protection law, non-discrimination law, consumer \nprotec'
#3  score=0.682  [regulation | p.9.0]  'intention to distort behaviour where the distortion results from factors external to the AI system which are outside \nth'

=== LLM reranked top 3 (11.2s) | Query: What AI practices are prohibited under the EU AI Act?
#1  score=0.910  [regulation | p.51.0]  'Article 4\nAI literacy\nProviders and deployers of AI systems shall take measures to ensure, to their best extent, a suffi'
#2  score=0.896  [regulation | p.12.0]  'generalisability. Therefore, AI systems identifying or inferring emotions or intentions of natural persons on the basis '
#3  score=0.205  [regulation | p.12.0]  '(45) Practices that are prohibited by Union law, including data protection law, non-discrimination law, consumer \nprotec'

=== Cross-encoder top 3 (4.0s; scores are logits) | Query: What AI practices are prohibited under the EU AI Act?
#1  score=6.939  [regulation | p.12.0]  '(45) Practices that are prohibited by Union law, including data protection law, non-discrimination law, consumer \nprotec'
#2  score=6.909  [regulation | p.9.0]  'intention to distort behaviour where the distortion results from factors external to the AI system which are outside \nth'
#3  score=6.685  [regulation | p.51.0]  'Article 4\nAI literacy\nProviders and deployers of AI systems shall take measures to ensure, to their best extent, a suffi'
```

What the output shows:
- **Fines query:** the cross-encoder and the LLM reranker agree on ranks 1–2 (p.115 and p.117) and both push Article 4 out of the top 3. They differ on rank 3: the LLM picks another p.117 penalty passage, the cross-encoder picks a p.108 chunk about providers misclassifying a system as non-high-risk, which is a different penalty rule rather than fines for prohibited practices.
- **Prohibited-practices query:** the two rerankers disagree. The LLM reranker keeps the Article 4/5 chunk at #1 with a high score (0.910). That is defensible, because the chunk ends with the start of Article 5(1)(a). The cross-encoder puts recital 45 and a p.9 recital first and the Article 4/5 chunk third. Neither brings up the rest of Article 5 (points (b) to (h)), which only appears with the Step 5 `Article 5` filter, so those chunks were probably not among the 10 baseline candidates. A reranker can only reorder what retrieval returns.
- **Speed:** the cross-encoder took 4.0 s per query against 11–13 s for the LLM reranker, which makes 10 API calls. Both timings include the Pinecone retrieval of the 10 candidates. The cross-encoder is local and costs nothing per call, apart from the one-time model download.

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

The Article 5 filter returns only regulation chunks labelled Article 5, and ranks 2 to 5 are genuine prohibition text (criminal-offence risk assessment, social scoring, behavioural distortion, workplace and education emotion recognition). Rank 1 is the chunk that begins with Article 4 and continues into the start of Article 5, so the label is only half right; see [Failure or limitation](#failure-or-limitation).

Source-only filter, same query (`source_type` = regulation vs podcast):

```
=== Regulation only | Query: What AI practices are prohibited under the EU AI Act?
#1  score=0.699  [regulation | p.51.0]  'Article 4\nAI literacy\nProviders and deployers of AI systems shall take measures to ensure, to their best extent, a suffi'
#2  score=0.684  [regulation | p.12.0]  '(45) Practices that are prohibited by Union law, including data protection law, non-discrimination law, consumer \nprotec'
#3  score=0.682  [regulation | p.9.0]  'intention to distort behaviour where the distortion results from factors external to the AI system which are outside \nth'
#4  score=0.673  [regulation | p.45.0]  '(b) prohibitions of certain AI practices;\n(c) specific requirements for high-risk AI systems and obligations for operato'
#5  score=0.660  [regulation | p.3.0]  'operators involved in the AI value chain under this Regulation should apply without prejudice to national law, in \ncompl'

=== Podcast only | Query: What AI practices are prohibited under the EU AI Act?
#1  score=0.519  [podcast | podcast-4]  "a pixel, it doesn't recognize or can be easily hacked, then it causes harm. An intentional harm is still harm. So lawful"
#2  score=0.502  [podcast | podcast-17]  "in danger's human autonomy. It's fundamentally anti-democratic. Another major red flag is law, WS legal autonomous weapo"
#3  score=0.491  [podcast | podcast-5]  "AI shouldn't hurt us or condition us or manipulate us. It shouldn't deceive you into clicking something you wouldn't oth"
#4  score=0.484  [podcast | podcast-11]  'with a frighteningly high accuracy. So even if I keep my data private, the AI just guesses it anyway. And once it guesse'
#5  score=0.479  [podcast | podcast-10]  'teaching a self-driving car that a stop sign is actually a speed limit 45 sign by feeding it thousands of corrupted imag'
```

The source filter works as expected: the regulation-only list is the same as the unfiltered baseline top 5, while the podcast-only list returns different chunks with clearly lower similarity (0.48–0.52 vs 0.66–0.70). That is useful as a signal, because the podcast discusses harm and manipulation in general terms but does not describe what the AI Act prohibits.

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

**Caveat resolved:** the printed snippet of the p.115 chunk is cut off at 120 characters, so I printed the full chunk (last cell of the notebook). It contains the cited wording: "3. Non-compliance with the prohibition of the AI practices referred to in Article 5 shall be subject to administrative fines of up to EUR 35 000 000 or, if the offender is an undertaking, up to 7 % of its total worldwide annual turnover for the preceding financial year, whichever is higher." All three claims in the trace above are therefore supported by that one chunk. (The same chunk also lists the lower tier in paragraph 4, up to EUR 15 000 000 or 3 %, for other provisions. The answer correctly does not mix the two.)

## Step 7: Evaluation

`evaluate_queries()` ran all five fixed queries with three retrieval methods (baseline, LLM reranked, cross-encoder) and generated a cited answer for each. That is 50 relevance-scoring calls plus 15 answer calls. The full printout (answers, evidence, seconds) is in the last cell of the notebook.

The judgments below are my draft reading of those printouts and should be checked against the chunks before they are treated as final.

| Query | Method | Top-1 chunk relevant? (Y/N) | Answer correct? (Y/Partial/N) | Grounded in cited chunk? (Y/N) | Notes |
|---|---|---|---|---|---|
| Q1 prohibited practices | Baseline | Partial (Article 4 text, then the start of Article 5) | Partial | Partial | Lists only 5(1)(a) subliminal manipulation, plus an "unfair commercial practices" item taken from a recital (p.9) about existing Union law. Misses the other Article 5 prohibitions. |
| Q1 prohibited practices | LLM reranked | Partial (same chunk, score 0.910) | Partial | Partial | Same answer text as the baseline. |
| Q1 prohibited practices | Cross-encoder | Partial (recital 45, p.12, about practices prohibited by Union law) | Partial | Partial | Adds items from recitals p.8 and p.9 and a closing sentence about dignity and rights, still no Article 5(1)(b)-(h). |
| Q2 seven requirements | Baseline | N (podcast intro, not the list) | Y | Y | Seven HLEG requirements listed, cited p.8 (that chunk was already at rank 2 and 5). |
| Q2 seven requirements | LLM reranked | Y (p.8) | Y | Y | Same list. |
| Q2 seven requirements | Cross-encoder | Y (p.8) | Y | Y | Same list. |
| Q3 human-in/on-the-loop | Baseline | Y (podcast-8) | Y | Partial | Cites podcast-8 and podcast-7. podcast-7 looks unrelated (crime data bias); check whether the human-on-the-loop definition is really in it. |
| Q3 human-in/on-the-loop | LLM reranked | Y (podcast-8) | Y | Partial | Same citations; wording of "human-on-the-loop" differs slightly between runs ("unusual or problematic", "unusual"). |
| Q3 human-in/on-the-loop | Cross-encoder | Y (podcast-8) | Y | Partial | Same citations ("if something appears to be wrong"). |
| Q4 fines | Baseline | N (Article 4 AI literacy part) | Y | Y | Correct because p.115 was already rank 3 in the baseline top 5. |
| Q4 fines | LLM reranked | Y (p.115) | Y | Y | Identical answer to the baseline. |
| Q4 fines | Cross-encoder | Y (p.115) | Y | Y | Identical answer to the baseline. |
| Q5 social scoring | Baseline | Y (p.9) | Y | Y | Matches the wording of Article 5(1)(c) (p.51); also adds a podcast remark that citizen scoring is "anti-democratic", cited separately. |
| Q5 social scoring | LLM reranked | Y (p.9) | Y | Y | Order unchanged: four chunks all scored about 0.85, so the similarity score decided. |
| Q5 social scoring | Cross-encoder | Partial (p.9 chunk starting with biometric text) | Y | Y | Moved the Article 5(1)(c) chunk from #3 down to #4; the answer still used it. |

Seconds per query (retrieval + scoring + answer generation; single run, so the numbers are noisy):

| Method | Q1 | Q2 | Q3 | Q4 | Q5 | Total | Mean |
|---|---|---|---|---|---|---|---|
| Baseline | 9.5 | 4.2 | 5.4 | 8.9 | 9.7 | 37.7 | 7.5 |
| LLM reranked | 18.9 | 12.3 | 13.2 | 19.4 | 19.0 | 82.8 | 16.6 |
| Cross-encoder | 7.4 | 6.4 | 11.2 | 10.0 | 10.8 | 45.8 | 9.2 |

**Did reranking change the answer, or only the order?** Mostly the order. For fines, seven requirements, human-in/on-the-loop and social scoring, the baseline answer was already correct and the reranked answers said the same thing, because the needed chunk was already in the baseline top 5. What reranking did change was the top-1 chunk: for Q2 the podcast intro was replaced by the p.8 recital, and for Q4 the Article 4/5 chunk was replaced by p.115. For Q1, the LLM reranker returned the same answer as the baseline, and the cross-encoder returned a different one that added recital-based items rather than the missing Article 5 prohibitions. Metadata filtering (Step 5) found those Article 5 chunks and reranking did not.

**What did it cost, and would a client pay for it?** The LLM reranker added 50 `gpt-4o-mini` calls for five queries and roughly doubled the time per query (mean 16.6 s vs 7.5 s). I did not record tokens or money. The cross-encoder added about 1.7 s on average (mean 9.2 s vs 7.5 s, within the run-to-run noise visible in the baseline column) and costs nothing per call. For this small corpus and k=5, I would not pay for the LLM reranker, because the baseline already contained the evidence. The cross-encoder is cheap enough to keep when the top-1 chunk matters, or when I shrink k to cut down the context.

## Failure or limitation

The prohibited-practices query ("What AI practices are prohibited under the EU AI Act?") produces a plausible but incomplete and partly misattributed answer with all three retrieval methods.
- **Incomplete:** the answers list the subliminal-manipulation prohibition (Article 5(1)(a)) and nothing else from Article 5. The other prohibitions (for example the social scoring, criminal-offence prediction and workplace emotion-recognition points that the `Article 5` filter returned in Step 5) are in chunks that never reach the top 5, and a reranker cannot promote chunks it was never given. The answer still reads as a complete list because nothing signals that points are missing.
- **Misattributed:** the baseline and LLM answers add "unfair commercial practices ... are prohibited under all circumstances". That comes from a recital (p.9) describing prohibitions that already exist in Union law. Citing it as an AI Act prohibition is not supported by that chunk. The citation is real and the claim is plausible, so a reviewer would only catch it by reading the chunk.

Related weaknesses:
- **Chunks cross article boundaries.** The first chunk of that query starts with Article 4 and continues into Article 5. My enrichment heuristic gives each chunk the *last* `Article N` heading it contains, so this chunk carries `article = "Article 5"`. The filter `article == "Article 5"` therefore returns it at rank 1, together with its AI-literacy text. A single-valued `article` field cannot describe a chunk like this. I have not checked how many other chunks straddle two articles.
- **LLM scores are coarse.** In all five queries the LLM reranker scored 1 to 4 chunks around 0.85–0.91 and the rest around 0.1–0.2. When several chunks score alike (Q5), the ordering is effectively decided by the similarity score.
- **Citations can be too generous.** In Q3 every method cites podcast-7 next to podcast-8 for a definition that, judging by the snippet, may only be in podcast-8.
- **Podcast chunks cannot be located precisely.** They have no page, timestamp or speaker label, so a reviewer finds them only through `chunk_id` plus a quoted snippet.
