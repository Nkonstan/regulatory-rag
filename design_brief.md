# **Part 2: System Design Brief**

## **The Problem**

Current RAG handles simple Q&A: *"What does ICH E3 say about adverse events?"*

Complex query: *"Find all adverse event reporting requirements across these documents, cross-reference with our protocol, and identify compliance gaps."*

**Challenge:** Standard RAG retrieves what *exists* in documents. Finding what's *missing* requires multi-step reasoning:
1. Extract requirements from regulations (multi-document retrieval)
2. Parse relevant protocol sections (structured extraction)
3. Compare contexts to detect gaps (what's required but absent)
4. Validate every claim against sources (prevent hallucinations)

***

## **Architecture: Bounded Agentic Pipeline**

### **What "Agentic" Means Here**

**Not fully autonomous:** LLM doesn't recursively decide next steps (unpredictable cost/latency, hard to audit for FDA).

Bounded agency: Fixed 5-node pipeline where every step is predefined, logged, and auditable:

    ✅ Decomposer — Breaks user query into 1–3 concrete subtasks (e.g., "retrieve AE requirements", "extract protocol Section 9")

    ✅ Retriever — Executes RAG search against regulation documents; returns top 10 chunks 

    ✅ Parser — Extracts the full relevant protocol section using bookmark-based, regex, or manual fallback strategies 

    ✅ Analyzer — Performs two-context comparison (regulations vs. protocol); identifies requirements present in regulations but absent in protocol

    ✅ Validator — Checks every claimed gap against source chunks via semantic similarity; assigns High/Medium/Low confidence scores and flags hallucinations

**Result:** Complex reasoning with predictable performance.

***

### **System Diagram**

```
┌─────────────────────────────────────────────────────┐
│                    USER QUERY                       │
│        "Find AE compliance gaps..."                 │
└──────────────────────┬──────────────────────────────┘
                       │ raw query (string)
                       ▼
┌─────────────────────────────────────────────────────┐
│  [1] DECOMPOSER                        [LLM]        │
│  Breaks query into 1–3 subtasks                     │
│  Output: subtask list + target doc sections         │
└──────────┬──────────────────────────────────────────┘
           │ subtask list (JSON)
           │
           ▼
┌─────────────────────────────────────────────────────┐
│  [2+3] PARALLEL EXECUTION                           │
│  ┌───────────────────┐  ┌────────────────────────┐  │
│  │  [2] RETRIEVER    │  │  [3] PARSER            │  │
│  │  RAG search over  │  │  Extract full protocol │  │
│  │  regulation docs  │  │  section (no chunking) │  │
│  │  → top 10 chunks  │  │  → 3-strategy fallback │  │
│  │  (~5k tokens)     │  │  (~6k tokens)          │  │
│  └────────┬──────────┘  └───────────┬────────────┘  │
└───────────┼─────────────────────────┼───────────────┘
            │ reg. chunks (JSON)      │ protocol section (markdown)
            └────────────┬────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  [4] ANALYZER                          [LLM]        │
│  Two-context comparison:                            │
│  "What's in regulations but absent in protocol?"    │
│  Output: gap list with claimed citations            │
└──────────────────────┬──────────────────────────────┘
                       │ gap list + citations (JSON)
                       ▼
┌─────────────────────────────────────────────────────┐
│  [5] VALIDATOR                  [Semantic Similarity]│
│  Check each gap claim against source chunks          │
│  Assign confidence: High (>0.80) / Med / Low (<0.65) │
└───────┬──────────────────────┬───────────────────────┘
        │                      │
        │ High confidence       │ Medium / Low confidence
        ▼                      ▼
┌──────────────┐     ┌─────────────────────────┐
│   RESPONSE   │     │   HUMAN REVIEW QUEUE    │
│ JSON output  │     │   Flagged for expert    │
│ + audit trail│     │   validation            │
└──────────────┘     └─────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AUDIT LOG (cross-cutting): Every node logs inputs,
 outputs, model version, tokens, timestamp, user ID
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
**Implementation:** LangGraph state machine (deterministic flow, retry logic, auditability)

***

## **Key Design Decisions**

### **1. Gap Detection: Two-Context Comparison**

**Core problem:** RAG is a presence-detection tool — it finds relevant content that exists. Gap detection is an absence-detection problem — it requires reasoning over a complete context to identify what doesn't exist. These require different retrieval strategies.

**Solution:**
1. Regulations → RAG (top 10 chunks, ~5k tokens): Regulations are large and multi-topic; RAG efficiently surfaces only the AE-relevant sections without exceeding context limits
2. Protocol → Full section extraction (not chunked, ~6k tokens): Chunking the protocol would create false gaps — a requirement split across two adjacent chunks would appear absent when it isn't
3. Single LLM call: *"Here are regulations [context A]. Here's protocol [context B]. List requirements in A but absent in B."*

**Tradeoff:** Higher token cost vs accuracy.

***

### **2. Citation Validation: Semantic Similarity**

**Problem:** LLM might claim *"ICH E3 requires 24h reporting"* when E3 doesn't say this.

**Two-layer check:**
1. **Document existence:** Does cited source (e.g., "ICH E3 Section 12.3") appear in retrieved chunks?
2. **Semantic similarity:** Is gap claim (e.g., "expedited reporting ≤7 days") similar to actual chunk text?

**Confidence scoring:**
- Similarity >0.80 → High confidence (auto-include in response)
- 0.65-0.80 → Medium (flag for review)
- <0.65 → Low (flag as "needs expert validation")

**Fallback:** If >30% of gaps fail validation → Error: "Insufficient source coverage, refine query or add documents." 

**Why this matters:** Pharma regulatory submissions must cite exact sections. Hallucinated citations = audit failure.

***

### **3. Protocol Parsing: Multi-Strategy Extraction**

**Challenge:** Protocols are messy PDFs—tables spanning pages, varying section names ("Adverse Events" vs "9.0 Safety"), headers/footers on every page.

**Three fallback strategies:**
1. **Bookmark-based** (80% success for ICH-format protocols)
2. **Regex search** with fuzzy matching: `Section \d+.*Adverse Event|Safety|AE`
3. **Manual fallback:** If auto-detection fails, API accepts page range: `{section_pages: [45, 52]}`

**Table handling:** Use pdfplumber (better than PyMuPDF for tables) → Convert to markdown for LLM readability

**Realistic success rate:**
- ✅ 80-90%: Standard protocols with bookmarks
- ✅ 60-70%: Text PDFs without bookmarks
- ❌ 20-30%: Scanned PDFs (needs OCR preprocessing)

***

## **Explicit Scope Boundaries**

**What this design includes:**
✅ Single complex query → Gap analysis response  
✅ Citation validation + confidence scoring  
✅ Human review workflow for flagged gaps  
✅ Full audit trail for regulatory compliance  

**What's excluded (for MVP):**
❌ Multi-turn conversation (single query/response only)  
❌ Automatic protocol amendments (identifies gaps but doesn't fix)  
❌ UI for side-by-side comparison (API only)  
❌ Multi-language support (English only)  

***

## **Risk Mitigation**

| Risk                      | Impact | Mitigation                                           |
| ------------------------- | ------ | ---------------------------------------------------- |
| LLM hallucinates gaps     | High   | Two-layer citation validation + confidence scoring   |
| Protocol parsing fails    | Medium | Three fallback strategies + manual page range option |
| Retrieval misses key regs | High   | Curated document set + version tracking + warnings   |
| Cost overruns             | Medium | Hard token limits (50k protocol, 10k regs)           |