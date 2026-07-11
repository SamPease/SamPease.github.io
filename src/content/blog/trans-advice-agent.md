---
title: "Trans Advice Agent"
description: "A retrieval-augmented QA system for community-sourced transgender healthcare information discovery."
date: 2025-08-15
tags: ["RAG", "NLP", "information retrieval"]
draft: false
---

# Trans Advice Agent

## Overview

TransAdviceAgent is an experimental tool designed to answer transgender-related questions by compiling and synthesizing community knowledge from forums and online discussions. The focus is on questions that require community guidance rather than simple factual answers, as there are very few trans-related topics backed by comprehensive research.

Many medical questions and practical recommendations are best answered by community forums where people share real experiences. This project aims to make that collective knowledge more accessible by using advanced natural language processing to search, summarize, and synthesize information from these community sources.

The system uses query enhancement with synonyms to improve search relevance, retrieval of diverse sources using embedding similarity, and hierarchical summarization to provide comprehensive answers grounded in community experiences.

Please note that this is an **amateur project** and not a professional medical resource. Answers are limited to the community documents included in the system and may be incomplete, biased, or outdated. Always consult licensed healthcare providers and multiple sources when making decisions about your care.

**Privacy Notice:** This system uses LangSmith for monitoring and logging all interactions, including questions and generated responses. Do not input sensitive personal information or medical details.

Currently, the system draws primarily from Wikipedia and Reddit. The database will be expanded in future iterations to include additional trans community forums, official resources, and curated documents.

Questions, feedback, or suggestions are welcome at [samantha.a.pease@gmail.com](mailto:samantha.a.pease@gmail.com).

## Try It Out

The embedded chat interface is available below.

<iframe id="trans-agent-iframe" src="/TransAdviceAgent.html" width="100%" height="380" style="border:none; min-height: 320px;" title="Trans Advice Agent"></iframe>

<script>
   (function () {
      const iframe = document.getElementById("trans-agent-iframe");
      if (!iframe) return;

      const minHeight = 320;
      const maxHeight = 1500;

      function setHeight(nextHeight) {
         const clamped = Math.max(minHeight, Math.min(maxHeight, Math.round(nextHeight)));
         iframe.style.height = clamped + "px";
      }

      window.addEventListener("message", (event) => {
         if (event.origin !== window.location.origin) return;
         const data = event.data || {};
         if (data.type !== "trans-agent:height") return;
         if (typeof data.height !== "number") return;
         setHeight(data.height + 8);
      });
   })();
</script>

If the embed does not load, open it directly: [TransAdviceAgent app](/TransAdviceAgent.html).

## How the Pipeline Works

TransAdviceAgent processes user questions through a multi-stage pipeline designed to surface relevant community knowledge and provide comprehensive, well-sourced answers. The system emphasizes diverse perspectives and remains grounded in community documents.

1. **Query Enhancement**  
   The original question is expanded with synonyms and alternative phrasings using Claude Haiku. This helps semantic retrieval find relevant content despite vocabulary differences.

2. **Embedding-Based Retrieval**  
   Documents are chunked and stored in a FAISS vector index with SQLite metadata. The enhanced query is embedded with all-MiniLM-L6-v2, then matched semantically. Maximal Marginal Relevance (MMR) is used for diversity and reduced redundancy.

3. **Parallel Batch Summarization**  
   Retrieved chunks are processed in parallel batches using Claude Haiku. Each batch summary preserves key details, outcomes, and conflicting viewpoints.

4. **Hierarchical Summary Combination**  
   Batch summaries are combined into a single synthesis that aggregates recurring patterns while retaining distinct perspectives.

5. **Final Answer Generation**  
   Using the original question and combined synthesis, Claude Haiku generates a final answer intended to be comprehensive, balanced, and source-grounded.

Project repository: [github.com/SamPease/TransAdviceAgent](https://github.com/SamPease/TransAdviceAgent)

## Tech Stack

- **Python**: Core language for the end-to-end pipeline.
- **FastAPI**: Backend API framework.
- **Claude Haiku**: Query enhancement, summarization, and answer generation.
- **all-MiniLM-L6-v2**: Embedding model for semantic similarity search.
- **LangChain/LangSmith**: Pipeline orchestration and logging.
- **FAISS**: Vector similarity search with MMR; uses IVFPQ for memory-efficient scaling.
- **SQLite**: Document metadata and text storage.
- **Hugging Face**: Hosting for prebuilt FAISS/SQLite artifacts and embedding inference.
- **Render**: Backend hosting.
- **GitHub Pages**: Static frontend hosting.
- **GitHub**: Version control and open-source code hosting.

## Limitations and Future Work

TransAdviceAgent is an experimental, hobby project and has several limitations that users should be aware of:

- **Document Coverage**: The system can only provide answers based on the documents in its database. If a topic, clinician, or procedure is not well-represented, the answer may be incomplete or indicate insufficient evidence.
- **Community Bias**: Retrieved documents reflect the perspectives of those who chose to share their experiences. They may not represent the full range of outcomes or experiences for a given healthcare provider or procedure.
- **Latency**: Because the system uses multiple LLM calls for summarization, synthesis, and answering, responses may be slow. Additionally, the biggest barrier to response time is that the backend runs on a free hosting tier that spins down due to inactivity and can take up to two minutes to spin back up.
- **Abbreviation Expansion**: Automatic expansion of initials to full clinician names is based on a curated mapping and context. Unmapped or ambiguous initials may lead to missed documents or incorrect matches.
- **No Medical Advice**: The system summarizes community-sourced information and is not a substitute for professional medical guidance. Users should consult licensed providers for personal healthcare decisions.

Planned future improvements:

- Expand data sources beyond Reddit to include additional trans community forums, resources, and documentation.
- Improve retrieval quality with additional techniques, such as sparse keyword search or hybrid dense-sparse approaches.
- Incorporate provenance tracking for each fact in the final answer, showing exactly which documents contributed to which statements.
- Optimize latency and scalability, including batching strategies and lighter-weight summarization models for faster responses.
- Enhance abbreviation and synonym expansion using automated context-aware methods to reduce manual mapping and increase accuracy.
