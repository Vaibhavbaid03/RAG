// ragHandler.ts, embedding using OLLAMA, huggingFace is SLOW.

import 'dotenv/config';
import * as fs from 'fs/promises';
import fetch from 'node-fetch';
import { setTimeout } from 'timers/promises';
import { RunnableLambda } from '@langchain/core/runnables';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { QdrantVectorStore } from '@langchain/qdrant';
import { QdrantClient } from '@qdrant/js-client-rest';
import { Document } from 'langchain/document';
import { RunnableMap, RunnableSequence } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import { OllamaEmbeddings } from '@langchain/ollama';
import cliProgress from 'cli-progress';


const COLLECTION_NAME = 'mistral_rag_ollama';
const QDRANT_URL = process.env.QDRANT_URL!;
const CSV_FILE = 'Refrens_Help_URLs.csv';
const BATCH_SIZE = 5;
const DELAY_MS = 4000;

// Qdrant Client (Cloud)
const client = new QdrantClient({
  url: QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY
});

// Embedding Model (HF)
/*
const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACE_API_KEY!,
  model: 'sentence-transformers/all-MiniLM-L6-v2'
});
*/

const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text' });

// RAG Prompt
const ragPrompt = PromptTemplate.fromTemplate(`
You are a helpful assistant. Use the following context to answer the question.
Cite relevant (Source: ...) links at the end if provided in the context.

Context:
{context}

Question:
{question}

Answer:
`);

async function readUrlsFromCsv(): Promise<string[]> {
  const fileContent = await fs.readFile(CSV_FILE, 'utf8');
  return fileContent
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.startsWith('http'));
}

async function loadAndSplitDocs(urls: string[]): Promise<Document[]> {
  const allDocs: Document[] = [];
  for (const url of urls) {
    try {
      console.log(`  Loading URL: ${url}`);
      const loader = new CheerioWebBaseLoader(url);
      const docs = await loader.load();
      console.log(`  Loaded ${docs.length} docs from: ${url}`);

      docs.forEach(doc => doc.metadata.source = url);
      const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 512, chunkOverlap: 64 });
      const split = await splitter.splitDocuments(docs);
      console.log(`  Split into ${split.length} chunks.`);
      allDocs.push(...split);
    } catch (e: any) {
      console.error(`  Failed to load ${url}:`, e.message || e);
    }
  }
  return allDocs;
}


async function checkCollectionExists(): Promise<boolean> {
  try {
    const collections = await client.getCollections();
    console.log(` Qdrant connected. Found ${collections.collections.length} collections.`);
    return collections.collections.some(c => c.name === COLLECTION_NAME);
  } catch (err) {
    console.error('Error checking Qdrant collections:', err);
    return false;
  }
}

async function callMistralLocally(prompt: string): Promise<string> {
  const res = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'mistral', prompt, stream: false })
  });
  const data = await res.json();
  return data?.response || 'No response from Mistral';
}

export default async function ragHandler(question: string): Promise<string> {
  const allUrls = await readUrlsFromCsv();
  let vectorStore: QdrantVectorStore;

  const exists = await checkCollectionExists();

  if (exists) {
  console.log(` Qdrant collection '${COLLECTION_NAME}' already exists.`);
  vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
    url: QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
    collectionName: COLLECTION_NAME
  });
} else {
  console.log(' First-time setup. Embedding all documents...');
  console.log(` Found ${allUrls.length} URLs in CSV`);

  vectorStore = await QdrantVectorStore.fromDocuments([], embeddings, {
    url: QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
    collectionName: COLLECTION_NAME,
    collectionOptions: {
      onDiskPayload: true,
      hnswConfig: {
        m: 16,
        efConstruct: 100
      }
    }
  });

  const progressBar = new cliProgress.SingleBar({
    format: 'Embedding Progress |{bar}| {percentage}% | Batch {value}/{total} | ETA: {eta_formatted}',
    barCompleteChar: '█',
    barIncompleteChar: '░',
    hideCursor: true
  });

  const totalBatches = Math.ceil(allUrls.length / BATCH_SIZE);
  progressBar.start(totalBatches, 0);
  const startTime = Date.now();

  for (let i = 0; i < allUrls.length; i += BATCH_SIZE) {
    const batchNum = i / BATCH_SIZE + 1;
    const batch = allUrls.slice(i, i + BATCH_SIZE);
    console.log(`\n Batch ${batchNum}/${totalBatches}...`);

    console.log(' Loading & splitting docs...');
    const docs = await loadAndSplitDocs(batch);
    console.log(` Extracted ${docs.length} chunks.`);

    if (docs.length > 0) {
      console.log(` Storing ${docs.length} chunks...`);
      await vectorStore.addDocuments(docs);
      console.log(`Embedded & stored.`);

      const elapsed = (Date.now() - startTime) / 1000;
      console.log(` Time elapsed so far: ${elapsed.toFixed(1)}s`);

      progressBar.increment();
    } else {
      console.warn(` Skipping batch — no content extracted.`);
    }

    if (i + BATCH_SIZE < allUrls.length) {
      console.log(` Waiting ${DELAY_MS}ms before next batch...`);
      await setTimeout(DELAY_MS);
    }
  }

  progressBar.stop();
  console.log(' Embedding complete!');
}



  const retriever = vectorStore.asRetriever();

  const ragChain = RunnableSequence.from([
    RunnableMap.from({
      context: async (input: { question: string }) => {
        const docs = await retriever.invoke(input.question);
        return docs.map(doc => `- ${doc.pageContent}\n(Source: ${doc.metadata?.source})`).join('\n\n');
      },
      question: (input: { question: string }) => input.question
    }),
    ragPrompt,
    new RunnableLambda({
      func: async (input: any) => {
        const prompt = typeof input === 'string' ? input : input?.toString?.();
        return await callMistralLocally(prompt);
      }
    })
  ]);

  const response = await ragChain.invoke({ question });
  return response;
}