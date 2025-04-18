//  Load 100+ Refrens Help URLs from CSV
//  Embed with HuggingFace
//  Store into Qdrant
//  Ask a question via Mistral (Ollama)

import 'dotenv/config';
import fs from 'fs/promises';
import fetch from 'node-fetch';
import csv from 'csv-parser';
import { setTimeout } from 'timers/promises';

import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { QdrantVectorStore } from '@langchain/community/vectorstores/qdrant';
import { QdrantClient } from '@qdrant/js-client-rest';
import { Document } from 'langchain/document';
import { RunnableMap, RunnableSequence } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';

const COLLECTION_NAME = 'mistral_rag_ollama';
const QDRANT_URL = 'http://localhost:6333';
const CSV_FILE = 'Refrens_Help_URLs.csv';
const BATCH_SIZE = 10;
const DELAY_MS = 4000;

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACE_API_KEY!,
  model: 'sentence-transformers/all-MiniLM-L6-v2'
});

const client = new QdrantClient({ url: QDRANT_URL });

async function readUrlsFromCsv(): Promise<string[]> {
  const data: string[] = [];
  const fileContent = await fs.readFile(CSV_FILE, 'utf8');
  const lines = fileContent.split('\n').map(line => line.trim()).filter(Boolean);
  for (const url of lines) {
    if (url.startsWith('http')) data.push(url);
  }
  return data;
}

async function loadAndSplitDocs(urls: string[]): Promise<Document[]> {
  const allDocs: Document[] = [];
  for (const url of urls) {
    try {
      const loader = new CheerioWebBaseLoader(url);
      const docs = await loader.load();
      docs.forEach(doc => doc.metadata.source = url);
      const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
      const split = await splitter.splitDocuments(docs);
      allDocs.push(...split);
    } catch (e) {
      console.error(` Failed to load ${url}:`, e.message);
    }
  }
  return allDocs;
}

async function checkCollectionExists(): Promise<boolean> {
  try {
    const collections = await client.getCollections();
    return collections.collections.some(c => c.name === COLLECTION_NAME);
  } catch (err) {
    console.error('Error checking Qdrant collections:', err);
    return false;
  }
}

async function callMistralLocally(prompt: string): Promise<string> {
  const payload = {
    model: 'mistral',
    prompt,
    stream: false
  };

  console.log('\n Sending to Ollama:\n', JSON.stringify(payload, null, 2));

  const res = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  console.log('\n RAW response from Mistral Ollama:\n', data);
  return data?.response || 'No response from Mistral';
}

const ragPrompt = PromptTemplate.fromTemplate(`
You are a helpful assistant. Use the following context to answer the question.
If relevant, cite the source at the end.

Context:
{context}

Question:
{question}

Answer:
`);

async function run() {
  const allUrls = await readUrlsFromCsv();
  let vectorStore: QdrantVectorStore;
  const exists = await checkCollectionExists();

  if (exists) {
    console.log(` Collection '${COLLECTION_NAME}' already exists. Using existing.`);
    vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
      url: QDRANT_URL,
      collectionName: COLLECTION_NAME
    });
  } else {
    console.log(' Collection not found. Loading and chunking sitemap articles...');
    vectorStore = await QdrantVectorStore.fromTexts(['dummy'], [{}], embeddings, {
      url: QDRANT_URL,
      collectionName: COLLECTION_NAME
    });
    for (let i = 0; i < allUrls.length; i += BATCH_SIZE) {
      const batch = allUrls.slice(i, i + BATCH_SIZE);
      console.log(` Processing batch ${i / BATCH_SIZE + 1} of ${Math.ceil(allUrls.length / BATCH_SIZE)}...`);
      const docs = await loadAndSplitDocs(batch);
      if (docs.length > 0) {
        await vectorStore.addDocuments(docs);
        console.log(` Embedded & stored ${docs.length} chunks.`);
      }
      if (i + BATCH_SIZE < allUrls.length) {
        console.log(`⏳ Waiting ${DELAY_MS}ms to avoid rate limits...`);
        await setTimeout(DELAY_MS);
      }
    }
    console.log(' All documents processed and stored in Qdrant!');
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
    {
      invoke: async (input: any) => {
        const prompt = typeof input === 'string' ? input : input?.toString?.();
        return await callMistralLocally(prompt);
      }
    }
  ]);

  const finalAnswer = await ragChain.invoke({ question: 'how to add a payment ?' });
  console.log('\n Answer from Mistral:\n', finalAnswer);
}

run();
