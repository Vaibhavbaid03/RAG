// ragHandler.ts
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
import { RunnableMap, RunnableSequence } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
const COLLECTION_NAME = 'mistral_rag_ollama';
const QDRANT_URL = process.env.QDRANT_URL;
const CSV_FILE = 'Refrens_Help_URLs.csv';
const BATCH_SIZE = 10;
const DELAY_MS = 4000;
//  Connect to Qdrant Cloud
const client = new QdrantClient({
    url: QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY
});
//  Embeddings (HF)
const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: 'sentence-transformers/all-MiniLM-L6-v2'
});
//  RAG Prompt
const ragPrompt = PromptTemplate.fromTemplate(`
You are a helpful assistant. Use the following context to answer the question.
If relevant, cite the source at the end.

Context:
{context}

Question:
{question}

Answer:
`);
async function readUrlsFromCsv() {
    const fileContent = await fs.readFile(CSV_FILE, 'utf8');
    return fileContent
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.startsWith('http'));
}
async function loadAndSplitDocs(urls) {
    const allDocs = [];
    for (const url of urls) {
        try {
            const loader = new CheerioWebBaseLoader(url);
            const docs = await loader.load();
            docs.forEach(doc => doc.metadata.source = url);
            const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
            const split = await splitter.splitDocuments(docs);
            allDocs.push(...split);
        }
        catch (e) {
            console.error(` Failed to load ${url}:`, e.message);
        }
    }
    return allDocs;
}
async function checkCollectionExists() {
    try {
        const collections = await client.getCollections();
        console.log(' Qdrant connection alive:', ping.status);
        return collections.collections.some(c => c.name === COLLECTION_NAME);
    }
    catch (err) {
        console.error('Error checking Qdrant collections:', err);
        return false;
    }
}
async function callMistralLocally(prompt) {
    const res = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'mistral', prompt, stream: false })
    });
    const data = await res.json();
    return data?.response || 'No response from Mistral';
}
export default async function ragHandler(question) {
    const allUrls = await readUrlsFromCsv();
    let vectorStore;
    const exists = await checkCollectionExists();
    if (exists) {
        console.log(` Qdrant collection '${COLLECTION_NAME}' already exists.`);
        vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
            url: QDRANT_URL,
            apiKey: process.env.QDRANT_API_KEY,
            collectionName: COLLECTION_NAME
        });
    }
    else {
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
        for (let i = 0; i < allUrls.length; i += BATCH_SIZE) {
            const batch = allUrls.slice(i, i + BATCH_SIZE);
            console.log(` Batch ${i / BATCH_SIZE + 1}/${Math.ceil(allUrls.length / BATCH_SIZE)}...`);
            const docs = await loadAndSplitDocs(batch);
            if (docs.length === 0) {
                console.warn(` Skipping batch — no content extracted.`);
                continue; // Skip embedding
            }
            await vectorStore.addDocuments(docs);
            if (docs.length > 0) {
                await vectorStore.addDocuments(docs);
                console.log(` Embedded & stored ${docs.length} chunks.`);
            }
            if (i + BATCH_SIZE < allUrls.length) {
                console.log(` Waiting ${DELAY_MS}ms...`);
                await setTimeout(DELAY_MS);
            }
        }
    }
    const retriever = vectorStore.asRetriever();
    const ragChain = RunnableSequence.from([
        RunnableMap.from({
            context: async (input) => {
                const docs = await retriever.invoke(input.question);
                return docs.map(doc => `- ${doc.pageContent}\n(Source: ${doc.metadata?.source})`).join('\n\n');
            },
            question: (input) => input.question
        }),
        ragPrompt,
        new RunnableLambda({
            func: async (input) => {
                const prompt = typeof input === 'string' ? input : input?.toString?.();
                return await callMistralLocally(prompt);
            }
        })
    ]);
    const response = await ragChain.invoke({ question });
    return response;
}
