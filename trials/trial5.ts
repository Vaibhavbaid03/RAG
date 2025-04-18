import 'dotenv/config';
import fetch from 'node-fetch';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { QdrantVectorStore } from '@langchain/community/vectorstores/qdrant';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence, RunnableMap } from '@langchain/core/runnables';
import { QdrantClient } from '@qdrant/js-client-rest';



//  Constants
const COLLECTION_NAME = 'mistral_rag_ollama';
const QDRANT_URL = 'http://localhost:6333';

const client = new QdrantClient({ url: QDRANT_URL });

const urls = [
  'https://en.wikipedia.org/wiki/Artificial_intelligence',
  'https://en.wikipedia.org/wiki/Machine_learning'
];

async function collectionExists(name: string): Promise<boolean> {
  try {
    const collections = await client.getCollections();
    return collections.collections.some(c => c.name === name);
  } catch (err) {
    console.error('Error checking Qdrant collections:', err);
    return false;
  }
}


async function loadWebDocuments(urlList: string[]) {
  const allDocs = [];
  for (const url of urlList) {
    console.log(` Loading URL: ${url}`);
    const loader = new CheerioWebBaseLoader(url);
    const docs = await loader.load();
    console.log(` Loaded: ${url} (${docs.length} docs)`);
    allDocs.push(...docs);
  }
  return allDocs;
}

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACE_API_KEY!,
  model: 'sentence-transformers/all-MiniLM-L6-v2'
});

const prompt = PromptTemplate.fromTemplate(`
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
`);

async function callMistralLocally(prompt: string): Promise<string> {
  const payload = {
    model: 'mistral',
    prompt: prompt,
    stream: false
  };

  console.log('\n\uD83E\uDDEA Sending to Ollama:\n', JSON.stringify(payload, null, 2));

  const res = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  console.log('\n\uD83E\uDDEA RAW response from Mistral Ollama:\n', data);

  return data?.response || 'No response from Mistral';
}

(async () => {
  let vectorstore: QdrantVectorStore;

  const alreadyExists = await collectionExists(COLLECTION_NAME);

  if (alreadyExists) {
    console.log(`✅ Qdrant collection '${COLLECTION_NAME}' already exists. Skipping reload.`);
    vectorstore = await QdrantVectorStore.fromExistingCollection(embeddings, {
      url: QDRANT_URL,
      collectionName: COLLECTION_NAME
    });
  } else {
    console.log('⏬ Collection not found. Loading and storing documents...');
    const docs = await loadWebDocuments(urls);
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const splitDocs = await splitter.splitDocuments(docs);
    vectorstore = await QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
      url: QDRANT_URL,
      collectionName: COLLECTION_NAME
    });
  }

  const retriever = vectorstore.asRetriever();

  const ragChain = RunnableSequence.from([
    RunnableMap.from({
      context: async (input: { question: string }) => {
        const docs = await retriever.invoke(input.question);
        return docs.map((d) => d.pageContent).join('\n\n');
      },
      question: (input: { question: string }) => input.question
    }),
    prompt,
    {
      invoke: async (input: any) => {
        const finalPrompt = typeof input === 'string' ? input : input?.toString?.();
        return await callMistralLocally(finalPrompt);
      }
    }
  ]);

  const response = await ragChain.invoke({
    question: 'What is artificial intelligence used for?'
  });

  console.log('\n✨ Mistral (via Ollama) Answer:\n', response);
})();
