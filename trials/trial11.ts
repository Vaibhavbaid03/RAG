// trial11.ts
//   Qdrant Cloud test
// Just connecting and uploading  a single page,

import 'dotenv/config';
import fetch from 'node-fetch';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { QdrantVectorStore } from '@langchain/community/vectorstores/qdrant';
import { QdrantClient } from '@qdrant/js-client-rest';

const QDRANT_URL = 'https://7be21947-a45c-49a8-a3ae-6682f86afdb3.eu-west-1-0.aws.cloud.qdrant.io';
const COLLECTION_NAME = 'mistral_rag_ollama_test';

const client = new QdrantClient({
  url: QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACE_API_KEY!,
  model: 'sentence-transformers/all-MiniLM-L6-v2',
});

async function run() {
  const url = 'https://help.refrens.com/en/article/how-to-delete-a-form-in-refrens-24kj7g/';
  const loader = new CheerioWebBaseLoader(url);
  const rawDocs = await loader.load();

  rawDocs.forEach(doc => doc.metadata.source = url);

  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const splitDocs = await splitter.splitDocuments(rawDocs);

  console.log(` Loaded and split ${splitDocs.length} chunks`);

  // Upload to Qdrant
  await QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
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


  console.log(' Successfully uploaded to Qdrant Cloud!');
}

run().catch(console.error);
