// ollama for embeddings.
// https://help.refrens.com/sitemap.xml is provided instead of 120 sub-URLs

import 'dotenv/config';
import fetch from 'node-fetch';
import { SitemapLoader } from '@langchain/community/document_loaders/web/sitemap';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { QdrantVectorStore } from '@langchain/community/vectorstores/qdrant';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence, RunnableMap } from '@langchain/core/runnables';
import { QdrantClient } from '@qdrant/js-client-rest';
import { Document } from 'langchain/document';
import { SitemapLoader } from '@langchain/community/document_loaders/web/sitemap';


const QDRANT_URL = 'http://localhost:6333';
const COLLECTION_NAME = 'refrens_help_articles';
const SITEMAP_URL = 'https://help.refrens.com/sitemap.xml';
const client = new QdrantClient({ url: QDRANT_URL });

const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text' });

async function collectionExists(): Promise<boolean> {
  try {
    const collections = await client.getCollections();
    return collections.collections.some(c => c.name === COLLECTION_NAME);
  } catch (err) {
    console.error('Error fetching collections:', err);
    return false;
  }
}

async function loadAllDocsFromSitemap(sitemapUrl: string): Promise<Document[]> {
  const loader = new SitemapLoader(sitemapUrl, {
    allowUrlPatterns: ['/en/article/']
  });

  const docs = await loader.load();

  return docs.map(doc => {
    doc.metadata.source = doc.metadata.source || doc.metadata.loc;
    return doc;
  });
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
Include a clickable link to the article if relevant.

Context:
{context}

Question:
{question}

Answer:
`);

(async () => {
  let vectorstore: QdrantVectorStore;

  const exists = await collectionExists();

  if (!exists) {
    console.log(' Collection not found. Loading and chunking sitemap articles...');
    const docs = await loadAllDocsFromSitemap(SITEMAP_URL);
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const splitDocs = await splitter.splitDocuments(docs);

    vectorstore = await QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
      url: QDRANT_URL,
      collectionName: COLLECTION_NAME
    });
  } else {
    console.log(` Qdrant collection '${COLLECTION_NAME}' already exists.`);
    vectorstore = await QdrantVectorStore.fromExistingCollection(embeddings, {
      url: QDRANT_URL,
      collectionName: COLLECTION_NAME
    });
  }

  const retriever = vectorstore.asRetriever();

  const ragChain = RunnableSequence.from([
    RunnableMap.from({
      context: async (input: { question: string }) => {
        const docs = await retriever.invoke(input.question);
        const context = docs.map(doc => `- ${doc.pageContent}\n[Source](${doc.metadata.source})`).join('\n\n');
        return context;
      },
      question: (input: { question: string }) => input.question,
    }),
    ragPrompt,
    {
      invoke: async (input: any) => {
        const finalPrompt = typeof input === 'string' ? input : input?.toString?.();
        return await callMistralLocally(finalPrompt);
      }
    }
  ]);

  const response = await ragChain.invoke({
    question: 'How do I delete my Refrens business?'
  });

  console.log('\n Answer from Mistral:\n', response);
})();
