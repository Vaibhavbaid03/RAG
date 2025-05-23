import 'dotenv/config';
import fetch from 'node-fetch';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { QdrantVectorStore } from '@langchain/community/vectorstores/qdrant';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence, RunnableMap } from '@langchain/core/runnables';

const urls = [
  'https://en.wikipedia.org/wiki/Artificial_intelligence',
  'https://en.wikipedia.org/wiki/Machine_learning'
];

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
  const docs = await loadWebDocuments(urls);
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const splitDocs = await splitter.splitDocuments(docs);

  const vectorstore = await QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
    url: 'http://localhost:6333',
    collectionName: 'mistral_rag_ollama'
  });

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

  console.log('\n\u2728 Mistral (via Ollama) Answer:\n', response);
})();
