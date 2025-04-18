// trial2.ts
// ✅ LangChainJS RAG with:
// - Mistral-7B-Instruct-v0.2 (text generation)
// - MiniLM (sentence-transformers/all-MiniLM-L6-v2) for embeddings
// - Qdrant for vector store
import 'dotenv/config';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { QdrantVectorStore } from '@langchain/community/vectorstores/qdrant';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence, RunnableMap } from '@langchain/core/runnables';
import fetch from 'node-fetch'; // required for custom LLM call
const urls = [
    'https://en.wikipedia.org/wiki/Artificial_intelligence',
    'https://en.wikipedia.org/wiki/Machine_learning'
];
async function loadWebDocuments(urlList) {
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
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: 'sentence-transformers/all-MiniLM-L6-v2'
});
const mistralPrompt = PromptTemplate.fromTemplate(`
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
`);
// 🔧 Mistral custom LLM using fetch
async function callMistralLocally(prompt) {
    const payload = {
        model: 'mistral',
        prompt: prompt,
        stream: false
    };
    console.log('\n🧪 Sending to Ollama:\n', JSON.stringify(payload, null, 2)); // log the payload
    const res = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const data = await res.json();
    console.log('\n🧪 RAW response from Mistral Ollama:\n', data); // log raw response
    return data?.response || '❌ No response from Mistral';
}
(async () => {
    const documents = await loadWebDocuments(urls);
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const splitDocs = await splitter.splitDocuments(documents);
    const vectorstore = await QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
        url: 'http://localhost:6333',
        collectionName: 'mistral_rag_demo'
    });
    const retriever = vectorstore.asRetriever();
    const ragChain = RunnableSequence.from([
        RunnableMap.from({
            context: async (input) => {
                const docs = await retriever.invoke(input.question);
                return docs.map((d) => d.pageContent).join('\n\n');
            },
            question: (input) => input.question
        }),
        prompt,
        {
            invoke: async (input) => {
                const finalPrompt = typeof input === 'string' ? input : input?.invoke || input?.toString?.();
                return await callMistralLocally(finalPrompt);
            }
        }
    ]);
    const response = await ragChain.invoke({
        question: 'What is artificial intelligence used for?'
    });
    console.log('\n Mistral RAG Answer:\n', response);
})();
