// hugging face for embeddings
// single url
import 'dotenv/config';
import fetch from 'node-fetch';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { QdrantVectorStore } from '@langchain/community/vectorstores/qdrant';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence, RunnableMap } from '@langchain/core/runnables';
import { QdrantClient } from '@qdrant/js-client-rest';
const COLLECTION_NAME = 'mistral_rag_ollama';
const QDRANT_URL = 'http://localhost:6333';
const urls = [
    'https://help.refrens.com/en/article/how-to-removedelete-your-refrens-business-1r2begj/'
];
const client = new QdrantClient({ url: QDRANT_URL });
const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: 'sentence-transformers/all-MiniLM-L6-v2',
});
async function checkCollectionExists() {
    try {
        const collections = await client.getCollections();
        return collections.collections.some(c => c.name === COLLECTION_NAME);
    }
    catch (err) {
        console.error('Error fetching collections:', err);
        return false;
    }
}
async function loadNewDocs(urls) {
    const docs = [];
    for (const url of urls) {
        const loader = new CheerioWebBaseLoader(url);
        const pageDocs = await loader.load();
        pageDocs.forEach(doc => doc.metadata.source = url);
        docs.push(...pageDocs);
    }
    return docs;
}
async function callMistralLocally(prompt) {
    const payload = {
        model: 'mistral',
        prompt,
        stream: false
    };
    console.log('\n\uD83D\uDCE4 Sending to Ollama:\n', JSON.stringify(payload, null, 2));
    const res = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const data = await res.json();
    console.log('\n\uD83D\uDCE5 RAW response from Mistral Ollama:\n', data);
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
(async () => {
    let vectorstore;
    const collectionExists = await checkCollectionExists();
    if (collectionExists) {
        console.log(`\u2705 Qdrant collection '${COLLECTION_NAME}' already exists. Skipping reload.`);
        vectorstore = await QdrantVectorStore.fromExistingCollection(embeddings, {
            url: QDRANT_URL,
            collectionName: COLLECTION_NAME
        });
        const newDocs = await loadNewDocs(urls);
        const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
        const splitDocs = await splitter.splitDocuments(newDocs);
        await vectorstore.addDocuments(splitDocs);
    }
    else {
        console.log('\u2B07\uFE0F Creating collection and loading docs...');
        const docs = await loadNewDocs(urls);
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
            context: async (input) => {
                const docs = await retriever.invoke(input.question);
                return docs.map(doc => `- ${doc.pageContent}\n(Source: ${doc.metadata?.source})`).join('\n\n');
            },
            question: (input) => input.question,
        }),
        ragPrompt,
        {
            invoke: async (input) => {
                const finalPrompt = typeof input === 'string' ? input : input?.toString?.();
                return await callMistralLocally(finalPrompt);
            }
        }
    ]);
    const response = await ragChain.invoke({
        question: 'how to delete business?'
    });
    console.log('\n\u2728 Mistral (via Ollama) Answer:\n', response);
})();
