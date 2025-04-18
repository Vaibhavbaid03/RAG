// ollama - embeddings
//single url
import 'dotenv/config';
import fetch from 'node-fetch';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { QdrantVectorStore } from '@langchain/community/vectorstores/qdrant';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence, RunnableMap } from '@langchain/core/runnables';
import { QdrantClient } from '@qdrant/js-client-rest';
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
const COLLECTION_NAME = 'mistral_rag_ollama';
const QDRANT_URL = 'http://localhost:6333';
const client = new QdrantClient({ url: QDRANT_URL });
const urls = [
    'https://help.refrens.com/en/article/how-to-removedelete-your-refrens-business-1r2begj/'
];
const embeddings = new OllamaEmbeddings({
    model: 'nomic-embed-text'
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
const ragPrompt = PromptTemplate.fromTemplate(`
You are a helpful assistant. Use the following context to answer the question.

If a source URL is provided, include a line at the end of your answer like:
"For more information, visit: [URL]"
If multiple sources exist, choose the most relevant one to include at the end of the answer.


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
        console.log(`✅ Qdrant collection '${COLLECTION_NAME}' already exists. Adding new docs.`);
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
        console.log(' Creating collection and loading docs...');
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
                const context = docs.map(doc => `- ${doc.pageContent}\n(Source: ${doc.metadata?.source})`).join('\n\n');
                return context;
            },
            question: (input) => input.question
        }),
        ragPrompt,
        {
            invoke: async (input) => {
                const finalPrompt = typeof input === 'string' ? input : input?.toString?.();
                return await callMistralLocally(finalPrompt);
            }
        }
    ]);
    const response = await ragChain.invoke({ question: 'how to delete business?' });
    console.log('\n Mistral (via Ollama) Answer:\n', response);
})();
