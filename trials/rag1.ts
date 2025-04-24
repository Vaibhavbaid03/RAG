// ragHandler.ts
import 'dotenv/config';
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RunnableSequence } from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts";
import * as fs from "fs/promises";
import { setTimeout } from "timers/promises";

const COLLECTION_NAME = "refrens_help_docs";
const CSV_FILE = "Refrens_Help_URLs.csv";
const BATCH_SIZE = 5;
const DELAY_MS = 3000;

const embeddingModel = new OpenAIEmbeddings({
  modelName: "text-embedding-3-small",
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const llm = new ChatGoogleGenerativeAI({
  modelName: "gemini-pro",
  apiKey: process.env.GEMINI_API_KEY,
  temperature: 0.3,
});

async function loadDocsFromCSV() {
  const urls = (await fs.readFile(CSV_FILE, "utf-8")).split("\n").filter(Boolean);
  const docs = [];

  for (let i = 0; i < urls.length; i += BATCH_SIZE) {
    const batch = urls.slice(i, i + BATCH_SIZE);
    for (const url of batch) {
      const loader = new CheerioWebBaseLoader(url);
      const batchDocs = await loader.load();
      docs.push(...batchDocs);
    }
    await setTimeout(DELAY_MS);
  }

  return docs;
}

export async function embedAndStore() {
  const docs = await loadDocsFromCSV();
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const splitDocs = await splitter.splitDocuments(docs);

  await Chroma.fromDocuments(splitDocs, embeddingModel, {
    collectionName: COLLECTION_NAME,
    url: process.env.CHROMA_URL,
  });

  console.log("✅ Docs embedded and stored in Azure-hosted ChromaDB");
}

export async function answerQuestion(query: string) {
  const vectorStore = await Chroma.fromExistingCollection(embeddingModel, {
    collectionName: COLLECTION_NAME,
    url: process.env.CHROMA_URL,
  });

  const retriever = vectorStore.asRetriever();

  const prompt = PromptTemplate.fromTemplate(`
You are a helpful assistant. Use the following context to answer the user’s question.

Context:
{context}

Question: {question}
`);

  const chain = RunnableSequence.from([
    {
      context: async (q) => {
        const docs = await retriever.getRelevantDocuments(q);
        return docs.map(d => d.pageContent).join("\n\n");
      },
      question: (input) => input,
    },
    prompt,
    llm,
  ]);

  const response = await chain.invoke(query);
  console.log("💬 Gemini says:", response);
  return response;
}
