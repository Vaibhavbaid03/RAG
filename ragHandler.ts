// ragHandler.ts — Azure-hosted ChromaDB!
import 'dotenv/config';
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RunnableLambda } from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts";
import type { BaseRetriever } from "@langchain/core/retrievers";
import * as fs from "fs/promises";
import { setTimeout } from "timers/promises";
import cliProgress from "cli-progress";

const COLLECTION_NAME = "refrens_help_docs";
const CSV_FILE = "Refrens_Help_URLs.csv";
const BATCH_SIZE = 5;
const DELAY_MS = 3000;

const embeddingModel = new OpenAIEmbeddings({
  modelName: "text-embedding-3-small",
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const llm = new ChatGoogleGenerativeAI({
  model: "models/gemini-1.5-flash",
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
async function loadAndSplitDocs(batch: string[]) {
  const allDocs = [];

  for (const url of batch) {
    const loader = new CheerioWebBaseLoader(url);
    const docs = await loader.load();
    allDocs.push(...docs);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  return splitter.splitDocuments(allDocs);
}

//  Embed help articles into Azure-hosted ChromaDB
export async function embedAndStore() {
  const urls = (await fs.readFile(CSV_FILE, "utf-8")).split("\n").filter(Boolean);
  const totalBatches = Math.ceil(urls.length / BATCH_SIZE);

  const embeddingModel = new OpenAIEmbeddings({
    modelName: "text-embedding-3-small",
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const vectorStore = await Chroma.fromExistingCollection(embeddingModel, {
    collectionName: COLLECTION_NAME,
    url: process.env.CHROMA_URL,
  });

  const progressBar = new cliProgress.SingleBar({
    format: 'Embedding Progress |{bar}| {percentage}% | Batch {value}/{total} | ETA: {eta_formatted}',
    barCompleteChar: '█',
    barIncompleteChar: '░',
    hideCursor: true
  });

  progressBar.start(totalBatches, 0);
  const startTime = Date.now();

  for (let i = 0; i < urls.length; i += BATCH_SIZE) {
    const batchNum = i / BATCH_SIZE + 1;
    const batch = urls.slice(i, i + BATCH_SIZE);

    console.log(`\n Batch ${batchNum}/${totalBatches}`);

    console.log(' Loading and splitting...');
    const docs = await loadAndSplitDocs(batch);
    console.log(` Got ${docs.length} chunks.`);

    if (docs.length > 0) {
      console.log(' Embedding & storing...');
      await vectorStore.addDocuments(docs);
      console.log(' Done.');
    } else {
      console.warn(' Skipping empty batch.');
    }

    const elapsed = (Date.now() - startTime) / 1000;
    console.log(` Time elapsed: ${elapsed.toFixed(1)}s`);

    progressBar.increment();

    if (i + BATCH_SIZE < urls.length) {
      console.log(`  Waiting ${DELAY_MS}ms before next batch...`);
      await setTimeout(DELAY_MS);
    }
  }

  progressBar.stop();
  console.log('\n Embedding complete!');
}

//  Answer any query using Chroma + Gemini
export async function answerQuestion(query: string) {
  const vectorStore = await Chroma.fromExistingCollection(embeddingModel, {
    collectionName: COLLECTION_NAME,
    url: process.env.CHROMA_URL,
  });

  const retriever: BaseRetriever = vectorStore.asRetriever();

  const prompt = PromptTemplate.fromTemplate(`
Use the following context to answer the question.

Context:
{context}

Question: {question}
`);

const chain = RunnableLambda.from(async (input: string) => {
  const docs = await retriever.getRelevantDocuments(input);
  return {
    context: docs.map(d => d.pageContent).join("\n\n"),
    question: input,
  };
}).pipe(prompt).pipe(llm);

  const response = await chain.invoke(query);
  console.log(" Gemini says:", response);
  return response;
}
