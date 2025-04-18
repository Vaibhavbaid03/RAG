// Load environment variables from .env file (e.g. GOOGLE_API_KEY)
import "dotenv/config";

//  Gemini LLM + Embedding model
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings
} from "@langchain/google-genai";

//  Qdrant vector store from LangChain community
import { QdrantVectorStore } from "@langchain/community/vectorstores/qdrant";

//  Load & parse web pages
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

// ️ Chunk documents into smaller segments
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

//  Prompt construction
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatHuggingFaceInference } from "@langchain/community/chat_models/hf";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";

//  Build RAG pipeline using modular runnables
import {
  RunnableMap,
  RunnableSequence
} from "@langchain/core/runnables";

//  Extract plain text from model output
import { StringOutputParser } from "@langchain/core/output_parsers";

//  Confirm Cheerio loader is ready
console.log(" Using CheerioWebBaseLoader:", CheerioWebBaseLoader);

//  Gemini Chat LLM for answering
const llm = new ChatHuggingFaceInference({
  model: "mistralai/Mistral-7B-Instruct-v0.2",
  apiKey: process.env.HUGGINGFACE_API_KEY,
});

// Gemini Embedding model for vectorization
const embeddings = new HuggingFaceInferenceEmbeddings({
  model: "sentence-transformers/all-MiniLM-L6-v2",
  apiKey: process.env.HUGGINGFACE_API_KEY,
});


// URLs to use as external knowledge source
const urls = [
  "https://en.wikipedia.org/wiki/Artificial_intelligence",
  "https://en.wikipedia.org/wiki/Machine_learning"
];

//  Load content from the web and log progress
async function loadWebDocuments(urlList: string[]) {
  const allDocs = [];
  for (const url of urlList) {
    try {
      console.log(` Loading URL: ${url}`);
      const loader = new CheerioWebBaseLoader(url);
      const docs = await loader.load();
      console.log(`✅ Loaded: ${url} (${docs.length} docs)`);
      allDocs.push(...docs);
    } catch (err) {
      console.error(`Failed to load: ${url}`, err); //  Interpolation is fine
    }
  }
  return allDocs; //  This is valid and NOT commented
}

//  Run entire RAG pipeline
(async () => {
  try {
    // Step 1: Load & split documents
    const documents = await loadWebDocuments(urls);
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const splitDocs = await splitter.splitDocuments(documents);

    // Step 2: Embed & store in Qdrant

    const vectorstore = await QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
      url: "http://localhost:6333", // Local Qdrant instance
      collectionName: "gemini_rag_urls",
      clientConfig: {
        compatibility: { check: false }, // ✅ Optional: disable version check
      }
    });

    const retriever = vectorstore.asRetriever();

    // Step 3: RAG prompt setup
const qaPrompt = PromptTemplate.fromTemplate(`
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}
`); //  This line should have the closing backtick **before** the closing parenthesis


    // Step 4: Build the full RAG chain
    const ragChain = RunnableSequence.from([
      RunnableMap.from({
        context: async (input: { question: string }) => {
          const docs = await retriever.invoke(input.question);
          return docs.map(d => d.pageContent).join("\n\n");
        },
        question: (input: { question: string }) => input.question,
      }),
      qaPrompt,
      llm,
      new StringOutputParser()
    ]);

    // Step 5: Ask a question
    const response = await ragChain.invoke({
      question: "What is artificial intelligence used for?"
    });

    console.log(" Gemini (RAG) says:", response);

  } catch (err) {
    console.error(" Top-level error caught:", err);
  }
})();
