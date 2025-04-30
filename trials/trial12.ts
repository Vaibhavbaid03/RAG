import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OpenAIEmbeddings } from "@langchain/openai";
import 'dotenv/config';

async function testChromaConnection() {
  const embeddingModel = new OpenAIEmbeddings({
    modelName: "text-embedding-3-small",
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  try {
    const vectorStore = await Chroma.fromExistingCollection(embeddingModel, {
      collectionName: "refrens_help_docs", // Change if needed
      url: process.env.CHROMA_URL,
    });

    console.log("✅ Successfully connected to Chroma and accessed collection!");
  } catch (err) {
    console.error("❌ Failed to connect Chroma:", err);
  }
}

testChromaConnection();
