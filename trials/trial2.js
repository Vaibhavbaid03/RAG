// trial1.ts
import "dotenv/config";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
/**
 * This test checks if your Hugging Face API key works for inference-based embeddings.
 * It uses the hosted model: sentence-transformers/all-MiniLM-L6-v2
 */
const run = async () => {
    const embeddings = new HuggingFaceInferenceEmbeddings({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        apiKey: process.env.HUGGINGFACE_API_KEY,
    });
    const input = ["Artificial intelligence is transforming technology."];
    try {
        const result = await embeddings.embedDocuments(input);
        console.log("✅ Hugging Face API Embedding succeeded!");
        console.log("📏 Vector length:", result[0].length);
        console.log("🔹 First 10 dims:", result[0].slice(0, 10));
    }
    catch (error) {
        console.error("❌ API Embedding failed:", error);
    }
};
run();
