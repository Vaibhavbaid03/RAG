// trial1.ts
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
// 🧪 Sample documents to embed
const sampleTexts = [
    "Artificial intelligence is revolutionizing industries.",
    "Machine learning is a subset of AI focused on pattern recognition."
];
async function testHuggingFaceEmbeddings() {
    try {
        // ✅ Load the MiniLM model from Hugging Face
        const embeddings = new HuggingFaceTransformersEmbeddings({
            modelName: "Xenova/all-MiniLM-L6-v2", // use Xenova prefix for web-compatible model
        });
        // 📌 Embed all documents
        const vectors = await embeddings.embedDocuments(sampleTexts);
        // 🎉 Log the results
        console.log("✅ Hugging Face embeddings generated successfully!");
        console.log(`🔢 Vector dimensions: ${vectors[0].length}`);
        console.log("🔍 First vector (truncated):", vectors[0].slice(0, 10));
    }
    catch (err) {
        console.error("❌ Hugging Face embedding error:", err);
    }
}
testHuggingFaceEmbeddings();
