import { QdrantClient } from "@qdrant/js-client-rest";
const client = new QdrantClient({ url: "http://localhost:6333" });
async function deleteCollection() {
    try {
        await client.deleteCollection("gemini_rag_urls");
        console.log("✅ Deleted collection: gemini_rag_urls");
    }
    catch (err) {
        console.error("❌ Failed to delete collection:", err);
    }
}
deleteCollection();
