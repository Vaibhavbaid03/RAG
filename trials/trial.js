import { GoogleGenerativeAI } from "@google/generative-ai";
import "dotenv/config";
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
(async () => {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });
    const result = await model.generateContent("What is artificial intelligence?");
    const response = await result.response;
    console.log("✅ Gemini says:", response.text());
})();
