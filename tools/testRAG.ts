import { answerQuestion } from "../ragHandler.js";

const result = await answerQuestion("How can I enable inventory in Refrens?");
console.log("🧠 RAG Answer:", result);