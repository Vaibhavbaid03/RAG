import { answerQuestion } from "../ragHandler.js";

export default async function ragService(question: string): Promise<string> {
  try {
    const result = await answerQuestion(question);
    return result?.content || "No content returned."; 
  } catch (err: any) {
    console.error(" RAG Error:", err); 
    return "Error: Could not generate response.";
  }
}