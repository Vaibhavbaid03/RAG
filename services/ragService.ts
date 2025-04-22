import ragHandler from '../ragHandler';

export default async function ragService(question: string): Promise<string> {
  try {
    return await ragHandler(question);
  } catch (err: any) {
    console.error('RAG Error:', err);
    return 'Error: Could not generate response.';
  }
}