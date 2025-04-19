import Fastify from 'fastify';
import ragHandler from './ragHandler';

const fastify = Fastify();

fastify.post('/rag', async (request, reply) => {
  const body = request.body as { question: string };

  if (!body.question) {
    return reply.code(400).send({ error: 'Missing question in request body' });
  }

  const answer = await ragHandler(body.question);
  return { answer };
});

async function main() {
  await fastify.listen({ port: 3000 });
  console.log(' Fastify server running at http://localhost:3000');

  const response = await ragHandler("How to Customize Email Templates?");
  console.log(" Response from RAG:\n", response);
}

main().catch(console.error);
