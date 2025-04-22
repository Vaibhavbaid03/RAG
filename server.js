// server.ts
import Fastify from 'fastify';
import ragHandler from './ragHandler';
const fastify = Fastify();
fastify.post('/rag', async (request, reply) => {
    const body = request.body;
    if (!body.question) {
        return reply.code(400).send({ error: 'Missing question in request body' });
    }
    const answer = await ragHandler(body.question); 
    return { answer };
});
fastify.listen({ port: 3000 }, (err, address) => {
    if (err)
        throw err;
    console.log(` Fastify server running at ${address}`);
});
