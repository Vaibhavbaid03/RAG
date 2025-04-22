import { Controller, POST } from 'fastify-decorators';
import { FastifyRequest, FastifyReply } from 'fastify';
import ragService from '../services/ragService';

interface RAGRequestBody {
  question: string;
}

@Controller('/rag')
export default class RAGController {
    constructor() {
        console.log(' RAGController loaded'); 
      }
  @POST('/')
  async handleQuery(
    req: FastifyRequest<{ Body: RAGRequestBody }>,
    reply: FastifyReply
  ) {
    const { question } = req.body;
    const result = await ragService(question);
    reply.send({ result });
  }
}
