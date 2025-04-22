import 'reflect-metadata';
import { bootstrap } from 'fastify-decorators';
import Fastify from 'fastify';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const fastify = Fastify();

fastify.register(bootstrap, {
  directory: join(__dirname, 'controllers'),
  mask: /\.ts$/,
});

fastify.listen({ port: 3000 }, (err) => {
  if (err) throw err;
  console.log(' Fastify server running at http://localhost:3000');
});
