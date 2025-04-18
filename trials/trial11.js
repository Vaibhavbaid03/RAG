"use strict";
// trial11.ts
//   Qdrant Cloud test
// Just connect and upload a single page → print how many chunks were embedded
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
require("dotenv/config");
var cheerio_1 = require("@langchain/community/document_loaders/web/cheerio");
var text_splitter_1 = require("langchain/text_splitter");
var hf_1 = require("@langchain/community/embeddings/hf");
var qdrant_1 = require("@langchain/community/vectorstores/qdrant");
var js_client_rest_1 = require("@qdrant/js-client-rest");
var QDRANT_URL = 'https://7be21947-a45c-49a8-a3ae-6682f86afdb3.eu-west-1-0.aws.cloud.qdrant.io';
var COLLECTION_NAME = 'mistral_rag_ollama_test';
var client = new js_client_rest_1.QdrantClient({
    url: QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
});
var embeddings = new hf_1.HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: 'sentence-transformers/all-MiniLM-L6-v2',
});
function run() {
    return __awaiter(this, void 0, void 0, function () {
        var url, loader, rawDocs, splitter, splitDocs;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    url = 'https://help.refrens.com/en/article/how-to-delete-a-form-in-refrens-24kj7g/';
                    loader = new cheerio_1.CheerioWebBaseLoader(url);
                    return [4 /*yield*/, loader.load()];
                case 1:
                    rawDocs = _a.sent();
                    rawDocs.forEach(function (doc) { return doc.metadata.source = url; });
                    splitter = new text_splitter_1.RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
                    return [4 /*yield*/, splitter.splitDocuments(rawDocs)];
                case 2:
                    splitDocs = _a.sent();
                    console.log(" Loaded and split ".concat(splitDocs.length, " chunks"));
                    // Upload to Qdrant
                    return [4 /*yield*/, qdrant_1.QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
                            url: QDRANT_URL,
                            apiKey: process.env.QDRANT_API_KEY,
                            collectionName: COLLECTION_NAME,
                            collectionOptions: {
                                onDiskPayload: true,
                                hnswConfig: {
                                    m: 16,
                                    efConstruct: 100
                                }
                            }
                        })];
                case 3:
                    // Upload to Qdrant
                    _a.sent();
                    console.log(' Successfully uploaded to Qdrant Cloud!');
                    return [2 /*return*/];
            }
        });
    });
}
run().catch(console.error);
