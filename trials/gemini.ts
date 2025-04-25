import 'dotenv/config';
import fetch from 'node-fetch';

const url = `https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`;


const response = await fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    contents: [
      {
        parts: [{ text: "Hey Gemini! Are you responding?" }],
      },
    ],
  }),
});

const data = await response.json();

// ✅ Print entire response for debug
console.log("🌐 Raw Gemini response:\n");
console.dir(data, { depth: null });

// ✅ Try parsing it ONLY if no error
if (data?.candidates?.[0]?.content?.parts?.[0]?.text) {
  console.log("\n✅ Gemini says:", data.candidates[0].content.parts[0].text);
} else {
  console.log("\n⚠️ Gemini returned no valid text.");
}
/*

const modelsResponse = await fetch(`https://generativelanguage.googleapis.com/v1/models?key=${process.env.GEMINI_API_KEY}`);
const models = await modelsResponse.json();
console.log("📦 Available models:", models);
*/