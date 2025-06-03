const fs = require('fs');
const path = require('path');

function parseJSONC(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8');
  const noBlock = raw.replace(/\/\*[\s\S]*?\*\//g, '');
  const noLine = noBlock
    .split('\n')
    .map(line => line.replace(/^\s*\/\/.*$/, ''))
    .join('\n');
  return JSON.parse(noLine);
}

exports.handler = async function(event, context) {
  try {
    const modelsPath = path.join(__dirname, '..', '..', 'data', 'models.jsonc');
    const quotesPath = path.join(__dirname, '..', '..', 'data', 'quotes.jsonc');
    const models = parseJSONC(modelsPath);
    const quotes = parseJSONC(quotesPath);
    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store'
      },
      body: JSON.stringify({ models, quotes })
    };
  } catch (err) {
    return { statusCode: 500, body: 'Server Error' };
  }
};
