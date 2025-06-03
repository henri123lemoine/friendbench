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

    console.log('Attempting to read files from:');
    console.log('Models path:', modelsPath);
    console.log('Quotes path:', quotesPath);
    console.log('Current working directory:', process.cwd());
    console.log('__dirname:', __dirname);

    // Check if files exist
    if (!fs.existsSync(modelsPath)) {
      throw new Error(`Models file not found at: ${modelsPath}`);
    }
    if (!fs.existsSync(quotesPath)) {
      throw new Error(`Quotes file not found at: ${quotesPath}`);
    }

    let models = parseJSONC(modelsPath);
    const quotes = parseJSONC(quotesPath);

    // Optional filtering via ?models=a,b,c
    const selected = event.queryStringParameters?.models;
    if (selected) {
      const allow = selected.split(',').map(s => s.trim());
      models = models.filter(m => allow.includes(m.name));
      console.log(`Filtered models to: ${allow.join(', ')}`);
    }

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store'
      },
      body: JSON.stringify({ models, quotes })
    };
  } catch (err) {
    console.error('Error in get-data function:', err.message);
    return {
      statusCode: 500,
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ error: err.message })
    };
  }
};