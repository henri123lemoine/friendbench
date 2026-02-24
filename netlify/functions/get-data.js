const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

function loadYAML(filePath) {
  return yaml.load(fs.readFileSync(filePath, 'utf8'));
}

exports.handler = async function(event, context) {
  try {
    let modelsPath = path.join(__dirname, 'data', 'models.yaml');
    let quotesPath = path.join(__dirname, 'data', 'quotes.yaml');
    if (!fs.existsSync(modelsPath) || !fs.existsSync(quotesPath)) {
      modelsPath = path.join(__dirname, '..', '..', 'data', 'models.yaml');
      quotesPath = path.join(__dirname, '..', '..', 'data', 'quotes.yaml');
    }

    if (!fs.existsSync(modelsPath)) {
      throw new Error(`Models file not found at: ${modelsPath}`);
    }
    if (!fs.existsSync(quotesPath)) {
      throw new Error(`Quotes file not found at: ${quotesPath}`);
    }

    const models = loadYAML(modelsPath);
    const quotes = loadYAML(quotesPath);

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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: err.message })
    };
  }
};
