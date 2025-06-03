const fs = require('fs');
const path = require('path');

exports.handler = async function(event, context) {
  try {
    const modelsPath = path.join(__dirname, '..', '..', 'data', 'models.json');
    const quotesPath = path.join(__dirname, '..', '..', 'data', 'quotes.json');

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

    const models = JSON.parse(fs.readFileSync(modelsPath, 'utf8'));
    const quotes = JSON.parse(fs.readFileSync(quotesPath, 'utf8'));

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
