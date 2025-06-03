const fs = require('fs');
const path = require('path');

exports.handler = async function(event, context) {
  try {
    const modelsPath = path.join(__dirname, '..', '..', 'data', 'models.json');
    const quotesPath = path.join(__dirname, '..', '..', 'data', 'quotes.json');
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
    return { statusCode: 500, body: 'Server Error' };
  }
};
