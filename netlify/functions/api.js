const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

exports.handler = async function(event) {
  try {
    let modelsPath = path.join(__dirname, 'data', 'models.yaml');
    if (!fs.existsSync(modelsPath)) {
      modelsPath = path.join(__dirname, '..', '..', 'data', 'models.yaml');
    }

    const models = yaml.load(fs.readFileSync(modelsPath, 'utf8'));

    const showAll = event.queryStringParameters?.showAll === 'true';
    const filtered = showAll ? models : models.filter(m => !m.hidden);

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'public, max-age=300'
      },
      body: JSON.stringify({
        name: 'FriendBench',
        description: 'A systematic assessment of amicability in large language models',
        version: 1,
        models: filtered.map(({ name, provider, score }) => ({ name, provider, score }))
      })
    };
  } catch (err) {
    console.error('Error in api function:', err.message);
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: err.message })
    };
  }
};
