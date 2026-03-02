const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

function loadYAML(filePath) {
  return yaml.load(fs.readFileSync(filePath, 'utf8'));
}

exports.handler = async function(event, context) {
  try {
    const dataDir = path.join(__dirname, '..', '..', 'benchmarks', 'friendbench', 'data');
    const modelsPath = path.join(dataDir, 'models.yaml');
    const quotesPath = path.join(dataDir, 'quotes.yaml');

    if (!fs.existsSync(modelsPath)) {
      throw new Error(`Models file not found at: ${modelsPath}`);
    }

    const models = loadYAML(modelsPath);

    const scoresPath = path.join(dataDir, 'scores.yaml');
    if (fs.existsSync(scoresPath)) {
      const scores = loadYAML(scoresPath) || {};
      for (const m of models) {
        if (m.name in scores) {
          m.score = scores[m.name];
        }
      }
    }

    const result = { models };

    if (fs.existsSync(quotesPath)) {
      result.quotes = loadYAML(quotesPath);
    }

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store'
      },
      body: JSON.stringify(result)
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
