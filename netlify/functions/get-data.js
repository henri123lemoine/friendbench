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
    // When deployed on Netlify the function bundle is executed from the
    // function's directory (e.g. `/var/task`). The `data` folder is packaged
    // alongside the function (see `netlify.toml`). When running `netlify dev`
    // locally, however, the working directory is the project root. We resolve
    // the paths accordingly by first checking for the files relative to
    // `__dirname` and falling back to the project root.
    let modelsPath = path.join(__dirname, 'data', 'models.jsonc');
    let quotesPath = path.join(__dirname, 'data', 'quotes.jsonc');
    if (!fs.existsSync(modelsPath) || !fs.existsSync(quotesPath)) {
      modelsPath = path.join(__dirname, '..', '..', 'data', 'models.jsonc');
      quotesPath = path.join(__dirname, '..', '..', 'data', 'quotes.jsonc');
    }
    const models = parseJSONC(modelsPath);
    const quotes = parseJSONC(quotesPath);

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
