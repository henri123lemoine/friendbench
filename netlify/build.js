const fs = require('fs');
const path = require('path');

const bench = process.argv[2];
if (!bench) {
  console.error('Usage: node build.js <benchmark>');
  process.exit(1);
}

const root = path.join(__dirname, '..');
const benchDir = path.join(root, 'benchmarks', bench);
const yaml = require(require.resolve('js-yaml', { paths: [process.cwd(), benchDir] }));
const dataDir = path.join(benchDir, 'data');
const frontendDir = path.join(root, 'benchmarks', bench, 'frontend');

const models = yaml.load(fs.readFileSync(path.join(dataDir, 'models.yaml'), 'utf8'));

const scoresPath = path.join(dataDir, 'scores.yaml');
if (fs.existsSync(scoresPath)) {
  const scores = yaml.load(fs.readFileSync(scoresPath, 'utf8')) || {};
  for (const m of models) {
    if (m.name in scores) m.score = scores[m.name];
  }
}

const result = { models };

const quotesPath = path.join(dataDir, 'quotes.yaml');
if (fs.existsSync(quotesPath)) {
  result.quotes = yaml.load(fs.readFileSync(quotesPath, 'utf8'));
}

fs.writeFileSync(path.join(frontendDir, 'data.json'), JSON.stringify(result));
console.log(`Built ${path.join(frontendDir, 'data.json')}`);
