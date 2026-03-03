const fs = require('fs');

const required = [
  'benchmarks/friendbench/data/models.yaml',
  'benchmarks/friendbench/data/scores.yaml',
  'benchmarks/friendbench/data/quotes.yaml',
];

let ok = true;
for (const f of required) {
  if (!fs.existsSync(f)) {
    console.error(`Missing required file: ${f}`);
    ok = false;
  }
}

if (!ok) process.exit(1);
