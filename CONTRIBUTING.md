# Contributing

## Adding a new benchmark

1. Create `benchmarks/<name>/` with:
   - `tasks.py` — the `@task` function (name must match directory)
   - `data/models.yaml` — model definitions (presence of this file is what makes the benchmark discoverable)
   - `data/questions.yaml` — benchmark questions
   - `frontend/index.html` — single-file frontend (vanilla JS, inline CSS)

2. The frontend should fetch data with `fetch('data.json')`. At build time, `netlify/build.js` generates this from the YAML files.

3. Optionally add `data/scores.yaml` and/or `data/quotes.yaml` — the build script merges these into `data.json` automatically.

## Local dev

```bash
cd benchmarks/friendbench
npm install
npm run dev
```

This builds `data.json` from the YAML data files and starts a local server.

## Deployment

Each benchmark is deployed as its own Netlify site, all pointing at the same repo. `netlify/build.js` reads a benchmark's YAML data files and writes a static `data.json` into its `frontend/` directory.

### Deploying a new benchmark

1. Add these files to `benchmarks/<name>/`:

   `netlify.toml`:
   ```toml
   [build]
     command = "node ../../netlify/build.js <name>"
     publish = "frontend"

   [dev]
     publish = "frontend"
     framework = "#static"
   ```

   `package.json`:
   ```json
   {
     "private": true,
     "scripts": {
       "dev": "node ../../netlify/build.js <name> && netlify dev"
     },
     "dependencies": {
       "js-yaml": "^4.1.0"
     }
   }
   ```

2. Create a new site in the Netlify UI:
   - Link it to this repo
   - Set **Base directory** to `benchmarks/<name>`
   - The build command and publish directory are read from the `netlify.toml` automatically

That's it. Netlify runs `npm install` in the base directory, then runs the build command.
