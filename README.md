# FriendBench

The official website of the FriendBench™ AI model friendliness benchmark.

Available at [friendbench.ai](https://friendbench.ai).

The model and quote data live in `.jsonc` files under `data/` so you can keep
inline `// comments` or temporarily comment out entries. They are served at
runtime by a Netlify Function (`/.netlify/functions/get-data`). To display a
subset of models, pass a comma separated `models` query parameter in the URL.

## Local Development

To run the website locally and test it exactly as it appears online:

1. Install the Netlify CLI globally:

   ```bash
   npm install -g netlify-cli
   ```

2. Run the local development server:
   ```bash
   netlify dev
   ```

This will start a local server (typically at `http://localhost:8888`) that serves both the static frontend files and runs the Netlify functions locally, giving you the exact same experience as the deployed site.
