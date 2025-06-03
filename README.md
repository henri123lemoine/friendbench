# FriendBench

The official website of the FriendBench™ AI model friendliness benchmark.

Available at [friendbench.ai](https://friendbench.ai).

The model and quote data live in `.jsonc` files under `data/` so you can keep
inline `// comments` or temporarily comment out entries. They are served at
runtime by a Netlify Function (`/.netlify/functions/get-data`).

To display only certain models you can append
`?models=Model+One,Model+Two` to the API URL. The front-end passes through any
query string automatically.

## License

© High Taste Testers 2025
