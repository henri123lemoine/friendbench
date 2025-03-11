# FriendBench

A simple, static website displaying the FriendBench™ AI model scores chart.

## Overview

This project provides a minimalist website that displays a bar chart comparing the "friendliness" scores of various AI models. The chart is generated dynamically based on data you can easily modify.

## Structure

- `frontend/index.html` - The main HTML page
- `frontend/styles.css` - Minimal CSS styling
- `frontend/data.js` - **Model data** (edit this file to modify chart data)
- `frontend/script.js` - JavaScript for generating the SVG chart
- `netlify.toml` - Configuration for Netlify deployment

## Modifying the Chart Data

To update the chart with different models or scores, simply edit the `frontend/data.js` file.

The data structure is an array of objects, where each object represents a model with the following properties:

- `name` - The name of the model
- `score` - The friendliness score (0-100)
- `color` - The color for the bar (hex code)

Example:

```javascript
const modelData = [
  { name: "Model Name", score: 85, color: "#9370DB" },
  // Add more models...
];
```

## Deployment to Netlify

### Option 1: Deploy with Netlify CLI

1. Install Netlify CLI: `npm install -g netlify-cli`
2. Run `netlify deploy` from the project root

### Option 2: Deploy from the Netlify Dashboard

1. Go to [app.netlify.com](https://app.netlify.com)
2. Drag and drop the `frontend` folder or connect to your Git repository
3. Netlify will automatically detect the `netlify.toml` configuration

## Local Development

To run the site locally, you can use any simple HTTP server. For example:

```bash
# Using Python
cd frontend
python -m http.server

# Or using Node.js' http-server
npx http-server frontend
```

Then visit `http://localhost:8000` in your browser.

## License

© High Taste Testers 2025
