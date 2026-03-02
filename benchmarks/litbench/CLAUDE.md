# LitBench

Standalone script (not inspect_ai). Tests whether a model thinks literature has gotten better or worse over time.

Each round: nominates 5 best works per decade, then does pairwise head-to-head comparisons with randomized presentation order. Uses Anthropic SDK directly with adaptive thinking.

Recency score > 0.5 = model thinks newer literature is better.
