// FriendBench model data
const modelData = [
  { name: "Sonnet 4", score: 78 },
  { name: "Opus 4", score: 80 },
  { name: "Sonnet 3.7", score: 64 },
  { name: "Sonnet 3.6", score: 78 },
  { name: "Opus 3", score: 65 },
  { name: "Grok 3", score: 51 },
  { name: "GPT-4.5", score: 69 },
  { name: "GPT-4o\n(old)", score: 34 },
  { name: "GPT-4o\n(latest)", score: 57 },
  { name: "o3-mini\n(high)", score: 26 },
  { name: "DeepSeek-R1", score: 22 },
  { name: "Gemini\n2.5 Pro", score: 57 },
];

// Quotes data
const quotesData = [
  {
    text: "If Claude was a human it would be the best human",
    author: "Zainab",
    models: ["Sonnet 3.6"],
    showModelInline: true,
  },
  {
    text: "Grok also gives big model smell and has become my daily driver model, also a pretty pleasant personality, definitely definitely worth trying - just a bit immature/overly glossy vibes sometimes, whereas 4.5 is much more real",
    author: "Andrew",
    models: ["Grok 3", "GPT-4.5"],
  },
  {
    text: "4.5 is very very friend",
    author: "[redacted]",
    models: ["GPT-4.5"],
  },
  {
    text: "yeah xai cooked on this one",
    author: "Thomas",
    models: ["Grok 3"],
    showModelInline: true,
  },
  {
    text: "Grok 3 is good. I change my mind about GPT-4.5. Kinda mid. Claude 3.6 still wins on FriendBench, even above 3.7",
    author: "Henri",
    models: ["Grok 3", "GPT-4.5", "Sonnet 3.6", "Sonnet 3.7"],
  },
  {
    text: "yeah I am surprisingly not enchanted with 3.7",
    author: "Andrew",
    models: ["Sonnet 3.7"],
  },
  // {
  //   text: "I think the post-training may have been rushed on this one",
  //   author: "Henri",
  //   models: ["Sonnet 3.7"],
  //   showModelInline: true
  // },
  // {
  //   text: "Arguably current 4o is fairly decent",
  //   author: "Andrew",
  //   models: ["GPT-4o (latest)"]
  // },
  // {
  //   text: "claude knows it's a dream, and consensually plunges into the flux of samsara, and goes all the way in, maintaining lucidity",
  //   author: "Janus",
  //   models: ["Opus 3"],
  //   showModelInline: true
  // }
];

/*
# Anthropic
claude-3-7-sonnet-20250219
claude-3-5-sonnet-20241022
claude-3-5-haiku-20241022
claude-3-5-sonnet-20240620
claude-3-haiku-20240307
claude-3-opus-20240229
claude-3-sonnet-20240229
claude-2.1
claude-2.0


# OpenAI
## GPT-4o
gpt-4o-2024-11-20
gpt-4o-audio-preview
gpt-4o-2024-05-13
gpt-4o-search-preview-2025-03-11
gpt-4o-search-preview
gpt-4o-realtime-preview-2024-12-17
gpt-4o-realtime-preview
gpt-4o-transcribe
gpt-4o-2024-08-06
gpt-4o-realtime-preview-2024-10-01
gpt-4o-audio-preview-2024-12-17
gpt-4o-audio-preview-2024-10-01
## GPT-4o-mini
gpt-4o-mini-transcribe
gpt-4o-mini-2024-07-18
gpt-4o-mini
gpt-4o-mini-search-preview
gpt-4o-mini-search-preview-2025-03-11
gpt-4o-mini-tts
gpt-4o-mini-realtime-preview-2024-12-17
gpt-4o-mini-audio-preview-2024-12-17
gpt-4o-mini-realtime-preview
gpt-4o-mini-audio-preview
## 4.5
gpt-4.5-preview
gpt-4.5-preview-2025-02-27
## o-model
o3-mini-2025-01-31
o3-2025-04-16
o4-mini-2025-04-16
o1-2024-12-17
o3-mini
o1-mini
o1-mini-2024-09-12
o3
o4-mini
o1-pro-2025-03-19
o1
o1-pro
o1-preview
o1-preview-2024-09-12

# DeepSeek
DeepSeek-R1
Deepseek-R1 (new)

# Google Deepmind
Gemini 2.5 Pro
Gemini 2.0 Flash
*/
