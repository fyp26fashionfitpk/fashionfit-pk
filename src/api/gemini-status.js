import { getGeminiConfig } from './_gemini.js'

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*')
  const cfg = getGeminiConfig()

  return res.status(200).json({
    schema_version: 'v4.0',
    provider: 'gemini_optional',
    configured: cfg.configured,
    enabled: cfg.enabled,
    // Presence of an API key does NOT guarantee billing or runtime readiness.
    ready: false,
    readiness_note: cfg.enabled
      ? 'Explicitly enabled by operator; runtime availability depends on provider quota/billing.'
      : 'Disabled by default. Core try-on operates independently on the open V4 GPU backend.',
    model: cfg.model,
  })
}
