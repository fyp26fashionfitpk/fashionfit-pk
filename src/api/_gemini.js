/**
 * FashionFit-PK: Optional Gemini Provider Helper (Disabled by Default)
 *
 * Operating rules:
 *  - Core try-on workflows MUST operate without Gemini or paid billing.
 *  - Gemini endpoints only run when the operator explicitly sets
 *    ENABLE_OPTIONAL_GEMINI=true AND configures a server-side GEMINI_API_KEY.
 *  - Single configured key is used (no multi-account/key rotation to bypass limits).
 *  - HTTP errors are classified accurately (never misreporting 400/5xx as 429).
 */

export function getGeminiConfig() {
  const explicitEnabled = String(process.env.ENABLE_OPTIONAL_GEMINI || '').trim().toLowerCase() === 'true'
  const apiKey = (process.env.GEMINI_API_KEY || process.env.GEMINI_API_KEY_MUTEE || '').trim()
  const keyPresent = Boolean(apiKey && apiKey !== 'YOUR_GEMINI_API_KEY_HERE')
  const model = process.env.GEMINI_IMAGE_MODEL || 'gemini-2.5-flash-image'

  return {
    configured: keyPresent,
    enabled: explicitEnabled && keyPresent,
    apiKey: explicitEnabled && keyPresent ? apiKey : null,
    model,
  }
}

export function getGeminiKeyPool() {
  const cfg = getGeminiConfig()
  return cfg.apiKey ? [{ owner: 'Configured Operator Key', id: 'primary', key: cfg.apiKey }] : []
}

export async function callGeminiWithFailover({ prompt, mimeType, rawData, generationConfig }) {
  return callGeminiMultiImageWithFailover({
    prompt,
    images: [{ mimeType, rawData }],
    generationConfig,
  })
}

export async function callGeminiMultiImageWithFailover({ prompt, images, generationConfig }) {
  const cfg = getGeminiConfig()

  if (!cfg.configured) {
    return {
      ok: false,
      status: 503,
      code: 'OPTIONAL_PROVIDER_NOT_CONFIGURED',
      error: 'Optional Gemini provider is not configured. Use the open V4 GPU backend instead.',
    }
  }

  if (!cfg.enabled) {
    return {
      ok: false,
      status: 503,
      code: 'OPTIONAL_PROVIDER_DISABLED',
      error:
        'Optional Gemini features are disabled by default (set ENABLE_OPTIONAL_GEMINI=true to opt in). Core virtual try-on runs on the open V4 GPU backend.',
    }
  }

  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${cfg.model}:generateContent`
  const imageParts = (images || []).map((img) => ({
    inline_data: { mime_type: img.mimeType || 'image/jpeg', data: img.rawData },
  }))

  const payload = {
    contents: [{ role: 'user', parts: [{ text: prompt }, ...imageParts] }],
    generationConfig: generationConfig || { responseModalities: ['TEXT', 'IMAGE'] },
  }

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': cfg.apiKey,
      },
      body: JSON.stringify(payload),
    })

    let data = null
    try {
      data = await response.json()
    } catch {
      data = null
    }

    if (!response.ok) {
      const upstreamMsg = data?.error?.message || `Upstream provider returned HTTP ${response.status}.`
      const code =
        response.status === 429
          ? 'PROVIDER_QUOTA_EXCEEDED'
          : response.status === 401 || response.status === 403
            ? 'PROVIDER_AUTH_OR_BILLING_REQUIRED'
            : response.status >= 500
              ? 'PROVIDER_UPSTREAM_ERROR'
              : 'PROVIDER_REQUEST_REJECTED'
      return {
        ok: false,
        status: response.status,
        code,
        error: upstreamMsg,
      }
    }

    const parts = data?.candidates?.[0]?.content?.parts || []
    const imgPart = parts.find((p) => p.inlineData?.data || p.inline_data?.data)
    if (!imgPart) {
      return {
        ok: false,
        status: 502,
        code: 'NO_IMAGE_OUTPUT',
        error: 'Optional provider returned a response without image bytes.',
      }
    }

    const inline = imgPart.inlineData || imgPart.inline_data
    const outMime = inline.mimeType || inline.mime_type || 'image/png'
    return {
      ok: true,
      status: 200,
      model: cfg.model,
      base64Image: `data:${outMime};base64,${inline.data}`,
    }
  } catch (netErr) {
    return {
      ok: false,
      status: 502,
      code: 'PROVIDER_NETWORK_ERROR',
      error: `Could not reach optional provider: ${netErr.message}`,
    }
  }
}
