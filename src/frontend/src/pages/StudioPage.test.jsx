import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'
import StudioPage from './StudioPage.jsx'

function createDeferred() {
  let resolve, reject
  const promise = new Promise((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

const LIVE_HEALTH = {
  mode: 'colab_gpu_live',
  colab_gpu_url: 'https://demo.trycloudflare.com',
  process_alive: true,
  gpu_accessible: true,
  files_verified: true,
  model_loaded: false,
  queue_accepting: true,
  auth_required: true,
  authenticated: true,
  parser: 'universal',
  direct: true,
}

beforeEach(() => {
  window.scrollTo = vi.fn()
  globalThis.URL.createObjectURL = vi.fn(() => 'blob:mock-preview')
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('StudioPage Rendered Component Lifecycle & Request Ownership', () => {
  it('survives React.StrictMode setup -> cleanup -> setup and completes live submission + download', async () => {
    const vtonDeferred = createDeferred()
    const fetchMock = vi.fn((url) => {
      if (String(url).includes('/api/gemini-status')) {
        return Promise.resolve({ ok: true, json: async () => ({ configured: true, enabled: false, ready: false }) })
      }
      if (String(url).startsWith('/brand/')) {
        return Promise.resolve({ ok: true, blob: async () => new Blob(['img'], { type: 'image/jpeg' }) })
      }
      if (String(url).includes('/vton')) {
        return vtonDeferred.promise
      }
      return Promise.resolve({ ok: false, status: 404, json: async () => ({ detail: 'not found' }) })
    })
    globalThis.fetch = fetchMock

    render(
      <React.StrictMode>
        <StudioPage
          health={LIVE_HEALTH}
          colabUrlInput="https://demo.trycloudflare.com"
          setColabUrlInput={() => {}}
          colabTokenInput="tok-123"
          setColabTokenInput={() => {}}
          colabStatusMsg=""
          onSaveColabUrl={() => {}}
          onBackHome={() => {}}
        />
      </React.StrictMode>
    )

    fireEvent.click(screen.getByRole('button', { name: /Adult Model/i }))
    await waitFor(() => expect(screen.getByText(/Adult Photo Ready/i)).toBeTruthy())

    const genBtn = screen.getByRole('button', { name: /Generate Live Virtual Try-On/i })
    // Double-click synchronously before rerender -> only 1 /vton call
    fireEvent.click(genBtn)
    fireEvent.click(genBtn)

    await waitFor(() => {
      const vtonCalls = fetchMock.mock.calls.filter(([u]) => String(u).includes('/vton'))
      expect(vtonCalls.length).toBe(1)
    })

    vtonDeferred.resolve({
      ok: true,
      json: async () => ({
        status: 'ok',
        engine: 'colab_gpu_live',
        model_id: 'catvton',
        parser_id: 'universal',
        tryon_mode: 'upper_body',
        seed: 42,
        seconds: 12.3,
        checkpoint_provenance: { verification_status: 'verified_against_manifest', files: { 'config.json': '4584bad889394a23' } },
        result_image: 'data:image/png;base64,iVBORw0KGgo=',
      }),
    })

    await waitFor(() => expect(screen.getByText(/Live Output • catvton/i)).toBeTruthy())
    expect(screen.getByText(/config\.json:4584bad889/i)).toBeTruthy()
    const click=vi.spyOn(HTMLAnchorElement.prototype,'click').mockImplementation(function () {})
    const append=vi.spyOn(document.body,'appendChild')
    fireEvent.click(screen.getByRole('button', { name: /Download PNG/i }))
    expect(click).toHaveBeenCalledTimes(1)
    const anchor=append.mock.calls.map(([el])=>el).find(el=>el.tagName==='A')
    expect(anchor.getAttribute('href')).toBe('data:image/png;base64,iVBORw0KGgo=')
    expect(anchor.download).toBe('fashionfit_adult_upper_catvton_front.png')
  })

  it('invalidates pending VTON when category/model changes mid-flight and ignores late response', async () => {
    const defA = createDeferred()
    globalThis.fetch = vi.fn((url) => {
      if (String(url).includes('/api/gemini-status')) {
        return Promise.resolve({ ok: true, json: async () => ({ configured: false, enabled: false, ready: false }) })
      }
      if (String(url).startsWith('/brand/')) {
        return Promise.resolve({ ok: true, blob: async () => new Blob(['img'], { type: 'image/jpeg' }) })
      }
      if (String(url).includes('/vton')) return defA.promise
      return Promise.resolve({ ok: false, status: 404, json: async () => ({}) })
    })

    render(
      <StudioPage
        health={LIVE_HEALTH}
        colabUrlInput="https://demo.trycloudflare.com"
        setColabUrlInput={() => {}}
        colabTokenInput="tok-123"
        setColabTokenInput={() => {}}
        colabStatusMsg=""
        onSaveColabUrl={() => {}}
        onBackHome={() => {}}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: /Adult Model/i }))
    await waitFor(() => expect(screen.getByText(/Adult Photo Ready/i)).toBeTruthy())

    fireEvent.click(screen.getByRole('button', { name: /Generate Live Virtual Try-On/i }))
    await waitFor(() => expect(screen.getByText(/Uploading|Running Neural Inference/i)).toBeTruthy())

    // Change model mid-flight -> loading immediately clears (never stuck!)
    fireEvent.click(screen.getByRole('button', { name: /FASHN VTON 1\.5/i }))
    expect(screen.getByRole('button', { name: /Generate Live Virtual Try-On/i })).toBeTruthy()

    // Late response from A arrives -> must NOT render stale output
    defA.resolve({
      ok: true,
      json: async () => ({ model_id: 'catvton', result_image: 'data:image/png;base64,STALE' }),
    })
    await new Promise((r) => setTimeout(r, 30))
    expect(screen.queryByText(/Live Output • catvton/i)).toBeNull()
  })

  it('invalidates pending segmentation when parser variant changes mid-flight', async () => {
    const segDef = createDeferred()
    globalThis.fetch = vi.fn((url) => {
      if (String(url).includes('/api/gemini-status')) {
        return Promise.resolve({ ok: true, json: async () => ({ configured: false, enabled: false, ready: false }) })
      }
      if (String(url).includes('/segment')) return segDef.promise
      return Promise.resolve({ ok: false, status: 404, json: async () => ({}) })
    })

    const { container } = render(
      <StudioPage
        initialTab="segment"
        health={LIVE_HEALTH}
        colabUrlInput="https://demo.trycloudflare.com"
        setColabUrlInput={() => {}}
        colabTokenInput="tok-123"
        setColabTokenInput={() => {}}
        colabStatusMsg=""
        onSaveColabUrl={() => {}}
        onBackHome={() => {}}
      />
    )

    const fileInput = container.querySelector('input[type="file"]')
    fireEvent.change(fileInput, { target: { files: [new File(['png'], 'outfit.png', { type: 'image/png' })] } })
    fireEvent.click(screen.getByRole('button', { name: /Extract Cutout/i }))

    // Switch parser variant mid-flight
    const select = container.querySelector('select')
    fireEvent.change(select, { target: { value: 'baseline' } })

    // Late adapted response arrives -> must NOT appear under baseline selection
    segDef.resolve({
      ok: true,
      json: async () => ({
        model_id: 'universal',
        overlay_image: 'data:image/png;base64,OLD',
        kurta_cutout_image: 'data:image/png;base64,OLD',
      }),
    })
    await new Promise((r) => setTimeout(r, 30))
    expect(screen.queryByText(/4-Class Overlay \(universal\)/i)).toBeNull()
  })
})
