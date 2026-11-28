import test from 'node:test'
import assert from 'node:assert/strict'

import {
  createInitialRequestState,
  createStudioRunOrchestrator,
  validateCapabilitySelection,
  formatCheckpointProvenance,
} from '../src/frontend/src/lib/studioRequestController.js'
import { getGeminiConfig } from '../src/api/_gemini.js'

function createDeferred() {
  let resolve, reject
  const promise = new Promise((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

function mountHarness() {
  let state = createInitialRequestState()
  let setCalls = 0
  const setState = (updater) => {
    setCalls += 1
    state = typeof updater === 'function' ? updater(state) : updater
  }
  const orch = createStudioRunOrchestrator({
    getState: () => state,
    setState,
  })
  return { orch, getState: () => state, getSetCalls: () => setCalls }
}

const VALID_ADULT = {
  category: 'adult_upper',
  tryon_mode: 'upper_body',
  model_id: 'catvton',
  garment_photo_type: 'flat-lay',
  person_age_group: 'adult',
  seed: 42,
}

const VALID_CHILD = {
  category: 'girl_child',
  tryon_mode: 'full_outfit',
  model_id: 'fashn_vton_1_5',
  garment_photo_type: 'flat-lay',
  person_age_group: 'child',
  seed: 42,
}

test('A starts -> inputs change -> A succeeds late: never stuck, stale result/poses ignored', async () => {
  const { orch, getState } = mountHarness()
  const defA = createDeferred()

  const pA = orch.submit({ snapshot: VALID_ADULT, executor: () => defA.promise })
  assert.equal(getState().loading, true)
  assert.equal(orch.isInFlight(), true)

  // Inputs change mid-flight
  orch.invalidate()
  assert.equal(getState().loading, false)
  assert.equal(getState().jobStage, 'idle')
  assert.equal(orch.isInFlight(), false)

  // A succeeds late
  defA.resolve({ result_image: 'stale_A_img' })
  const resA = await pA
  assert.equal(resA.ignored, true)
  assert.equal(getState().loading, false)
  assert.equal(getState().result, null)
  assert.deepEqual(getState().generatedPoses, {})
})

test('A starts -> inputs change -> A fails late: stale error ignored', async () => {
  const { orch, getState } = mountHarness()
  const defA = createDeferred()

  const pA = orch.submit({ snapshot: VALID_ADULT, executor: () => defA.promise })
  orch.invalidate()

  defA.reject(new Error('Late network error from A'))
  const resA = await pA
  assert.equal(resA.ignored, true)
  assert.equal(getState().error, '')
  assert.equal(getState().jobStage, 'idle')
})

test('A invalidated -> B starts -> B completes before A -> A fails/succeeds late: B preserved', async () => {
  const { orch, getState } = mountHarness()
  const defA = createDeferred()
  const defB = createDeferred()

  const pA = orch.submit({ snapshot: VALID_ADULT, executor: () => defA.promise })
  orch.invalidate()

  const pB = orch.submit({ snapshot: VALID_CHILD, executor: () => defB.promise })
  assert.equal(getState().loading, true)

  // B completes first
  defB.resolve({ result_image: 'fresh_B_img', model_id: 'fashn_vton_1_5' })
  await pB
  assert.equal(getState().loading, false)
  assert.equal(getState().result.result_image, 'fresh_B_img')
  assert.equal(getState().resultSnapshot.category, 'girl_child')

  // A finishes late -> must NOT overwrite B's result or poses
  defA.resolve({ result_image: 'stale_A_img' })
  await pA
  assert.equal(getState().result.result_image, 'fresh_B_img')
  assert.equal(getState().generatedPoses.front, 'fresh_B_img')
  assert.equal(getState().resultSnapshot.category, 'girl_child')
})

test('Synchronous double-click before rerender is blocked immediately', async () => {
  const { orch } = mountHarness()
  const def1 = createDeferred()
  let execCount = 0

  const p1 = orch.submit({
    snapshot: VALID_ADULT,
    executor: () => {
      execCount += 1
      return def1.promise
    },
  })
  // Second click synchronously before any await or rerender
  const r2 = await orch.submit({
    snapshot: VALID_ADULT,
    executor: () => {
      execCount += 1
      return Promise.resolve({ result_image: 'dup' })
    },
  })

  assert.equal(r2.started, false)
  assert.equal(execCount, 1)
  def1.resolve({ result_image: 'ok' })
  await p1
})

test('React Strict Mode cycle (mount -> cleanup/unmount -> remount) allows subsequent submission', async () => {
  const { orch, getState } = mountHarness()
  // Simulate React 18 Strict Mode dev double-effect: mount -> unmount -> mount
  orch.mount()
  orch.unmount()
  assert.equal(orch.isMounted(), false)
  orch.mount()
  assert.equal(orch.isMounted(), true)

  const res = await orch.submit({
    snapshot: VALID_ADULT,
    executor: async () => ({ result_image: 'strict_mode_ok', model_id: 'catvton' }),
  })
  assert.equal(res.started, true)
  assert.equal(res.completed, true)
  assert.equal(getState().result.result_image, 'strict_mode_ok')
  assert.equal(getState().loading, false)
})

test('Unmount while pending aborts browser fetch and produces zero post-unmount state writes', async () => {
  const { orch, getSetCalls } = mountHarness()
  const defA = createDeferred()
  let aborted = false

  const pA = orch.submit({
    snapshot: VALID_ADULT,
    executor: ({ signal }) => {
      signal.addEventListener('abort', () => {
        aborted = true
      })
      return defA.promise
    },
  })

  const callsBeforeUnmount = getSetCalls()
  orch.unmount()
  assert.equal(aborted, true)

  defA.resolve({ result_image: 'post_unmount' })
  await pA
  assert.equal(getSetCalls(), callsBeforeUnmount)
})

test('Combination capability validation rejects invalid category/mode/model combinations', () => {
  assert.ok(validateCapabilitySelection({ ...VALID_ADULT, category: 'wedding_lehenga', tryon_mode: 'upper_body' }))
  assert.ok(validateCapabilitySelection({ ...VALID_ADULT, model_id: 'catvton', garment_photo_type: 'model' }))
  assert.ok(validateCapabilitySelection({ ...VALID_ADULT, category: 'girl_child', person_age_group: 'adult' }))
  assert.equal(validateCapabilitySelection(VALID_ADULT), null)
  assert.equal(validateCapabilitySelection(VALID_CHILD), null)
})

test('Checkpoint provenance formats dict cleanly and Gemini stays disabled by default', () => {
  const formatted = formatCheckpointProvenance({
    verification_status: 'hashes_recorded_no_manifest',
    files: { 'config.json': 'abcdef1234567890' },
  })
  assert.ok(!formatted.includes('[object Object]'))
  assert.ok(formatted.includes('config.json:abcdef1234'))

  const prev = process.env.ENABLE_OPTIONAL_GEMINI
  delete process.env.ENABLE_OPTIONAL_GEMINI
  try {
    const cfg = getGeminiConfig()
    assert.equal(cfg.enabled, false)
  } finally {
    if (prev !== undefined) process.env.ENABLE_OPTIONAL_GEMINI = prev
  }
})

