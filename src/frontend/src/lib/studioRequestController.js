/**
 * Synchronous, race-safe request-ownership & capability controller for StudioPage.
 *
 * Guarantees:
 *  - Run IDs and duplicate-submission guards are maintained synchronously via refs
 *    (never relying on React state-updater callbacks executing synchronously).
 *  - ALL request-derived state (result, resultSnapshot, generatedPoses, resultPose,
 *    loading, jobStage, error) is updated atomically under runId ownership.
 *  - Changing inputs or unmounting aborts browser fetch and invalidates obsolete state
 *    immediately without ever leaving loading stuck.
 *  - Browser abort explicitly does NOT claim to cancel an in-flight GPU kernel.
 */

export function createInitialRequestState() {
  return {
    activeRunId: 0,
    loading: false,
    jobStage: 'idle', // idle | uploading | processing | completed | failed
    result: null,
    resultSnapshot: null,
    generatedPoses: {},
    resultPose: 'front',
    error: '',
  }
}

export function validateCapabilitySelection({
  model_id,
  category,
  tryon_mode,
  garment_photo_type,
  person_age_group,
}) {
  if (category === 'wedding_lehenga' && tryon_mode !== 'flared_lehenga') {
    return 'Wedding Lehenga requires Flared A-Line mode; silent downgrade to upper-body is disallowed.'
  }
  if (category === 'adult_upper' && (tryon_mode !== 'upper_body' || person_age_group !== 'adult')) {
    return 'Adult Upper-Body requires an adult target and upper_body mode.'
  }
  if (category === 'girl_child' && person_age_group !== 'child') {
    return 'Girl-Child category requires a Girl Child target (use Cross-Age for adult target).'
  }
  if (category === 'cross_age' && tryon_mode === 'flared_lehenga') {
    return 'Cross-Age with Flared Lehenga is not supported; use Wedding Lehenga or Cross-Age Kurta/Outfit.'
  }
  if (model_id === 'catvton' && garment_photo_type === 'model') {
    return 'CatVTON requires a Flat-Lay / Product garment reference. Switch to FASHN VTON 1.5 for Model-Worn garment photos.'
  }
  return null
}

/**
 * Create a deterministic synchronous orchestrator bound to refs + React setState.
 * Testable directly with deferred promises and simulated unmounts / double-clicks.
 */
export function createStudioRunOrchestrator({ getState, setState }) {
  let activeRunId = 0
  let inFlight = false
  let mounted = true
  let currentAbort = null

  function mount() {
    mounted = true
  }

  function invalidate() {
    activeRunId += 1
    inFlight = false
    if (currentAbort) {
      currentAbort.abort()
      currentAbort = null
    }
    if (!mounted) return
    setState({
      ...createInitialRequestState(),
      activeRunId,
    })
  }

  function unmount() {
    mounted = false
    activeRunId += 1
    inFlight = false
    if (currentAbort) {
      currentAbort.abort()
      currentAbort = null
    }
  }

  async function submit({ snapshot, executor }) {
    if (inFlight || !mounted) {
      return { started: false, runId: activeRunId }
    }
    const validationErr = validateCapabilitySelection(snapshot)
    if (validationErr) {
      setState((prev) => ({ ...prev, error: validationErr }))
      return { started: false, runId: activeRunId, error: validationErr }
    }

    const runId = ++activeRunId
    inFlight = true
    const controller = new AbortController()
    currentAbort = controller
    const frozenSnapshot = Object.freeze({ ...snapshot })

    setState({
      activeRunId: runId,
      loading: true,
      jobStage: 'uploading',
      result: null,
      resultSnapshot: frozenSnapshot,
      generatedPoses: {},
      resultPose: 'front',
      error: '',
    })

    try {
      const markProcessing = () => {
        if (!mounted || activeRunId !== runId) return
        setState((prev) => (prev.activeRunId === runId ? { ...prev, jobStage: 'processing' } : prev))
      }

      const data = await executor({ signal: controller.signal, markProcessing, runId })

      if (!mounted || activeRunId !== runId) {
        return { started: true, runId, ignored: true }
      }

      inFlight = false
      if (currentAbort === controller) currentAbort = null

      setState({
        activeRunId: runId,
        loading: false,
        jobStage: 'completed',
        result: data,
        resultSnapshot: frozenSnapshot,
        generatedPoses: { front: data.result_image },
        resultPose: 'front',
        error: '',
      })
      return { started: true, runId, completed: true, data }
    } catch (err) {
      if (currentAbort === controller) currentAbort = null
      if (!mounted || activeRunId !== runId) {
        return { started: true, runId, ignored: true }
      }
      inFlight = false
      if (err?.name === 'AbortError') {
        return { started: true, runId, aborted: true }
      }
      const msg = err?.message || 'GPU inference failed.'
      setState({
        activeRunId: runId,
        loading: false,
        jobStage: 'failed',
        result: null,
        resultSnapshot: frozenSnapshot,
        generatedPoses: {},
        resultPose: 'front',
        error: msg,
      })
      return { started: true, runId, failed: true, error: msg }
    }
  }

  return {
    mount,
    submit,
    invalidate,
    unmount,
    isInFlight: () => inFlight,
    isMounted: () => mounted,
    getActiveRunId: () => activeRunId,
    getState,
  }
}

export function formatCheckpointProvenance(checkpoint) {
  if (!checkpoint) return 'none (pretrained / external)'
  if (typeof checkpoint === 'string') return checkpoint
  if (typeof checkpoint === 'object') {
    const files = checkpoint.files && typeof checkpoint.files === 'object' ? checkpoint.files : checkpoint
    const entries = Object.entries(files).filter(([, v]) => typeof v === 'string')
    if (entries.length === 0) return checkpoint.verification_status || 'unverified'
    const summary = entries.map(([name, hash]) => `${name}:${hash.slice(0, 10)}`).join(', ')
    const status = checkpoint.verification_status ? ` [${checkpoint.verification_status}]` : ''
    return `${summary}${status}`
  }
  return 'unknown'
}
