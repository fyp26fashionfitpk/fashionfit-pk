import React, { useState, useEffect, useRef } from 'react'
import {
  ArrowLeft,
  Upload,
  Sparkles,
  Scissors,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Wand2,
  Camera,
  Settings2,
  UserCheck,
  Shirt,
  ArrowLeftRight,
  Repeat,
  ShieldCheck,
  Download,
  Info,
  KeyRound,
} from 'lucide-react'
import {
  STUDIO_POSES,
  STUDIO_MODELS,
  WARDROBE_CATEGORIES,
  ALL_WARDROBE_ITEMS,
  STUDIO_TIPS,
} from '../data/verifiedData.js'
import {
  createInitialRequestState,
  createStudioRunOrchestrator,
  formatCheckpointProvenance,
} from '../lib/studioRequestController.js'

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

async function urlToFile(url, filename = 'image.jpg') {
  const res = await fetch(url)
  const blob = await res.blob()
  return new File([blob], filename, { type: blob.type || 'image/jpeg' })
}

const STUDIO_MODEL_PATHS = new Set(STUDIO_MODELS.map((m) => m.image))

export default function StudioPage({
  initialTab = 'photo',
  initialKurtaId = null,
  onBackHome,
  health,
  colabUrlInput,
  setColabUrlInput,
  colabTokenInput = '',
  setColabTokenInput = () => {},
  colabStatusMsg,
  onSaveColabUrl,
}) {
  const initialGarment =
    ALL_WARDROBE_ITEMS.find((item) => item.id === initialKurtaId) || ALL_WARDROBE_ITEMS[0]

  const [activeTab, setActiveTab] = useState(initialTab)
  const [showServerPanel, setShowServerPanel] = useState(false)

  // Person state (Adult or Girl Child)
  const [personFile, setPersonFile] = useState(null)
  const [personPreview, setPersonPreview] = useState(null)
  const [personAgeGroup, setPersonAgeGroup] = useState(
    initialGarment.category === 'kids' ? 'child' : 'adult'
  )
  const [preppedPersonPreview, setPreppedPersonPreview] = useState(null)
  // Optional Gemini cleanup is strictly OFF by default (core workflow is Gemini-free)
  const [autoRefinePerson, setAutoRefinePerson] = useState(false)
  const [personLoading, setPersonLoading] = useState(false)

  // Garment state (Women's Kurtas, Kids Girls, Wedding Lehengas, or Custom)
  const [wardrobeCategory, setWardrobeCategory] = useState(initialGarment.category || 'kurta')
  const [selectedKurtaId, setSelectedKurtaId] = useState(initialGarment.id)
  const [garmentFile, setGarmentFile] = useState(null)
  const [garmentPreview, setGarmentPreview] = useState(initialGarment.image)
  const [garmentPhotoType, setGarmentPhotoType] = useState('flat-lay')
  const [canonicalPreview, setCanonicalPreview] = useState(null)
  const [autoPerfectGarment, setAutoPerfectGarment] = useState(false)
  const [preserveDupatta, setPreserveDupatta] = useState(true)
  const [garmentLoading, setGarmentLoading] = useState(false)

  // Try-On, Routing & Lifecycle state (100% owned by synchronous orchestrator)
  const [selectedModel, setSelectedModel] = useState('catvton')
  const [demoPose, setDemoPose] = useState('front')
  const [reqState, setReqState] = useState(createInitialRequestState)
  const [showOriginalToggle, setShowOriginalToggle] = useState(false)
  const reqStateRef = useRef(reqState)
  reqStateRef.current = reqState

  const orchestratorRef = useRef(null)
  if (!orchestratorRef.current) {
    orchestratorRef.current = createStudioRunOrchestrator({
      getState: () => reqStateRef.current,
      setState: setReqState,
    })
  }

  const vtonResult = reqState.result
  const vtonSnapshot = reqState.resultSnapshot
  const generatedPoses = reqState.generatedPoses
  const resultPose = reqState.resultPose
  const jobStage = reqState.jobStage
  const loading = reqState.loading

  // Cutout state with its own synchronous run-ID & AbortController guard
  const [segFile, setSegFile] = useState(null)
  const [segPreview, setSegPreview] = useState(null)
  const [segVariant, setSegVariant] = useState('adapted')
  const [segResult, setSegResult] = useState(null)
  const [segLoading, setSegLoading] = useState(false)
  const segRunIdRef = useRef(0)
  const segAbortRef = useRef(null)
  const mountedRef = useRef(true)

  const invalidateSegResult = () => {
    segRunIdRef.current += 1
    if (segAbortRef.current) {
      segAbortRef.current.abort()
      segAbortRef.current = null
    }
    setSegLoading(false)
    setSegResult(null)
  }

  const [geminiStatus, setGeminiStatus] = useState({ configured: false, enabled: false, ready: false })
  const [poseLoading, setPoseLoading] = useState(false)
  const [localError, setLocalError] = useState('')
  const error = reqState.error || localError
  const setError = (msg) => {
    setLocalError(msg)
    if (!msg && reqStateRef.current.error) {
      setReqState((prev) => ({ ...prev, error: '' }))
    }
  }

  useEffect(() => {
    mountedRef.current = true
    orchestratorRef.current?.mount()
    window.scrollTo(0, 0)
    fetch('/api/gemini-status')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d && mountedRef.current) setGeminiStatus(d)
      })
      .catch(() => {})
    return () => {
      mountedRef.current = false
      segRunIdRef.current += 1
      if (segAbortRef.current) {
        segAbortRef.current.abort()
        segAbortRef.current = null
      }
      orchestratorRef.current?.unmount()
    }
  }, [])

  // No deadlock: worker can be cold (files_verified=true, model_loaded=false) and lazy-load on first run
  const isGpuReachable = health?.mode === 'colab_gpu_live' && Boolean(health?.gpu_accessible)
  const isGpuAuthenticated = !health?.auth_required || Boolean(health?.authenticated)
  const isGpuAccepting = isGpuReachable && Boolean(health?.files_verified) && Boolean(health?.queue_accepting)
  const isGpuConnected = isGpuAccepting && isGpuAuthenticated
  const geminiEnabled = Boolean(geminiStatus?.enabled)

  const directEndpoint = (path) => {
    if (health?.direct && health?.colab_gpu_url) {
      return `${health.colab_gpu_url}${path}`
    }
    return `/api${path}`
  }

  const buildAuthHeaders = () => {
    const t = (colabTokenInput || '').trim()
    return t ? { Authorization: `Bearer ${t}` } : {}
  }

  // Synchronously invalidate stale result & abort browser fetch (never leaves loading stuck)
  const invalidateStaleResult = () => {
    orchestratorRef.current.invalidate()
    setShowOriginalToggle(false)
    setLocalError('')
  }

  // Derive explicit category & try-on mode (no silent downgrades)
  const activeGarmentItem = ALL_WARDROBE_ITEMS.find((item) => item.id === selectedKurtaId)
  const effectiveGarmentCategory = activeGarmentItem ? activeGarmentItem.category : wardrobeCategory
  const effectiveGarmentAge = effectiveGarmentCategory === 'kids' ? 'child' : 'adult'
  const isCrossAge = personAgeGroup !== effectiveGarmentAge

  const routingCategory =
    effectiveGarmentCategory === 'lehenga'
      ? 'wedding_lehenga'
      : isCrossAge
        ? 'cross_age'
        : effectiveGarmentCategory === 'kids'
          ? 'girl_child'
          : 'adult_upper'

  const activeTryonMode =
    effectiveGarmentCategory === 'lehenga'
      ? 'flared_lehenga'
      : effectiveGarmentCategory === 'kids' && !isCrossAge
        ? 'full_outfit'
        : 'upper_body'

  const activeModeBadge =
    routingCategory === 'wedding_lehenga'
      ? {
          title: 'Wedding Lehenga • Experimental',
          desc: 'Full-body photo required. Flared A-line mask heuristic; unseen backs/components cannot be reproduced.',
          tone: 'bg-rose-50 text-[#9B1D4A] border-rose-200',
        }
      : routingCategory === 'cross_age'
        ? {
            title:
              personAgeGroup === 'child'
                ? 'Cross-Age (Adult → Girl Child) • Experimental'
                : 'Cross-Age (Kids → Adult) • Experimental',
            desc: 'Visual appearance transfer preserving target age & proportions; not physical tailoring/sizing.',
            tone: 'bg-amber-50 text-amber-800 border-amber-200',
          }
        : routingCategory === 'girl_child'
          ? {
              title: 'Girl-Child Ethnic • Experimental',
              desc: 'Evaluated separately with V4 Universal 4-Class parser (77.63% kids FG mIoU pilot).',
              tone: 'bg-emerald-50 text-[#0F5C4D] border-emerald-200',
            }
          : {
              title: 'Adult Pakistani Kurta • Experimental',
              desc: 'Upper-body transfer preserving lower clothing and draped dupatta (visual preview only).',
              tone: 'bg-slate-50 text-slate-700 border-slate-200',
            }

  const activeCategoryObj =
    WARDROBE_CATEGORIES.find((c) => c.id === wardrobeCategory) || WARDROBE_CATEGORIES[0]

  const handlePersonUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (file.size > 10 * 1024 * 1024) {
      setError('Person image exceeds the 10 MB upload limit.')
      return
    }
    invalidateStaleResult()
    setPersonFile(file)
    setPersonPreview(URL.createObjectURL(file))
    setPreppedPersonPreview(null)
    setError('')
  }

  const handleUseStudioModel = async (modelId = 'adult') => {
    const model = STUDIO_MODELS.find((m) => m.id === modelId) || STUDIO_MODELS[0]
    invalidateStaleResult()
    setError('')
    setPersonAgeGroup(model.ageGroup)
    setPersonPreview(model.image)
    setPreppedPersonPreview(null)
    try {
      const f = await urlToFile(model.image, model.filename)
      setPersonFile(f)
    } catch {
      setError('Could not load studio model.')
    }
  }

  const handleSwapCrossAge = async () => {
    invalidateStaleResult()
    const nextAge = personAgeGroup === 'adult' ? 'child' : 'adult'
    if (!personPreview || STUDIO_MODEL_PATHS.has(personPreview)) {
      await handleUseStudioModel(nextAge)
    } else {
      setPersonAgeGroup(nextAge)
    }
  }

  const handleSelectCategory = (catId) => {
    invalidateStaleResult()
    setWardrobeCategory(catId)
    const cat = WARDROBE_CATEGORIES.find((c) => c.id === catId)
    if (cat && cat.items.length > 0 && selectedKurtaId !== 'custom') {
      const first = cat.items[0]
      setSelectedKurtaId(first.id)
      setGarmentFile(null)
      setGarmentPreview(first.image)
      setGarmentPhotoType('flat-lay')
      setCanonicalPreview(null)
    }
    setError('')
  }

  const handleSelectPredefinedKurta = (kurta) => {
    invalidateStaleResult()
    setSelectedKurtaId(kurta.id)
    setWardrobeCategory(kurta.category)
    setGarmentFile(null)
    setGarmentPreview(kurta.image)
    setGarmentPhotoType('flat-lay')
    setCanonicalPreview(null)
    setError('')
  }

  const handleGarmentUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (file.size > 10 * 1024 * 1024) {
      setError('Garment image exceeds the 10 MB upload limit.')
      return
    }
    invalidateStaleResult()
    setSelectedKurtaId('custom')
    setGarmentFile(file)
    setGarmentPreview(URL.createObjectURL(file))
    setCanonicalPreview(null)
    setError('')
  }

  const runRefinePerson = async () => {
    if (!geminiEnabled) {
      setError('Optional Gemini cleanup is disabled. Core try-on uses your original photo directly.')
      return null
    }
    if (!personFile && !personPreview) {
      setError('Upload your photo first to refine it.')
      return null
    }
    setPersonLoading(true)
    setError('')
    try {
      const fileObj = personFile || (await urlToFile(personPreview, 'person.jpg'))
      const dataUrl = await fileToDataUrl(fileObj)
      const res = await fetch('/api/gemini-prep-person', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: dataUrl, person_age_group: personAgeGroup }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Could not refine person photo.')
      setPreppedPersonPreview(data.prepped_person_image)
      return data.prepped_person_image
    } catch (err) {
      setError(err.message)
      return null
    } finally {
      setPersonLoading(false)
    }
  }

  const runPerfectGarment = async () => {
    if (!geminiEnabled) {
      setError('Optional Gemini garment reconstruction is disabled. Core try-on uses your uploaded garment directly.')
      return null
    }
    if (!garmentFile && !garmentPreview) {
      setError('Select or upload an outfit photo first.')
      return null
    }
    setGarmentLoading(true)
    setError('')
    try {
      const fileObj = garmentFile || (await urlToFile(garmentPreview, 'garment.jpg'))
      const dataUrl = await fileToDataUrl(fileObj)
      const res = await fetch('/api/gemini-canonicalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: dataUrl,
          garment_category: effectiveGarmentCategory,
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Could not refine garment photo.')
      setCanonicalPreview(data.canonical_image)
      return data.canonical_image
    } catch (err) {
      setError(err.message)
      return null
    } finally {
      setGarmentLoading(false)
    }
  }

  const runVton = async () => {
    if (orchestratorRef.current.isInFlight()) return
    if (!personPreview) {
      setError('Please upload a photo or pick Adult Model / Girl Child above.')
      return
    }
    if (!garmentPreview) {
      setError('Please choose an outfit from the wardrobe or upload your own.')
      return
    }
    if (!isGpuReachable) {
      setShowServerPanel(true)
      setError(
        'No live V4 GPU worker connected. Start FashionFit_PK_Live_GPU_Server.ipynb on Colab GPU, paste its URL + Bearer token in "Connect V4 GPU", and retry.'
      )
      return
    }
    if (!isGpuAuthenticated) {
      setShowServerPanel(true)
      setError('GPU worker reached, but Bearer Access Token is missing or rejected. Enter the valid token and click Connect & Verify.')
      return
    }
    if (!isGpuAccepting) {
      setShowServerPanel(true)
      setError('GPU worker is currently busy or missing verified model files. Check worker status in the top bar.')
      return
    }

    const snapshot = {
      category: routingCategory,
      tryon_mode: activeTryonMode,
      model_id: selectedModel,
      garment_photo_type: garmentPhotoType,
      person_age_group: personAgeGroup,
      seed: 42,
    }
    setLocalError('')
    setShowOriginalToggle(false)

    await orchestratorRef.current.submit({
      snapshot,
      executor: async ({ signal, markProcessing }) => {
        let personToSend = personFile || (await urlToFile(personPreview, 'person.jpg'))
        let garmentToSend = garmentFile || (await urlToFile(garmentPreview, 'garment.jpg'))

        if (autoRefinePerson && geminiEnabled && !STUDIO_MODEL_PATHS.has(personPreview)) {
          const preppedUrl = preppedPersonPreview || (await runRefinePerson())
          if (preppedUrl) personToSend = await urlToFile(preppedUrl, 'prepped_person.png')
        }
        if (autoPerfectGarment && geminiEnabled && selectedKurtaId === 'custom') {
          const canonicalUrl = canonicalPreview || (await runPerfectGarment())
          if (canonicalUrl) garmentToSend = await urlToFile(canonicalUrl, 'canonical_garment.png')
        }

        const fd = new FormData()
        fd.append('person_image', personToSend)
        fd.append('garment_image', garmentToSend)
        fd.append('model_id', snapshot.model_id)
        fd.append('category', snapshot.category)
        fd.append('tryon_mode', snapshot.tryon_mode)
        fd.append('garment_photo_type', snapshot.garment_photo_type)
        fd.append('person_age_group', snapshot.person_age_group)
        fd.append('seed', String(snapshot.seed))

        markProcessing()
        const res = await fetch(directEndpoint('/vton'), {
          method: 'POST',
          headers: buildAuthHeaders(),
          body: fd,
          signal,
        })
        const data = await res.json()
        if (!res.ok) {
          const detail = typeof data.detail === 'object' ? data.detail?.message : data.detail
          throw new Error(detail || `GPU inference failed (HTTP ${res.status}).`)
        }
        return data
      },
    })
  }

  const handleSelectResultPose = async (poseId) => {
    const ownerRunId = orchestratorRef.current.getActiveRunId()
    if (poseId === 'front') {
      setReqState((prev) => (prev.activeRunId === ownerRunId ? { ...prev, resultPose: 'front' } : prev))
      return
    }
    if (!geminiEnabled) {
      setError(
        'Additional AI-synthesized angles (3/4 & Back) are disabled because optional Gemini billing is off. Unseen back details cannot be reconstructed from a single front image.'
      )
      return
    }
    if (!vtonResult?.result_image) return
    setReqState((prev) => (prev.activeRunId === ownerRunId ? { ...prev, resultPose: poseId } : prev))
    if (generatedPoses[poseId]) return
    setPoseLoading(true)
    setError('')
    try {
      const res = await fetch('/api/gemini-views', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          result_image_base64: vtonResult.result_image,
          view_mode: poseId,
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Could not generate optional angle preview.')
      if (!mountedRef.current || orchestratorRef.current.getActiveRunId() !== ownerRunId) return
      setReqState((prev) =>
        prev.activeRunId === ownerRunId
          ? { ...prev, generatedPoses: { ...prev.generatedPoses, [poseId]: data.view_image } }
          : prev
      )
    } catch (err) {
      if (!mountedRef.current || orchestratorRef.current.getActiveRunId() !== ownerRunId) return
      setReqState((prev) => (prev.activeRunId === ownerRunId ? { ...prev, resultPose: 'front' } : prev))
      setError(err.message)
    } finally {
      if (mountedRef.current) setPoseLoading(false)
    }
  }

  const handleDownloadResult = () => {
    const imgSrc = generatedPoses[resultPose] || vtonResult?.result_image
    if (!imgSrc) return
    const a = document.createElement('a')
    a.href = imgSrc
    a.download = `fashionfit_${vtonSnapshot?.category || routingCategory}_${vtonSnapshot?.model_id || selectedModel}_${resultPose}.png`
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  const runSegment = async () => {
    if (segLoading) return
    if (!segFile) {
      setError('Upload a full-body outfit photo first.')
      return
    }
    if (!isGpuConnected) {
      setShowServerPanel(true)
      setError('Connect and authenticate your V4 GPU worker in the top bar first.')
      return
    }
    const segRunId = ++segRunIdRef.current
    const controller = new AbortController()
    segAbortRef.current = controller
    setSegLoading(true)
    setError('')
    setSegResult(null)
    try {
      const fd = new FormData()
      fd.append('image', segFile)
      fd.append('seg_variant', segVariant)
      const res = await fetch(directEndpoint('/segment'), {
        method: 'POST',
        headers: buildAuthHeaders(),
        body: fd,
        signal: controller.signal,
      })
      const data = await res.json()
      if (!res.ok) {
        const detail = typeof data.detail === 'object' ? data.detail?.message : data.detail
        throw new Error(detail || 'Extraction failed.')
      }
      if (!mountedRef.current || segRunIdRef.current !== segRunId) return
      setSegResult(data)
    } catch (err) {
      if (err?.name === 'AbortError') return
      if (!mountedRef.current || segRunIdRef.current !== segRunId) return
      setError(err.message)
    } finally {
      if (segAbortRef.current === controller) segAbortRef.current = null
      if (mountedRef.current && segRunIdRef.current === segRunId) setSegLoading(false)
    }
  }

  const activeDemoPoseObj = STUDIO_POSES.find((p) => p.id === demoPose) || STUDIO_POSES[0]

  return (
    <div className="min-h-screen flex flex-col bg-[#FAF8F5] text-slate-900">
      {/* Dedicated Studio Top Bar */}
      <header className="sticky top-0 z-40 bg-white/90 backdrop-blur-md border-b border-slate-200/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 sm:gap-5">
            <button
              onClick={onBackHome}
              className="inline-flex items-center gap-1.5 text-xs sm:text-sm font-semibold text-slate-600 hover:text-slate-900 transition cursor-pointer"
            >
              <ArrowLeft className="w-4 h-4" />
              <span className="hidden sm:inline">Back to Home</span>
            </button>
            <div className="h-5 w-px bg-slate-200 hidden sm:block" />
            <div className="flex items-center gap-2">
              <span className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#0F5C4D] to-[#9B1D4A] text-white font-display font-bold text-sm flex items-center justify-center">
                F
              </span>
              <span className="font-bold text-sm sm:text-base tracking-tight text-slate-900">
                FashionFit<span className="text-[#9B1D4A]">-PK</span>{' '}
                <span className="font-medium text-slate-400 hidden md:inline">Studio V4</span>
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`hidden md:inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-full border ${
                isGpuConnected
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : 'bg-slate-50 text-slate-600 border-slate-200'
              }`}
            >
              {isGpuConnected
                ? `V4 GPU Ready (${health.parser || 'universal'})`
                : 'Frontend Standby • Open V4 Core'}
            </span>
            <button
              onClick={() => setShowServerPanel(!showServerPanel)}
              className={`inline-flex items-center gap-1.5 text-xs font-semibold px-3.5 py-1.5 rounded-full border transition cursor-pointer ${
                isGpuConnected
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : health?.mode === 'gpu_worker_unreachable'
                    ? 'bg-amber-50 text-amber-800 border-amber-300'
                    : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
              }`}
            >
              <Settings2 className="w-3.5 h-3.5" />
              {isGpuConnected
                ? 'GPU Connected'
                : health?.mode === 'gpu_worker_unreachable'
                  ? 'GPU Unreachable'
                  : 'Connect V4 GPU'}
            </button>
          </div>
        </div>

        {/* Collapsible Authenticated V4 GPU Worker Bar */}
        {showServerPanel && (
          <div className="bg-slate-50 border-t border-slate-200 px-4 sm:px-6 py-3">
            <div className="max-w-7xl mx-auto space-y-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
                <span>
                  <strong className="text-slate-800">V4 GPU Worker (Interactive Colab / Authenticated GPU): </strong>
                  {colabStatusMsg || 'Paste the HTTPS URL and Bearer token from 06_Live_GPU_Server.ipynb.'}
                </span>
                <span className="text-[11px] text-slate-500">
                  Gemini optional provider: <strong>{geminiEnabled ? 'Enabled' : 'Disabled (Core is $0 open pipeline)'}</strong>
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <input
                  type="url"
                  value={colabUrlInput}
                  onChange={(e) => setColabUrlInput(e.target.value)}
                  placeholder="https://your-v4-worker.trycloudflare.com"
                  className="text-xs px-3 py-1.5 rounded-lg border border-slate-300 bg-white flex-1 min-w-[220px] focus:outline-none focus:ring-2 focus:ring-[#0F5C4D]"
                />
                <div className="relative flex-1 min-w-[200px]">
                  <KeyRound className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2" />
                  <input
                    type="password"
                    value={colabTokenInput}
                    onChange={(e) => setColabTokenInput(e.target.value)}
                    placeholder="Bearer Access Token (FF_SERVER_TOKEN)"
                    className="text-xs pl-8 pr-3 py-1.5 rounded-lg border border-slate-300 bg-white w-full focus:outline-none focus:ring-2 focus:ring-[#0F5C4D]"
                  />
                </div>
                <button
                  onClick={onSaveColabUrl}
                  className="text-xs font-semibold px-4 py-1.5 rounded-lg bg-[#0F5C4D] text-white hover:bg-[#0b473b] transition cursor-pointer"
                >
                  Connect &amp; Verify
                </button>
              </div>
            </div>
          </div>
        )}
      </header>

      {/* Main Studio Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8 space-y-6">
        {/* Clean Mode Switcher */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="inline-flex flex-wrap gap-2 bg-white p-1.5 rounded-full border border-slate-200/80 shadow-xs">
            {[
              { id: 'photo', label: 'Live Virtual Try-On', icon: Sparkles },
              { id: 'demo', label: 'Saved Showcase (Static)', icon: Camera },
              { id: 'segment', label: '4-Class Garment Cutout', icon: Scissors },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => {
                  setActiveTab(id)
                  setError('')
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs sm:text-sm font-semibold transition cursor-pointer ${
                  activeTab === id
                    ? 'bg-slate-900 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="flex items-start gap-2.5 p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-sm">
            <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* TAB 1: FULL-PAGE VIRTUAL TRY-ON STUDIO */}
        {activeTab === 'photo' && (
          <div className="grid lg:grid-cols-12 gap-8 items-start">
            {/* LEFT COLUMN: Controls, Uploads, Wardrobe & Routing */}
            <div className="lg:col-span-5 space-y-6">
              {/* Step 1: Person Photo (Adult & Girl Child) */}
              <div className="rounded-3xl bg-white border border-slate-200/80 p-5 shadow-xs space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-[#0F5C4D] text-white text-xs font-bold flex items-center justify-center">
                      1
                    </span>
                    <h2 className="text-sm font-bold text-slate-900">Your Photo</h2>
                  </div>
                  <div className="flex items-center gap-1.5">
                    {STUDIO_MODELS.map((m) => {
                      const active = personPreview === m.image
                      return (
                        <button
                          key={m.id}
                          onClick={() => handleUseStudioModel(m.id)}
                          className={`inline-flex items-center gap-1 text-[11px] font-semibold px-2.5 py-1 rounded-full border transition cursor-pointer ${
                            active
                              ? 'bg-[#0F5C4D] text-white border-[#0F5C4D]'
                              : 'bg-white text-[#0F5C4D] border-emerald-200 hover:bg-emerald-50'
                          }`}
                        >
                          <UserCheck className="w-3 h-3" />
                          {m.label}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {personPreview ? (
                  <div className="flex items-center gap-4 p-3 rounded-2xl bg-[#FAF8F5] border border-slate-200/70">
                    <img
                      src={preppedPersonPreview || personPreview}
                      alt="Selected person"
                      className="w-20 h-28 object-cover rounded-xl bg-white border border-slate-200 shrink-0"
                    />
                    <div className="space-y-2 flex-1 min-w-0">
                      <p className="text-xs font-bold text-slate-800 truncate">
                        {personAgeGroup === 'child' ? 'Girl Child Photo Ready' : 'Adult Photo Ready'}
                      </p>

                      <div className="inline-flex rounded-full bg-white border border-slate-200 p-0.5">
                        {[
                          { id: 'adult', label: 'Adult' },
                          { id: 'child', label: 'Girl Child' },
                        ].map((opt) => (
                          <button
                            key={opt.id}
                            onClick={() => {
                              invalidateStaleResult()
                              setPersonAgeGroup(opt.id)
                            }}
                            className={`px-2.5 py-0.5 rounded-full text-[11px] font-semibold transition cursor-pointer ${
                              personAgeGroup === opt.id
                                ? 'bg-slate-900 text-white'
                                : 'text-slate-600 hover:text-slate-900'
                            }`}
                          >
                            {opt.label}
                          </button>
                        ))}
                      </div>

                      <div className="flex flex-wrap gap-2">
                        <label className="cursor-pointer text-xs font-semibold px-3 py-1 rounded-full bg-white border border-slate-300 hover:bg-slate-50">
                          Change
                          <input type="file" accept="image/*" className="hidden" onChange={handlePersonUpload} />
                        </label>
                        {geminiEnabled && !STUDIO_MODEL_PATHS.has(personPreview) && (
                          <button
                            onClick={runRefinePerson}
                            disabled={personLoading}
                            className="inline-flex items-center gap-1 text-xs font-semibold px-3 py-1 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 cursor-pointer"
                          >
                            {personLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Wand2 className="w-3 h-3" />}
                            Optional AI Refine
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <label className="block cursor-pointer rounded-2xl border-2 border-dashed border-slate-300 hover:border-[#0F5C4D] bg-[#FAF8F5]/70 p-6 text-center transition">
                    <Upload className="w-7 h-7 text-slate-400 mx-auto mb-2" />
                    <p className="text-sm font-semibold text-slate-800">Upload a full-body photo (max 10 MB)</p>
                    <p className="text-xs text-slate-500 mt-1">
                      JPG, PNG or WebP • Or pick <strong>Adult Model</strong> / <strong>Girl Child</strong> above
                    </p>
                    <input type="file" accept="image/*" className="hidden" onChange={handlePersonUpload} />
                  </label>
                )}

                {personAgeGroup === 'child' && (
                  <p className="flex items-start gap-1.5 text-[11px] text-emerald-900 bg-emerald-50/80 border border-emerald-200/70 rounded-xl px-3 py-2">
                    <ShieldCheck className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                    <span>
                      Studio Girl Child preset is synthetic. Uploaded photos are sent directly to your connected GPU worker
                      and stripped of EXIF metadata; this frontend does not persist user images.
                    </span>
                  </p>
                )}
              </div>

              {/* Step 2: Wardrobe Category, Pipeline Routing & Generate */}
              <div className="rounded-3xl bg-white border border-slate-200/80 p-5 shadow-xs space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-[#9B1D4A] text-white text-xs font-bold flex items-center justify-center">
                      2
                    </span>
                    <h2 className="text-sm font-bold text-slate-900">Choose or Upload Outfit</h2>
                  </div>
                  <label className="inline-flex items-center gap-1 text-xs font-semibold text-[#9B1D4A] hover:underline cursor-pointer">
                    <Upload className="w-3.5 h-3.5" />
                    Upload Own
                    <input type="file" accept="image/*" className="hidden" onChange={handleGarmentUpload} />
                  </label>
                </div>

                {/* 3-Category Wardrobe Switcher */}
                <div className="grid grid-cols-3 gap-1.5 p-1 rounded-2xl bg-[#FAF8F5] border border-slate-200/70">
                  {WARDROBE_CATEGORIES.map((cat) => {
                    const active = wardrobeCategory === cat.id
                    return (
                      <button
                        key={cat.id}
                        onClick={() => handleSelectCategory(cat.id)}
                        className={`py-2 px-2 rounded-xl text-[11px] sm:text-xs font-bold transition cursor-pointer ${
                          active
                            ? 'bg-white text-slate-900 shadow-xs border border-slate-200/80'
                            : 'text-slate-500 hover:text-slate-800'
                        }`}
                      >
                        {cat.label}
                      </button>
                    )
                  })}
                </div>

                {/* Predefined Wardrobe Grid */}
                <div className="grid grid-cols-3 gap-3">
                  {activeCategoryObj.items.map((kurta) => {
                    const active = selectedKurtaId === kurta.id
                    return (
                      <button
                        key={kurta.id}
                        onClick={() => handleSelectPredefinedKurta(kurta)}
                        className={`group rounded-2xl p-2 text-left border transition cursor-pointer ${
                          active
                            ? 'border-[#9B1D4A] bg-rose-50/40 ring-1 ring-[#9B1D4A]'
                            : 'border-slate-200 bg-white hover:border-slate-300'
                        }`}
                      >
                        <div className="aspect-[3/4] rounded-xl bg-white overflow-hidden border border-slate-100 mb-2">
                          <img
                            src={kurta.image}
                            alt={kurta.name}
                            className="w-full h-full object-contain p-1 group-hover:scale-105 transition duration-300"
                          />
                        </div>
                        <p className="text-[11px] font-bold text-slate-900 leading-tight line-clamp-2">
                          {kurta.name}
                        </p>
                      </button>
                    )
                  })}
                </div>

                {/* Custom Garment Preview Bar */}
                {selectedKurtaId === 'custom' && garmentPreview && (
                  <div className="flex items-center gap-3.5 p-3 rounded-2xl bg-[#FAF8F5] border border-slate-200/80">
                    <img
                      src={canonicalPreview || garmentPreview}
                      alt="Custom outfit"
                      className="w-16 h-20 object-contain rounded-lg bg-white border border-slate-200"
                    />
                    <div className="flex-1 min-w-0 space-y-1.5">
                      <p className="text-xs font-bold text-slate-800">Custom Outfit Uploaded</p>
                      <div className="inline-flex rounded-full bg-white border border-slate-200 p-0.5">
                        {[
                          { id: 'flat-lay', label: 'Product / Flat-Lay' },
                          { id: 'model', label: 'Model-Worn' },
                        ].map((pt) => (
                          <button
                            key={pt.id}
                            onClick={() => {
                              invalidateStaleResult()
                              setGarmentPhotoType(pt.id)
                            }}
                            className={`px-2 py-0.5 rounded-full text-[10px] font-semibold cursor-pointer ${
                              garmentPhotoType === pt.id ? 'bg-slate-900 text-white' : 'text-slate-600'
                            }`}
                          >
                            {pt.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Explicit Model & Pipeline Selector */}
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { id: 'catvton', label: 'CatVTON + V4 Mask', sub: 'Mask-guided (~63s T4)' },
                    { id: 'fashn_vton_1_5', label: 'FASHN VTON 1.5', sub: 'Mask-free (model/flat-lay)' },
                  ].map((m) => (
                    <button
                      key={m.id}
                      onClick={() => {
                        invalidateStaleResult()
                        setSelectedModel(m.id)
                      }}
                      className={`p-2.5 rounded-2xl border text-left transition cursor-pointer ${
                        selectedModel === m.id
                          ? 'border-slate-900 bg-slate-900 text-white'
                          : 'border-slate-200 bg-[#FAF8F5] text-slate-800 hover:bg-slate-100'
                      }`}
                    >
                      <p className="text-xs font-bold">{m.label}</p>
                      <p className={`text-[10px] mt-0.5 ${selectedModel === m.id ? 'text-slate-300' : 'text-slate-500'}`}>
                        {m.sub}
                      </p>
                    </button>
                  ))}
                </div>

                {/* Active Category & Release Status Banner */}
                <div className={`rounded-2xl border px-3.5 py-2.5 flex items-center justify-between gap-2 ${activeModeBadge.tone}`}>
                  <div className="min-w-0">
                    <p className="text-xs font-bold leading-tight">{activeModeBadge.title}</p>
                    <p className="text-[11px] opacity-85 mt-0.5 leading-snug">{activeModeBadge.desc}</p>
                  </div>
                  <button
                    onClick={handleSwapCrossAge}
                    title="Toggle Adult vs Girl Child for Cross-Age Try-On"
                    className="shrink-0 inline-flex items-center gap-1 text-[11px] font-bold px-2.5 py-1.5 rounded-full bg-white/90 text-slate-800 border border-slate-200 hover:bg-white transition cursor-pointer"
                  >
                    <Repeat className="w-3 h-3" />
                    {personAgeGroup === 'adult' ? 'Try on Child' : 'Try on Adult'}
                  </button>
                </div>

                <div className="rounded-2xl bg-slate-50 border border-slate-200/80 px-3 py-2 text-[11px] text-slate-600 leading-relaxed">
                  {selectedModel === 'catvton' ? (
                    <span>
                      <strong>Dupatta behaviour (fixed):</strong> V4 mask always excludes detected recipient dupatta
                      pixels (class 3). It does <em>not</em> transfer a donor outfit’s dupatta.
                    </span>
                  ) : (
                    <span>
                      <strong>Dupatta behaviour:</strong> FASHN VTON 1.5 is mask-free; dupatta handling is internal
                      and not externally toggled.
                    </span>
                  )}
                </div>

                <button
                  onClick={runVton}
                  disabled={loading}
                  className="w-full flex items-center justify-center gap-2 py-3.5 rounded-full bg-gradient-to-r from-[#0F5C4D] to-[#9B1D4A] text-white font-bold text-sm hover:opacity-95 disabled:opacity-50 shadow-md transition cursor-pointer"
                >
                  {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                  {jobStage === 'uploading'
                    ? 'Uploading to V4 GPU...'
                    : jobStage === 'processing'
                      ? 'Running Neural Inference...'
                      : 'Generate Live Virtual Try-On'}
                </button>
              </div>

              {/* Honest Research & Input Tips Card */}
              <div className="rounded-3xl bg-white border border-rose-100 p-5 space-y-2.5">
                <p className="text-xs font-bold uppercase tracking-wider text-[#9B1D4A]">
                  Input Guidance &amp; Scope
                </p>
                <ul className="space-y-1.5 text-xs text-slate-600">
                  {STUDIO_TIPS.map((tip) => (
                    <li key={tip} className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#0F5C4D] shrink-0" />
                      <span>{tip}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* RIGHT COLUMN: Studio Result Canvas, Download & Model Provenance */}
            <div className="lg:col-span-7 rounded-3xl bg-white border border-slate-200/80 p-5 sm:p-6 shadow-xs space-y-5">
              {/* Pose Tabs (Honest: Additional AI angles clearly marked unavailable when Gemini is disabled) */}
              <div className="grid grid-cols-3 gap-2.5">
                {STUDIO_POSES.map((pose) => {
                  const isFront = pose.id === 'front'
                  const unavailable = !isFront && !geminiEnabled
                  const active = (vtonResult ? resultPose : 'front') === pose.id
                  const ready = Boolean(generatedPoses[pose.id])
                  return (
                    <button
                      key={pose.id}
                      onClick={() => (vtonResult ? handleSelectResultPose(pose.id) : null)}
                      disabled={unavailable && Boolean(vtonResult)}
                      className={`p-3 rounded-2xl text-left border transition ${
                        unavailable
                          ? 'bg-slate-50 text-slate-400 border-slate-200/70 cursor-not-allowed'
                          : active
                            ? 'bg-slate-900 text-white border-slate-900 shadow-xs cursor-pointer'
                            : 'bg-[#FAF8F5] text-slate-800 border-slate-200/80 hover:bg-slate-100 cursor-pointer'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-xs font-bold">{pose.label}</span>
                        {vtonResult &&
                          (ready ? (
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                          ) : unavailable ? (
                            <span className="text-[9px] uppercase font-bold">Off</span>
                          ) : (
                            <Wand2 className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                          ))}
                      </div>
                      <p className={`text-[10px] sm:text-[11px] mt-0.5 truncate ${active ? 'text-slate-300' : 'text-slate-500'}`}>
                        {unavailable ? 'Optional AI view disabled' : pose.desc}
                      </p>
                    </button>
                  )
                })}
              </div>

              {/* Studio Display Stage */}
              {vtonResult ? (
                <div className="space-y-4">
                  <div className="relative rounded-2xl bg-[#FAF8F5] border border-slate-200/70 overflow-hidden flex items-center justify-center min-h-[480px]">
                    {poseLoading ? (
                      <div className="flex flex-col items-center gap-3 text-slate-500 py-20">
                        <Loader2 className="w-8 h-8 animate-spin text-[#9B1D4A]" />
                        <p className="text-xs font-semibold">Generating optional angle preview...</p>
                      </div>
                    ) : (
                      <img
                        src={
                          showOriginalToggle
                            ? preppedPersonPreview || personPreview
                            : generatedPoses[resultPose] || vtonResult.result_image
                        }
                        alt="Live virtual try-on output"
                        className="w-full max-h-[540px] object-contain"
                      />
                    )}
                    <span className="absolute top-4 left-4 px-3 py-1 rounded-full bg-white/95 backdrop-blur border border-slate-200 text-[11px] font-bold text-slate-800 shadow-xs">
                      Live Output • {vtonResult.model_id}
                      {vtonResult.seconds ? ` • ${vtonResult.seconds}s` : ''}
                    </span>
                    <div className="absolute bottom-4 right-4 flex items-center gap-2">
                      <button
                        onClick={handleDownloadResult}
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-[#0F5C4D] text-white text-xs font-bold shadow-sm hover:bg-[#0b473b] cursor-pointer"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Download PNG
                      </button>
                      <button
                        onClick={() => setShowOriginalToggle(!showOriginalToggle)}
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-white/95 backdrop-blur border border-slate-200 text-xs font-bold text-slate-800 shadow-sm hover:bg-white cursor-pointer"
                      >
                        <ArrowLeftRight className="w-3.5 h-3.5" />
                        {showOriginalToggle ? 'Show Try-On' : 'Compare Original'}
                      </button>
                    </div>
                  </div>

                  {/* Verified Model Provenance Bar */}
                  <div className="rounded-2xl bg-slate-50 border border-slate-200/80 p-3.5 text-[11px] text-slate-600 space-y-1">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="inline-flex items-center gap-1.5 font-bold text-slate-800">
                        <Info className="w-3.5 h-3.5 text-[#0F5C4D]" />
                        Run Provenance
                      </span>
                      <button
                        onClick={invalidateStaleResult}
                        className="font-semibold text-[#9B1D4A] hover:underline cursor-pointer"
                      >
                        Clear &amp; Start New Look
                      </button>
                    </div>
                    <p className="font-mono text-[10px] text-slate-600 break-all">
                      backend={vtonResult.engine || 'colab_gpu_live'} | model={vtonResult.model_id || vtonSnapshot?.model_id} | parser=
                      {vtonResult.parser_id || 'internal'} | category={vtonSnapshot?.category || vtonResult.category} | mode=
                      {vtonSnapshot?.tryon_mode || vtonResult.tryon_mode} | photo_type=
                      {vtonSnapshot?.garment_photo_type || vtonResult.garment_photo_type} | seed=
                      {vtonSnapshot?.seed ?? vtonResult.seed ?? 42}
                      {vtonResult.load_seconds != null ? ` | load=${vtonResult.load_seconds}s` : ''}
                      {vtonResult.inference_seconds != null ? ` | infer=${vtonResult.inference_seconds}s` : ''}
                    </p>
                    <p className="font-mono text-[10px] text-slate-500 break-all">
                      checkpoint: {formatCheckpointProvenance(vtonResult.checkpoint_provenance)}
                    </p>
                    <p className="text-[11px] text-slate-500">
                      Visual appearance preview only — not a physical tailoring or sizing guarantee.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="rounded-2xl bg-[#FAF8F5] border border-slate-200/70 p-6 min-h-[480px] flex flex-col items-center justify-center text-center space-y-6">
                  {personPreview ? (
                    <div className="space-y-4 w-full max-w-md">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="rounded-2xl bg-white p-2.5 border border-slate-200 shadow-xs">
                          <p className="text-[11px] font-bold text-slate-500 mb-2">
                            {personAgeGroup === 'child' ? 'Girl Child Target' : 'Adult Target'}
                          </p>
                          <img
                            src={preppedPersonPreview || personPreview}
                            alt="Person ready"
                            className="w-full h-64 object-contain rounded-xl"
                          />
                        </div>
                        <div className="rounded-2xl bg-white p-2.5 border border-slate-200 shadow-xs">
                          <p className="text-[11px] font-bold text-slate-500 mb-2">
                            {effectiveGarmentCategory === 'lehenga'
                              ? 'Wedding Lehenga Ref'
                              : effectiveGarmentCategory === 'kids'
                                ? 'Kids Outfit Ref'
                                : 'Kurta Ref'}
                          </p>
                          <img
                            src={canonicalPreview || garmentPreview}
                            alt="Outfit ready"
                            className="w-full h-64 object-contain rounded-xl"
                          />
                        </div>
                      </div>
                      <p className="text-xs text-slate-500">
                        {isGpuConnected
                          ? `V4 GPU Ready (${selectedModel}) — click Generate Live Virtual Try-On.`
                          : 'Frontend standby: connect your authenticated V4 GPU worker in the top bar for live inference, or view static examples in Saved Showcase.'}
                      </p>
                    </div>
                  ) : (
                    <div className="max-w-sm space-y-3">
                      <div className="w-14 h-14 rounded-2xl bg-white border border-slate-200 shadow-xs flex items-center justify-center mx-auto text-[#0F5C4D]">
                        <Shirt className="w-7 h-7" />
                      </div>
                      <h3 className="font-display text-xl font-bold text-slate-900">Your Dressing Room Awaits</h3>
                      <p className="text-xs text-slate-500 leading-relaxed">
                        Select <strong>Adult Model</strong> or <strong>Girl Child</strong> (or upload a photo), choose a
                        category (<strong>Women’s Kurtas</strong>, <strong>Kids</strong>, or{' '}
                        <strong>Wedding Lehengas</strong>), connect your V4 GPU worker, and generate a live preview.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: CLEARLY LABELED STATIC SAVED SHOWCASE */}
        {activeTab === 'demo' && (
          <div className="grid lg:grid-cols-12 gap-8 items-start">
            <div className="lg:col-span-5 rounded-3xl bg-white border border-slate-200/80 p-6 shadow-xs space-y-5">
              <div>
                <p className="text-xs font-bold uppercase tracking-widest text-[#0F5C4D]">
                  Static Pre-Saved Showcase (Not Live Inference)
                </p>
                <h2 className="font-display text-2xl font-bold text-slate-900 mt-1">
                  Emerald Zari Embroidered Kurta
                </h2>
                <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
                  These three editorial studio images are static demonstration assets illustrating the target 3-pose
                  presentation layout. For live neural try-on, switch to the Live Virtual Try-On tab with a V4 GPU worker.
                </p>
              </div>

              <div className="space-y-2.5">
                {STUDIO_POSES.map((pose) => (
                  <button
                    key={pose.id}
                    onClick={() => setDemoPose(pose.id)}
                    className={`w-full p-4 rounded-2xl border text-left transition flex items-center justify-between cursor-pointer ${
                      demoPose === pose.id
                        ? 'border-slate-900 bg-slate-900 text-white shadow-sm'
                        : 'border-slate-200 bg-[#FAF8F5] text-slate-800 hover:bg-slate-100'
                    }`}
                  >
                    <div>
                      <p className="text-sm font-bold">{pose.label}</p>
                      <p className={`text-xs mt-0.5 ${demoPose === pose.id ? 'text-slate-300' : 'text-slate-500'}`}>
                        {pose.desc}
                      </p>
                    </div>
                    <CheckCircle2
                      className={`w-5 h-5 shrink-0 ${demoPose === pose.id ? 'text-emerald-400' : 'text-slate-300'}`}
                    />
                  </button>
                ))}
              </div>

              <div className="pt-2">
                <button
                  onClick={() => setActiveTab('photo')}
                  className="w-full py-3 rounded-full bg-[#0F5C4D] text-white text-sm font-semibold hover:bg-[#0b473b] transition cursor-pointer"
                >
                  Go to Live Virtual Try-On →
                </button>
              </div>
            </div>

            <div className="lg:col-span-7 rounded-3xl bg-white border border-slate-200/80 p-5 shadow-xs">
              <div className="relative rounded-2xl overflow-hidden bg-[#FAF8F5] flex items-center justify-center min-h-[480px]">
                <img
                  key={activeDemoPoseObj.id}
                  src={activeDemoPoseObj.demoImage}
                  alt={activeDemoPoseObj.label}
                  className="w-full max-h-[600px] object-contain"
                />
                <span className="absolute top-4 left-4 px-3.5 py-1.5 rounded-full bg-white/95 backdrop-blur text-xs font-bold text-slate-900 shadow-xs">
                  Static Example • {activeDemoPoseObj.label}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: GARMENT CUTOUT */}
        {activeTab === 'segment' && (
          <div className="rounded-3xl bg-white border border-slate-200/80 p-6 shadow-xs space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-bold text-slate-900">4-Class Pakistani Outfit Cutout</h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Compare V4 Universal SegFormer-B2 (adapted) vs. pretrained baseline on any outfit photo.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2.5">
                <select
                  value={segVariant}
                  onChange={(e) => {
                    invalidateSegResult()
                    setSegVariant(e.target.value)
                  }}
                  className="text-xs px-3.5 py-2 rounded-full border border-slate-300 bg-white font-medium text-slate-700"
                >
                  <option value="adapted">V4 Universal 4-Class Parser (Adapted)</option>
                  <option value="baseline">Pretrained Baseline (18-to-4 Mapped)</option>
                </select>
                <label className="cursor-pointer text-xs font-semibold px-4 py-2 rounded-full bg-[#FAF8F5] border border-slate-300 hover:bg-slate-100">
                  Upload Outfit
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0]
                      if (f) {
                        invalidateSegResult()
                        setSegFile(f)
                        setSegPreview(URL.createObjectURL(f))
                      }
                    }}
                  />
                </label>
                <button
                  onClick={runSegment}
                  disabled={segLoading}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold px-5 py-2 rounded-full bg-[#0F5C4D] text-white hover:bg-[#0b473b] disabled:opacity-50 cursor-pointer"
                >
                  {segLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Scissors className="w-3.5 h-3.5" />}
                  Extract Cutout
                </button>
              </div>
            </div>

            {segResult ? (
              <div className="space-y-3">
                <div className="grid md:grid-cols-3 gap-5">
                  {[
                    { label: 'Original', src: segPreview },
                    { label: `4-Class Overlay (${segResult.model_id})`, src: segResult.overlay_image },
                    { label: 'Isolated Upper Garment', src: segResult.kurta_cutout_image },
                  ].map((card) => (
                    <div key={card.label} className="rounded-2xl bg-[#FAF8F5] border border-slate-200 p-3 space-y-2">
                      <p className="text-xs font-bold text-slate-700">{card.label}</p>
                      <img src={card.src} alt={card.label} className="w-full h-72 object-contain rounded-xl bg-white" />
                    </div>
                  ))}
                </div>
                {segResult.checkpoint_sha256 && (
                  <p className="font-mono text-[10px] text-slate-500">
                    parser={segResult.model_id} | ckpt_sha256={String(segResult.checkpoint_sha256).slice(0, 16)} |{' '}
                    {segResult.seconds}s
                  </p>
                )}
              </div>
            ) : (
              <div className="rounded-2xl bg-[#FAF8F5] border border-dashed border-slate-300 p-10 text-center text-xs text-slate-500">
                {segPreview ? (
                  <img src={segPreview} alt="Preview" className="h-64 mx-auto object-contain rounded-xl" />
                ) : (
                  'Upload a full-body Pakistani outfit photo above and click Extract Cutout.'
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
