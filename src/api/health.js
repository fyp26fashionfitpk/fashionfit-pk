export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*')
  return res.status(200).json({
    schema_version: 'v4.0',
    status: 'ok',
    mode: 'frontend_standby',
    process_alive: true,
    gpu_accessible: false,
    model_loaded: false,
    queue_accepting: false,
    colab_gpu_url: null,
    supported_categories: [],
    note: 'Static frontend gateway is alive. Connect an authenticated V4 GPU worker for live neural try-on.',
  })
}
