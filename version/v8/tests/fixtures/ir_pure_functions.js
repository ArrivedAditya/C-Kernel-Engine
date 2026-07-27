// IR Visualizer — pure utility functions exported for testing.
// These are canonical copies of functions defined in ir_visualizer.html.
// Keep in sync: if you change the source, update this fixture and bump
// the hash in ir_visualizer_contract.json.
//
// Usage: node version/v8/tests/fixtures/ir_pure_functions.js
// (self-test when run directly)

function formatBytes(bytes) {
    if (bytes >= 1024*1024*1024) return (bytes / (1024*1024*1024)).toFixed(2) + ' GB';
    if (bytes >= 1024*1024) return (bytes / (1024*1024)).toFixed(2) + ' MB';
    if (bytes >= 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return bytes + ' B';
}

function normalizeShapeInput(shape) {
    if (shape === null || shape === undefined) return [];
    if (Array.isArray(shape)) {
        return shape
            .filter(v => v !== null && v !== undefined && String(v).trim() !== '')
            .map(v => String(v));
    }
    if (typeof shape === 'number' && Number.isFinite(shape)) {
        return [String(shape)];
    }
    if (typeof shape === 'string') {
        const s = shape.trim();
        if (!s) return [];
        const wrapped = (s.startsWith('[') && s.endsWith(']')) || (s.startsWith('(') && s.endsWith(')'));
        if (wrapped) {
            const inner = s.slice(1, -1).trim();
            if (!inner) return [];
            try {
                const parsed = JSON.parse(s.replace(/\(/g, '[').replace(/\)/g, ']'));
                if (Array.isArray(parsed)) return parsed.map(v => String(v));
            } catch (_) {}
            return inner.split(/[x×,\s]+/).filter(Boolean);
        }
        const parts = s.split(/[x×,\s]+/).filter(Boolean);
        if (parts.length > 1) return parts;
        return [s];
    }
    if (typeof shape === 'object') {
        if (Array.isArray(shape.shape)) return normalizeShapeInput(shape.shape);
        if (Array.isArray(shape.dims)) return normalizeShapeInput(shape.dims);
        if (Array.isArray(shape.dimensions)) return normalizeShapeInput(shape.dimensions);
        const numericKeys = Object.keys(shape)
            .filter(k => /^\d+$/.test(k))
            .sort((a, b) => Number(a) - Number(b));
        if (numericKeys.length > 0) return numericKeys.map(k => String(shape[k]));
        const scalarValues = Object.values(shape).filter(v => typeof v === 'number' || typeof v === 'string');
        if (scalarValues.length > 0 && scalarValues.length <= 4) return scalarValues.map(v => String(v));
        try { return [JSON.stringify(shape)]; } catch (_) { return ['[object]']; }
    }
    return [String(shape)];
}

function formatShapeDisplay(shape, separator = ' × ') {
    const dims = normalizeShapeInput(shape);
    if (dims.length === 0) return '-';
    if (dims.length === 1) return `[${dims[0]}]`;
    return dims.join(separator);
}

function normalizeMode(mode) {
    return mode === 'prefill' ? 'prefill' : 'decode';
}

function escapeHtml(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function quoteShell(value) {
    const s = String(value || '');
    if (!s) return '';
    if (/^[A-Za-z0-9_./:-]+$/.test(s)) return s;
    return `"${s.replace(/"/g, '\\"')}"`;
}

function normalizePathString(value) {
    return String(value || '').replace(/\\/g, '/').replace(/\/+$/, '');
}

function pathDirname(value) {
    const p = normalizePathString(value);
    if (!p) return '';
    const idx = p.lastIndexOf('/');
    if (idx <= 0) return idx === 0 ? '/' : '';
    return p.slice(0, idx);
}

function extractGgufStem(modelInput) {
    const raw = String(modelInput || '');
    if (!raw) return '';
    const clean = raw.split('?')[0].replace(/\\/g, '/');
    const idx = clean.lastIndexOf('/');
    const base = idx >= 0 ? clean.slice(idx + 1) : clean;
    if (!base.toLowerCase().endsWith('.gguf')) return '';
    return base.slice(0, -5);
}

function relativePathFromTo(fromDir, toPath) {
    const from = normalizePathString(fromDir);
    const to = normalizePathString(toPath);
    if (!from || !to) return null;
    if (!from.startsWith('/') || !to.startsWith('/')) return null;
    const a = from.split('/').filter(Boolean);
    const b = to.split('/').filter(Boolean);
    let i = 0;
    while (i < a.length && i < b.length && a[i] === b[i]) i += 1;
    const up = new Array(Math.max(0, a.length - i)).fill('..');
    const down = b.slice(i);
    const parts = up.concat(down);
    return parts.length ? parts.join('/') : '.';
}

// ── X-Ray tab pure helpers (mirrors of ir_visualizer.html) ──────────────────

function xrayReportKindLabel(report) {
    if (!report || typeof report !== 'object') return 'X-Ray Report';
    const schema = String(report.schema || '');
    if (schema === 'cke.whisper_encoder_pytorch_xray') return 'Whisper Encoder X-Ray';
    if (schema === 'cke.xray_ranking_report') return 'Ranking Report';
    if (schema === 'cke.xray_execution_trace') return 'Execution Trace';
    if (schema === 'cke.xray_execution_state_report') return 'Execution State Report';
    if (schema === 'cke.xray.decoder_pytorch') return 'Decoder X-Ray';
    if (schema === 'cke.xray_orchestration_report') return 'Vision X-Ray (orchestration)';
    if (schema === 'cke.xray_numerical_report') return 'Numerical Parity X-Ray';
    return schema || 'X-Ray Report';
}

function xrayBackendLabel(report) {
    if (!report || typeof report !== 'object') return 'unknown';
    const schema = String(report.schema || '');
    if (schema === 'cke.whisper_encoder_pytorch_xray') return 'PyTorch';
    if (schema === 'cke.xray.decoder_pytorch') return 'PyTorch';
    if (schema === 'cke.xray_ranking_report') return 'ggml-oracle';
    if (schema === 'cke.xray_execution_trace') return String(report.backend || 'unknown');
    const oracle = report.oracle_backend
        || (report.final_report && report.final_report.oracle_backend)
        || (report.provenance && report.provenance.pytorch ? 'pytorch' : null)
        || report.backend;
    const b = String(oracle || 'unknown').toLowerCase();
    if (b.indexOf('llama') >= 0) return 'llama.cpp';
    if (b.indexOf('torch') >= 0) return 'PyTorch';
    if (b.indexOf('ggml') >= 0) return 'ggml-oracle';
    return oracle ? String(oracle) : 'unknown';
}

function xrayReportStatus(report) {
    if (!report || typeof report !== 'object') return 'unknown';
    if (typeof report.status === 'string' && report.status) return report.status;
    if (Array.isArray(report.checks) && report.checks.length) {
        return report.checks.every(c => c && c.status === 'pass') ? 'pass' : 'fail';
    }
    return 'unknown';
}

// Normalize any checkpoint-table X-ray report into chart rows:
// [{stop, label, layer, maxAbs, rmse, byteExact, exactRatio, raw}]
function xrayDriftRows(report) {
    if (!report || typeof report !== 'object') return [];
    const cps = Array.isArray(report.checkpoints) ? report.checkpoints : null;
    if (cps && cps.length && cps[0] && cps[0].metrics) {
        return cps.map((cp, i) => ({
            stop: (cp.stop != null ? cp.stop : i),
            label: String(cp.checkpoint != null ? cp.checkpoint : ('stop ' + i)),
            layer: (cp.layer != null ? cp.layer : null),
            maxAbs: (cp.metrics && cp.metrics.max_abs != null) ? Number(cp.metrics.max_abs) : null,
            rmse: (cp.metrics && cp.metrics.rmse != null) ? Number(cp.metrics.rmse) : null,
            byteExact: !!(cp.metrics && cp.metrics.byte_exact),
            exactRatio: (cp.metrics && cp.metrics.exact_ratio != null) ? Number(cp.metrics.exact_ratio) : null,
            raw: cp,
        }));
    }
    const num = (report.final_report && report.final_report.drift_progression) ? report.final_report
        : (report.drift_progression ? report : null);
    if (num && Array.isArray(num.drift_progression.checkpoints)) {
        return num.drift_progression.checkpoints.map((cp, i) => ({
            stop: i,
            label: String(cp.checkpoint_id != null ? cp.checkpoint_id : ('stop ' + i)),
            layer: (cp.resolved_execution && cp.resolved_execution.layer != null) ? cp.resolved_execution.layer : null,
            maxAbs: (cp.max_abs != null) ? Number(cp.max_abs) : null,
            rmse: (cp.rmse != null) ? Number(cp.rmse) : null,
            byteExact: !!cp.byte_exact,
            exactRatio: null,
            raw: cp,
        }));
    }
    return [];
}

function buildXrayRunbookHtml(runCtx) {
    const hasRun = !!(runCtx && runCtx.runDir);
    const runDir = hasRun ? quoteShell(runCtx.runDir) : '<run_dir>';
    const out = (name) => hasRun ? quoteShell(runCtx.runDir + '/' + name) : ('<run_dir>/' + name);
    const whisperCmd = 'python3 version/v8/scripts/compare_whisper_encoder_pytorch_v8.py \\\n'
        + '  --run-dir ' + runDir + ' \\\n'
        + '  --checkpoint <hf_whisper_checkpoint> \\\n'
        + '  --stops key \\\n'
        + '  --output ' + out('whisper-encoder-xray.json');
    const visionLlamaCmd = 'make xray-vision-parity BACKEND=llamacpp GGUF=<mmproj.gguf> XRAY_OUTPUT_DIR=build/xray/qwen3vl_llamacpp';
    const visionTorchCmd = 'make xray-vision-parity BACKEND=pytorch CHECKPOINT=<hf_qwen3vl_ckpt> RUNTIME_DIR=<runtime_dir> WEIGHTS_BUMP=<weights.bump> CALL_IR=<call.json> XRAY_OUTPUT_DIR=build/xray/qwen3vl_bf16';
    const decoderCmd = 'python3 version/v8/scripts/xray_decoder_pytorch_v8.py \\\n'
        + '  --checkpoint <hf_decoder_ckpt> --runtime <libdecoder.so> --call-ir <call.json> \\\n'
        + '  --token-ids 1,2,3 --output-dir build/xray/decoder_pytorch';
    const rankingCmd = 'python3 version/v8/scripts/normalize_xray_ranking_report_v8.py \\\n'
        + '  --input <logits_trace.json> --kind teacher_forced \\\n'
        + '  --output ' + out('xray_ranking_report.json');
    const stateCmd = 'python3 version/v8/scripts/xray_execution_state_v8.py \\\n'
        + '  --subject-trace <ck_trace.json> --oracle-trace <oracle_trace.json> \\\n'
        + '  --output ' + out('xray_execution_state_report.json');
    const refreshCmd = hasRun
        ? 'python3 version/v8/tools/open_ir_visualizer_v8.py --generate --run ' + runDir + ' --html-only --strict-run-artifacts --output ' + out('ir_report.html')
        : 'python3 version/v8/tools/open_ir_visualizer_v8.py --generate --run <run_dir> --html-only --strict-run-artifacts --output <run_dir>/ir_report.html';
    const block = (title, cmd) => ''
        + '<div style="margin-top:0.45rem;">' + escapeHtml(title) + '</div>'
        + '<pre style="font-size:0.72rem;white-space:pre-wrap;margin-top:0.35rem;">' + escapeHtml(cmd) + '</pre>';
    return ''
        + '<div style="margin-top:0.5rem; padding:0.75rem 1rem; background:rgba(255,180,0,0.08); border:1px solid rgba(255,180,0,0.3); border-radius:8px; font-size:0.72rem; color:var(--text-secondary); line-height:1.6;">'
        + '<strong style="color:var(--orange);">No X-ray artifacts loaded.</strong>'
        + '<div style="margin-top:0.35rem;">Produce backend parity X-ray reports with any of these, then refresh this report:</div>'
        + block('Whisper encoder vs PyTorch (per-checkpoint drift):', whisperCmd)
        + block('Qwen3-VL vision parity vs llama.cpp:', visionLlamaCmd)
        + block('Qwen3-VL vision parity vs PyTorch (bf16):', visionTorchCmd)
        + block('Decoder parity vs PyTorch:', decoderCmd)
        + block('Logits ranking report (ggml oracle):', rankingCmd)
        + block('Execution-state divergence stages:', stateCmd)
        + block('Refresh this report after artifacts land:', refreshCmd)
        + '</div>';
}

// Export for test harness (CommonJS for Node.js compatibility)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        formatBytes, normalizeShapeInput, formatShapeDisplay,
        normalizeMode, escapeHtml, quoteShell, normalizePathString,
        pathDirname, extractGgufStem, relativePathFromTo,
        xrayReportKindLabel, xrayBackendLabel, xrayReportStatus,
        xrayDriftRows, buildXrayRunbookHtml
    };
}
