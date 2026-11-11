const BATTERY_THRESHOLDS = {
  good: 70,
  caution: 35,
};

const STORAGE_THRESHOLDS = {
  good: 70,
  caution: 85,
};

const CARD_CLASSES = [
  'is-good',
  'is-live',
  'is-caution',
  'is-low',
  'is-waiting',
  'is-stale',
  'is-offline',
];

const config = window.MONITOR_CONFIG || {
  statusEndpoint: '/api/status',
  graphEndpoint: '/api/graph',
  frameEndpoint: '/api/frame',
  debugFrameGrayscaleEndpoint: '/api/frame/grayscale',
  debugFrameBlurEndpoint: '/api/frame/blur',
  debugFrameEdgeEndpoint: '/api/frame/edge',
  debugImageEnabled: false,
  placeholderUrl: '/api/frame/placeholder',
  refreshIntervalMs: 1000,
  imageRefreshIntervalMs: 150,
};

const LANE_OFFSET_MAX_PX = 320;

const elements = {
  batteryCard: document.getElementById('battery-card'),
  batteryChip: document.getElementById('battery-chip'),
  batteryValue: document.getElementById('battery-value'),
  batteryUpdated: document.getElementById('battery-updated'),
  batteryMeterFill: document.getElementById('battery-meter-fill'),
  imageCard: document.getElementById('image-card'),
  imageChip: document.getElementById('image-chip'),
  imageUpdated: document.getElementById('image-updated'),
  imageResolution: document.getElementById('image-resolution'),
  cameraFrame: document.getElementById('camera-frame'),
  debugFrameGrayscale: document.getElementById('debug-frame-grayscale'),
  debugFrameBlur: document.getElementById('debug-frame-blur'),
  debugFrameEdge: document.getElementById('debug-frame-edge'),
  recordBadge: document.getElementById('record-badge'),
  recordBadgeLabel: document.getElementById('record-badge-label'),
  controlCard: document.getElementById('control-card'),
  controlChip: document.getElementById('control-chip'),
  controlUpdated: document.getElementById('control-updated'),
  throttleBar: document.getElementById('throttle-bar'),
  throttleBarFill: document.getElementById('throttle-bar-fill'),
  throttleValue: document.getElementById('throttle-value'),
  steeringBar: document.getElementById('steering-bar'),
  steeringBarFill: document.getElementById('steering-bar-fill'),
  steeringValue: document.getElementById('steering-value'),
  storageCard: document.getElementById('storage-card'),
  storageChip: document.getElementById('storage-chip'),
  storageValue: document.getElementById('storage-value'),
  storageUpdated: document.getElementById('storage-updated'),
  storageMeterFill: document.getElementById('storage-meter-fill'),
  storageDetail: document.getElementById('storage-detail'),
  graphCard: document.getElementById('graph-card'),
  graphChip: document.getElementById('graph-chip'),
  graphCanvas: document.getElementById('ros-graph-canvas'),
  graphUpdated: document.getElementById('graph-updated'),
  graphSummary: document.getElementById('graph-summary'),
  laneCard: document.getElementById('lane-card'),
  laneChip: document.getElementById('lane-chip'),
  laneUpdated: document.getElementById('lane-updated'),
  laneOffsetBar: document.getElementById('lane-offset-bar'),
  laneOffsetBarFill: document.getElementById('lane-offset-bar-fill'),
  laneOffsetValue: document.getElementById('lane-offset-value'),
  yellowDetectedChip: document.getElementById('yellow-detected-chip'),
  lineFrameBev: document.getElementById('line-frame-bev'),
  lineFrameSliding: document.getElementById('line-frame-sliding'),
};

let imageRequestInFlight = false;
let debugImageRequestInFlight = false;
let lineImageRequestInFlight = false;

function clampPercent(value) {
  return Math.max(0, Math.min(100, Number(value) || 0));
}

function clampControl(value) {
  return Math.max(-1, Math.min(1, Number(value) || 0));
}

function formatUpdatedAt(updatedAt) {
  if (!updatedAt) {
    return 'Waiting for message';
  }

  const date = new Date(updatedAt);

  if (Number.isNaN(date.getTime())) {
    return 'Invalid timestamp';
  }

  return date.toLocaleString();
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[character]));
}

function getGraphLabel(state) {
  switch (state) {
    case 'live':
      return 'LIVE';
    case 'stale':
      return 'STALE';
    default:
      return 'WAITING';
  }
}

function renderGraphNodeCard(node, role) {
  if (!node) {
    return '<span class="ros-graph__node-card ros-graph__node-card--placeholder">--</span>';
  }

  return `
    <span class="ros-graph__node-card ros-graph__node-card--${role}" title="${escapeHtml(node.label)}">
      ${escapeHtml(node.label || node.id)}
    </span>
  `;
}

function renderGraphEdge(edge, topicNode) {
  const topicLabel = topicNode?.label || edge.topic || 'topic';
  return `
    <div class="ros-graph__row">
      <div class="ros-graph__column ros-graph__column--source">
        ${renderGraphNodeCard(edge.source_node, 'source')}
      </div>
      <div class="ros-graph__edge" aria-label="${escapeHtml(topicLabel)}">
        <span class="ros-graph__edge-label" title="${escapeHtml(topicLabel)}">
          ${escapeHtml(topicLabel)}
        </span>
      </div>
      <div class="ros-graph__column ros-graph__column--target">
        ${renderGraphNodeCard(edge.target_node, 'target')}
      </div>
    </div>
  `;
}

function renderGraphPublisherTopicRow(edge) {
  const topicLabel = edge.topic_node?.label || edge.topic || 'topic';
  return `
    <div class="ros-graph__publisher-topic-row">
      <div class="ros-graph__edge" aria-label="${escapeHtml(topicLabel)}">
        <span class="ros-graph__edge-label" title="${escapeHtml(topicLabel)}">
          ${escapeHtml(topicLabel)}
        </span>
      </div>
      <div class="ros-graph__column ros-graph__column--target">
        ${renderGraphNodeCard(edge.target_node, 'target')}
      </div>
    </div>
  `;
}

function renderGraphPublisherGroup(group) {
  return `
    <div class="ros-graph__row ros-graph__publisher-group">
      <div class="ros-graph__column ros-graph__column--source">
        ${renderGraphNodeCard(group.source_node, 'source')}
      </div>
      <div class="ros-graph__publisher-topic-list">
        ${group.edges.map(renderGraphPublisherTopicRow).join('')}
      </div>
    </div>
  `;
}

function buildGraphRenderItems(rows) {
  const items = [];
  let currentPublisherGroup = null;

  rows.forEach((edge) => {
    if (edge.direction === 'publishes') {
      if (!currentPublisherGroup || currentPublisherGroup.source !== edge.source) {
        currentPublisherGroup = {
          kind: 'publisherGroup',
          source: edge.source,
          source_node: edge.source_node,
          edges: [],
        };
        items.push(currentPublisherGroup);
      }

      currentPublisherGroup.edges.push(edge);
      return;
    }

    currentPublisherGroup = null;
    items.push({ kind: 'edge', edge });
  });

  return items;
}

function renderGraph(payload) {
  if (!elements.graphCard || !elements.graphCanvas) {
    return;
  }

  const nodes = Array.isArray(payload.nodes) ? payload.nodes : [];
  const edges = Array.isArray(payload.edges) ? payload.edges : [];
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const rows = edges.map((edge) => ({
    ...edge,
    source_node: nodeById.get(edge.source),
    target_node: nodeById.get(edge.target),
    topic_node: nodeById.get(`topic:${edge.topic}`),
  }));
  const graphState = rows.length > 0 ? 'live' : 'waiting';
  const renderItems = buildGraphRenderItems(rows);
  const rowMarkup = renderItems.map((item) => (
    item.kind === 'publisherGroup'
      ? renderGraphPublisherGroup(item)
      : renderGraphEdge(item.edge, item.edge.topic_node)
  )).join('');

  setCardState(elements.graphCard, graphState);
  elements.graphChip.textContent = getGraphLabel(graphState);
  elements.graphUpdated.textContent = formatUpdatedAt(payload.updated_at);
  elements.graphSummary.textContent = `${nodes.length} nodes / ${edges.length} edges`;
  elements.graphCanvas.innerHTML = rowMarkup || '<p class="ros-graph__placeholder">Waiting for ROS graph</p>';
}

function setCardState(cardElement, state) {
  cardElement.classList.remove(...CARD_CLASSES);
  cardElement.classList.add(`is-${state}`);
}

function getBatteryState(data) {
  if (!data.has_data) {
    return 'waiting';
  }

  if (data.is_stale) {
    return 'stale';
  }

  const battery = clampPercent(data.battery_status);

  if (battery >= BATTERY_THRESHOLDS.good) {
    return 'good';
  }

  if (battery >= BATTERY_THRESHOLDS.caution) {
    return 'caution';
  }

  return 'low';
}

function getBatteryLabel(state) {
  switch (state) {
    case 'good':
      return 'GOOD';
    case 'caution':
      return 'CAUTION';
    case 'low':
      return 'LOW';
    case 'stale':
      return 'STALE';
    default:
      return 'WAITING';
  }
}

function getImageState(data) {
  if (!data.has_data) {
    return 'waiting';
  }

  if (data.is_stale) {
    return 'stale';
  }

  return 'live';
}

function getImageLabel(state) {
  switch (state) {
    case 'live':
      return 'LIVE';
    case 'stale':
      return 'STALE';
    default:
      return 'WAITING';
  }
}

function getControlState(data) {
  if (!data.has_data) {
    return 'waiting';
  }

  if (data.is_stale) {
    return 'stale';
  }

  return 'live';
}

function getControlLabel(state) {
  switch (state) {
    case 'live':
      return 'LIVE';
    case 'stale':
      return 'STALE';
    default:
      return 'WAITING';
  }
}

function getStorageState(data) {
  if (!data.has_data) {
    return 'waiting';
  }

  if (data.is_stale) {
    return 'stale';
  }

  const storageUsed = clampPercent(data.used_percentage);

  if (storageUsed >= STORAGE_THRESHOLDS.caution) {
    return 'low';
  }

  if (storageUsed >= STORAGE_THRESHOLDS.good) {
    return 'caution';
  }

  return 'good';
}

function getStorageLabel(state) {
  switch (state) {
    case 'good':
      return 'GOOD';
    case 'caution':
      return 'CAUTION';
    case 'low':
      return 'HIGH';
    case 'stale':
      return 'STALE';
    default:
      return 'WAITING';
  }
}

function formatControlValue(value, hasData) {
  if (!hasData || value === null || value === undefined) {
    return '--.--';
  }

  return Number(value).toFixed(2);
}

function setControlBar(barElement, fillElement, value, hasData) {
  barElement.classList.remove('is-positive', 'is-negative', 'is-neutral');

  if (!hasData || value === null || value === undefined) {
    barElement.classList.add('is-neutral');
    fillElement.style.left = '50%';
    fillElement.style.width = '0%';
    return;
  }

  const clampedValue = clampControl(value);
  const magnitude = Math.abs(clampedValue) * 50;

  if (magnitude < 0.5) {
    barElement.classList.add('is-neutral');
    fillElement.style.left = '50%';
    fillElement.style.width = '0%';
    return;
  }

  fillElement.style.left = clampedValue >= 0 ? '50%' : `${50 - magnitude}%`;
  fillElement.style.width = `${magnitude}%`;
  barElement.classList.add(clampedValue >= 0 ? 'is-positive' : 'is-negative');
}

function renderBattery(data) {
  const state = getBatteryState(data);
  const batteryPercent = data.has_data ? clampPercent(data.battery_status) : 0;

  setCardState(elements.batteryCard, state);
  elements.batteryChip.textContent = getBatteryLabel(state);
  elements.batteryValue.textContent = data.battery_display || '--.-%';
  elements.batteryUpdated.textContent = formatUpdatedAt(data.updated_at);
  elements.batteryMeterFill.style.width = `${batteryPercent}%`;
}

function renderImage(data) {
  const state = getImageState(data);

  setCardState(elements.imageCard, state);
  elements.imageChip.textContent = getImageLabel(state);
  elements.imageUpdated.textContent = formatUpdatedAt(data.updated_at);
  elements.imageResolution.textContent = data.resolution_display || 'Waiting for frame';
}

function renderControl(data) {
  const state = getControlState(data);
  const hasData = Boolean(data.has_data);

  setCardState(elements.controlCard, state);
  elements.controlChip.textContent = getControlLabel(state);
  elements.controlUpdated.textContent = formatUpdatedAt(data.updated_at);
  elements.throttleValue.textContent = formatControlValue(data.throttle, hasData);
  elements.steeringValue.textContent = formatControlValue(data.steering, hasData);

  setControlBar(elements.throttleBar, elements.throttleBarFill, data.throttle, hasData);
  setControlBar(elements.steeringBar, elements.steeringBarFill, data.steering, hasData);
}

function renderRecording(data) {
  const isRecording = Boolean(data.is_recording);

  elements.recordBadge.classList.toggle('is-recording', isRecording);
  elements.recordBadgeLabel.textContent = isRecording ? 'REC ON' : 'REC OFF';
}

function getLaneState(data) {
  if (!data.has_data) {
    return 'waiting';
  }

  if (data.is_stale) {
    return 'stale';
  }

  return 'live';
}

function renderLane(data) {
  if (!elements.laneCard) {
    return;
  }

  const state = getLaneState(data);
  const hasData = Boolean(data.has_data);

  setCardState(elements.laneCard, state);
  elements.laneChip.textContent = state === 'live' ? 'LIVE' : state === 'stale' ? 'STALE' : 'WAITING';
  elements.laneUpdated.textContent = formatUpdatedAt(data.updated_at);

  const offset = hasData && data.line_offset !== null ? data.line_offset : null;
  const normalizedOffset = offset !== null ? Math.max(-1, Math.min(1, offset / LANE_OFFSET_MAX_PX)) : null;
  elements.laneOffsetValue.textContent = offset !== null ? Number(offset).toFixed(1) + ' px' : '--.--- px';
  setControlBar(elements.laneOffsetBar, elements.laneOffsetBarFill, normalizedOffset, hasData && offset !== null);

  if (elements.yellowDetectedChip) {
    const detected = data.yellow_detected;
    if (!hasData || detected === null || detected === undefined) {
      elements.yellowDetectedChip.textContent = '--';
      elements.yellowDetectedChip.className = 'status-chip';
    } else if (detected) {
      elements.yellowDetectedChip.textContent = 'ON';
      elements.yellowDetectedChip.className = 'status-chip is-good';
    } else {
      elements.yellowDetectedChip.textContent = 'OFF';
      elements.yellowDetectedChip.className = 'status-chip is-stale';
    }
  }
}

function renderStorage(data) {
  const state = getStorageState(data);
  const usedPercent = data.has_data ? clampPercent(data.used_percentage) : 0;

  setCardState(elements.storageCard, state);
  elements.storageChip.textContent = getStorageLabel(state);
  elements.storageValue.textContent = data.used_display || '--.-%';
  elements.storageUpdated.textContent = formatUpdatedAt(data.updated_at);
  elements.storageMeterFill.style.width = `${usedPercent}%`;
  elements.storageDetail.textContent = `${data.used_space_display || '--'} / ${data.total_space_display || '--'} used`;
}

function renderOffline() {
  setCardState(elements.batteryCard, 'offline');
  setCardState(elements.imageCard, 'offline');
  setCardState(elements.controlCard, 'offline');
  setCardState(elements.storageCard, 'offline');
  if (elements.graphCard) {
    setCardState(elements.graphCard, 'offline');
  }
  if (elements.laneCard) {
    setCardState(elements.laneCard, 'offline');
    elements.laneChip.textContent = 'OFFLINE';
    elements.laneUpdated.textContent = 'Unable to reach monitor server';
  }
  elements.batteryChip.textContent = 'OFFLINE';
  elements.imageChip.textContent = 'OFFLINE';
  elements.controlChip.textContent = 'OFFLINE';
  elements.storageChip.textContent = 'OFFLINE';
  if (elements.graphChip) {
    elements.graphChip.textContent = 'OFFLINE';
  }
  renderRecording({ is_recording: false });
  elements.batteryUpdated.textContent = 'Unable to reach monitor server';
  elements.imageUpdated.textContent = 'Unable to reach monitor server';
  elements.controlUpdated.textContent = 'Unable to reach monitor server';
  elements.storageUpdated.textContent = 'Unable to reach monitor server';
  if (elements.graphUpdated) {
    elements.graphUpdated.textContent = 'Unable to reach monitor server';
  }
}

async function fetchGraph() {
  if (!config.graphEndpoint) {
    return;
  }

  try {
    const response = await fetch(config.graphEndpoint, { cache: 'no-store' });

    if (!response.ok) {
      throw new Error(`Unexpected response: ${response.status}`);
    }

    const payload = await response.json();
    renderGraph(payload || {});
  } catch (error) {
    console.error('Failed to fetch ROS graph', error);
    if (elements.graphCard) {
      setCardState(elements.graphCard, 'stale');
      elements.graphChip.textContent = 'STALE';
      elements.graphUpdated.textContent = 'Unable to reach graph endpoint';
    }
  }
}

async function fetchStatus() {
  try {
    const response = await fetch(config.statusEndpoint, { cache: 'no-store' });

    if (!response.ok) {
      throw new Error(`Unexpected response: ${response.status}`);
    }

    const payload = await response.json();
    renderBattery(payload.battery || {});
    renderImage(payload.image || {});
    renderControl(payload.control || {});
    renderRecording(payload.recording || {});
    renderStorage(payload.storage || {});
    renderLane(payload.lane || {});
  } catch (error) {
    console.error('Failed to fetch monitor status', error);
    renderOffline();
  }
}

function refreshCameraFrame() {
  if (imageRequestInFlight) {
    return;
  }

  imageRequestInFlight = true;

  const image = new Image();
  image.onload = () => {
    elements.cameraFrame.src = image.src;
    imageRequestInFlight = false;
  };
  image.onerror = () => {
    elements.cameraFrame.src = config.placeholderUrl;
    imageRequestInFlight = false;
  };
  image.src = `${config.frameEndpoint}?t=${Date.now()}`;
}

function refreshImageByEndpoint(targetElement, endpoint) {
  if (!targetElement) {
    return;
  }

  const image = new Image();
  image.onload = () => {
    targetElement.src = image.src;
  };
  image.onerror = () => {
    targetElement.src = config.placeholderUrl;
  };
  image.src = `${endpoint}?t=${Date.now()}`;
}

function refreshDebugFrames() {
  if (!config.debugImageEnabled || debugImageRequestInFlight) {
    return;
  }

  debugImageRequestInFlight = true;
  refreshImageByEndpoint(elements.debugFrameGrayscale, config.debugFrameGrayscaleEndpoint);
  refreshImageByEndpoint(elements.debugFrameBlur, config.debugFrameBlurEndpoint);
  refreshImageByEndpoint(elements.debugFrameEdge, config.debugFrameEdgeEndpoint);
  debugImageRequestInFlight = false;
}

function refreshLineFrames() {
  if (lineImageRequestInFlight) {
    return;
  }

  lineImageRequestInFlight = true;
  let pending = 0;

  function done() {
    pending -= 1;
    if (pending <= 0) lineImageRequestInFlight = false;
  }

  if (config.lineFrameBevEndpoint && elements.lineFrameBev) {
    pending += 1;
    const img = new Image();
    img.onload = () => { elements.lineFrameBev.src = img.src; done(); };
    img.onerror = () => { done(); };
    img.src = `${config.lineFrameBevEndpoint}?t=${Date.now()}`;
  }
  if (config.lineFrameSlidingEndpoint && elements.lineFrameSliding) {
    pending += 1;
    const img = new Image();
    img.onload = () => { elements.lineFrameSliding.src = img.src; done(); };
    img.onerror = () => { done(); };
    img.src = `${config.lineFrameSlidingEndpoint}?t=${Date.now()}`;
  }

  if (pending === 0) lineImageRequestInFlight = false;
}

function startPolling() {
  fetchStatus();
  fetchGraph();
  refreshCameraFrame();
  refreshLineFrames();
  if (config.debugImageEnabled) {
    refreshDebugFrames();
  }
  window.setInterval(fetchStatus, config.refreshIntervalMs);
  window.setInterval(fetchGraph, config.refreshIntervalMs);
  window.setInterval(refreshCameraFrame, config.imageRefreshIntervalMs);
  window.setInterval(refreshLineFrames, config.lineImageRefreshIntervalMs || 300);
  if (config.debugImageEnabled) {
    window.setInterval(refreshDebugFrames, config.imageRefreshIntervalMs);
  }
}

const HSV_PARAMS = [
  'white_h_min', 'white_h_max', 'white_s_min', 'white_s_max', 'white_v_min', 'white_v_max',
  'yellow_h_min', 'yellow_h_max', 'yellow_s_min', 'yellow_s_max', 'yellow_v_min', 'yellow_v_max',
];

let hsvDebounceTimers = {};

async function loadLineParams() {
  const statusEl = document.getElementById('hsv-tuner-status');
  try {
    const res = await fetch('/api/line/params', { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const params = await res.json();
    HSV_PARAMS.forEach((name) => {
      const val = params[name];
      if (val === null || val === undefined) return;
      const slider = document.getElementById('sl-' + name);
      const label = document.getElementById('val-' + name);
      if (slider) slider.value = val;
      if (label) label.textContent = val;
    });
    if (statusEl) statusEl.textContent = '';
  } catch {
    if (statusEl) statusEl.textContent = '(line_node 미실행)';
  }
}

async function sendLineParam(name, value) {
  const statusEl = document.getElementById('hsv-tuner-status');
  try {
    await fetch('/api/line/param', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, value }),
    });
    if (statusEl) statusEl.textContent = '';
  } catch {
    if (statusEl) statusEl.textContent = '전송 실패';
  }
}

function initHsvSliders() {
  HSV_PARAMS.forEach((name) => {
    const slider = document.getElementById('sl-' + name);
    if (!slider) return;
    slider.addEventListener('input', () => {
      const val = Number(slider.value);
      const label = document.getElementById('val-' + name);
      if (label) label.textContent = val;

      clearTimeout(hsvDebounceTimers[name]);
      hsvDebounceTimers[name] = setTimeout(() => sendLineParam(name, val), 150);
    });
  });

  const tuner = document.getElementById('hsv-tuner');
  if (tuner) {
    tuner.addEventListener('toggle', () => {
      if (tuner.open) loadLineParams();
    });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  startPolling();
  initHsvSliders();
});
