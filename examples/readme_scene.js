// IndoorLoc's five layers as a 3D stack. One real UJIIndoorLoc scan travels up from the data layer to
// evaluation; the application layer replays a held-out ILC 2020 phone trace. Every number drawn is recorded.
const $ = id => document.getElementById(id);
const stage = $('stage'), holder = $('scene'), tagLayer = $('tags');
const query = new URLSearchParams(location.search);
const theme = query.get('theme') === 'dark' ? 'dark' : 'light';  // White unless ?theme=dark.
document.documentElement.dataset.theme = theme;
// ?lang=zh or ?lang=en; otherwise the browser language. Exports always pass it explicitly.
const zh = query.has('lang') ? query.get('lang') === 'zh' : navigator.language.toLowerCase().startsWith('zh');
const tr = (en, cn) => zh ? cn : en;
document.documentElement.lang = zh ? 'zh-CN' : 'en';
await Promise.all(['500 16px Inter', '600 16px Inter', '400 13px "JetBrains Mono"', '500 16px "Noto Sans SC"', '600 16px "Noto Sans SC"']
  .map(font => document.fonts.load(font, zh ? '数据' : 'A')));
const dark = theme === 'dark';
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const color = name => new THREE.Color(css(name));
const COLOR = {reference: color('--reference'), match: color('--match'), prediction: color('--prediction'),
  truth: color('--truth'), accent: color('--accent'), shadow: new THREE.Color(dark ? '#000000' : '#8ea3b0')};
const clamp01 = x => Math.min(1, Math.max(0, x));
const smooth = x => (x = clamp01(x), x * x * (3 - 2 * x));
const ease = x => (x = clamp01(x), x < .5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
const ramp = (t, a, b) => ease((t - a) / (b - a));
const lerp = (a, b, x) => a + (b - a) * x;
const within = (t, a, b, fadeIn = .4, fadeOut = .4) => Math.min(smooth((t - a) / fadeIn), smooth((b - t) / fadeOut));
const number = (x, digits = 0) => x.toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits});

// The five layers of the architecture figure (assets/architecture.png), bottom to top.
const LAYERS = [
  {id: 'L1', name: tr('Data', '数据'), exit: tr('→ numpy / torch', '→ 导出 numpy / torch'), color: '--l1'},
  {id: 'L2', name: tr('Signals', '信号'), exit: tr('→ process one signal', '→ 处理单条信号'), color: '--l2'},
  {id: 'L3', name: tr('Methods', '方法'), exit: tr('→ train on your data', '→ 训练自有数据'), color: '--l3'},
  {id: 'L4', name: tr('Evaluation', '评测'), exit: tr('→ score your predictions', '→ 评测自有预测'), color: '--l4'},
  {id: 'L5', name: tr('Applications', '应用'), exit: tr('→ track a live stream', '→ 跟踪实时数据流'), color: '--l5'},
];
LAYERS.forEach(layer => layer.rgb = color(layer.color));
const W = 300, D = 180, GAP = 64;  // Plate size and spacing, in scene units.
const level = i => i * GAP;

// Timeline, in seconds: a sweep, one act per layer, then the whole stack. The last frame matches the first.
const T = {l1: 2.4, l2: 5.4, l3: 8.3, l4: 11.8, l5: 15.2, outro: 19.2, end: DATA.duration};
const ACT_KEYS = ['l1', 'l2', 'l3', 'l4', 'l5'];
const actAt = t => t < T.l1 ? 'intro' : t < T.l2 ? 'l1' : t < T.l3 ? 'l2' : t < T.l4 ? 'l3' : t < T.l5 ? 'l4' : t < T.outro ? 'l5' : 'outro';
const layerOf = act => ACT_KEYS.indexOf(act);
const CLEAR = T.end - 1.2;  // Layer contents fade out, leaving the bare stack the loop starts from.

const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true, preserveDrawingBuffer: true});
renderer.setClearColor(0x000000, 0);
renderer.outputColorSpace = THREE.SRGBColorSpace;
holder.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 16 / 9, 4, 6000);
scene.add(new THREE.HemisphereLight(0xffffff, dark ? 0x1b2a38 : 0xb9c5cd, dark ? 1.3 : 2.1));
const sun = new THREE.DirectionalLight(0xffffff, dark ? 1.3 : 1.5);
sun.position.set(160, 420, 260);
scene.add(sun);
const blending = dark ? THREE.AdditiveBlending : THREE.NormalBlending;

// Rounded-rectangle panels drawn from an analytic signed distance: crisp edges at any zoom, optional dashes.
const panelShader = {
  vertexShader: 'varying vec2 vLocal; void main(){ vLocal = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
  fragmentShader: `uniform vec2 halfSize; uniform float radius, fillAlpha, edgeAlpha, glowAlpha, soft, dash, opacity, width;
uniform vec3 fillColor, edgeColor; varying vec2 vLocal;
float box(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
void main() {
  float d = box(vLocal, halfSize, radius);
  float aa = max(fwidth(d), 1e-4);
  float inside = 1.0 - smoothstep(-max(aa, soft), max(aa, soft), d);
  float w = max(width, aa * .9);
  float edge = 1.0 - smoothstep(w - aa, w + aa, abs(d));
  if (dash > 0.0 && fract((abs(vLocal.x) + abs(vLocal.y)) / dash) > .55) edge = 0.0;
  float glow = inside * exp(d / 7.0);
  float alpha = (fillAlpha * inside + glowAlpha * glow + edgeAlpha * edge) * opacity;
  if (alpha < .002) discard;
  gl_FragColor = vec4(mix(fillColor, edgeColor, clamp(edge + glow * .4, 0.0, 1.0)), alpha);
  #include <colorspace_fragment>
}`,
};
function panel(w, d, fill, edge, {radius = 10, fillAlpha = .07, edgeAlpha = .85, glowAlpha = .06, soft = 0, dash = 0, width = .45} = {}) {
  const material = new THREE.ShaderMaterial({...panelShader, transparent: true, depthWrite: false, blending, side: THREE.DoubleSide,
    uniforms: {halfSize: {value: new THREE.Vector2(w / 2, d / 2)}, radius: {value: radius}, fillAlpha: {value: fillAlpha},
      edgeAlpha: {value: edgeAlpha}, glowAlpha: {value: glowAlpha}, soft: {value: soft}, dash: {value: dash}, width: {value: width},
      opacity: {value: 1}, fillColor: {value: fill}, edgeColor: {value: edge}}});
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w + 2 * soft + 4, d + 2 * soft + 4), material);
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}

// Round sprites sized in scene units, capped per unit on screen.
const pointShader = {
  vertexShader: `attribute float aSize, aAlpha, aHalo; attribute vec3 aColor; uniform float scale, perUnit;
varying float vAlpha, vHalo, vPixels; varying vec3 vColor;
void main() {
  vec4 view = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * view;
  vPixels = max(min(aSize * scale / -view.z, aSize * perUnit), 1.6);
  gl_PointSize = vPixels * (1.0 + 2.0 * aHalo);
  vAlpha = aAlpha; vHalo = aHalo; vColor = aColor;
}`,
  fragmentShader: `varying float vAlpha, vHalo, vPixels; varying vec3 vColor;
void main() {
  float r = length(gl_PointCoord * 2.0 - 1.0);
  float core = 1.0 / (1.0 + 2.0 * vHalo);
  float aa = 2.0 / (vPixels * (1.0 + 2.0 * vHalo));
  float disc = 1.0 - smoothstep(core - aa, core + aa, r);
  float halo = vHalo * (1.0 - smoothstep(core, 1.0, r)) * .32;
  float alpha = vAlpha * max(disc, halo);
  if (alpha < .003) discard;
  gl_FragColor = vec4(vColor, alpha);
  #include <colorspace_fragment>
}`,
};
const clouds = [];
function pointCloud(count, parent) {
  const geometry = new THREE.BufferGeometry();
  for (const [name, size] of [['position', 3], ['aColor', 3], ['aSize', 1], ['aAlpha', 1], ['aHalo', 1]])
    geometry.setAttribute(name, new THREE.BufferAttribute(new Float32Array(count * size), size));
  const points = new THREE.Points(geometry, new THREE.ShaderMaterial({...pointShader, transparent: true, depthWrite: false, blending,
    uniforms: {scale: {value: 1}, perUnit: {value: 12}}}));
  points.frustumCulled = false;
  points.renderOrder = 30;
  parent.add(points);
  clouds.push(points);
  return points;
}

// Rods: unit cylinders from y = 0 to 1, shaded along their length; optional dashes.
const rodGeometry = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true).translate(0, .5, 0);
const rodMaterial = (a, b) => new THREE.ShaderMaterial({transparent: true, depthWrite: false,
  uniforms: {colorA: {value: a}, colorB: {value: b}, opacity: {value: 1}, dash: {value: 0}, length: {value: 1}},
  vertexShader: 'varying float vT; void main(){ vT = position.y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
  fragmentShader: `uniform vec3 colorA, colorB; uniform float opacity, dash, length; varying float vT;
void main() {
  if (dash > 0.0 && fract(vT * length / dash) > .56) discard;
  if (opacity < .003) discard;
  gl_FragColor = vec4(mix(colorA, colorB, vT), opacity);
  #include <colorspace_fragment>
}`});
const UP = new THREE.Vector3(0, 1, 0), direction = new THREE.Vector3(), turn = new THREE.Quaternion(), stretch = new THREE.Vector3();
function setRod(mesh, a, b, radius, progress, opacity) {
  direction.subVectors(b, a);
  const length = Math.max(direction.length(), 1e-6);
  turn.setFromUnitVectors(UP, direction.divideScalar(length));
  mesh.position.copy(a); mesh.quaternion.copy(turn); mesh.scale.copy(stretch.set(radius, Math.max(length * progress, 1e-4), radius));
  mesh.material.uniforms.opacity.value = opacity;
  mesh.material.uniforms.length.value = length * progress;
  mesh.visible = opacity > .003 && progress > .001;
}
function rod(a, b, parent) {
  const mesh = new THREE.Mesh(rodGeometry, rodMaterial(a, b));
  mesh.renderOrder = 26;
  parent.add(mesh);
  return mesh;
}
const basic = (c, opacity = 1) => new THREE.MeshBasicMaterial({color: c, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide});

// DOM tags anchored to 3D points.
function tag(className, html = '', style = '') {
  const node = document.createElement('div');
  node.className = `tag ${className}`;
  node.innerHTML = html;
  if (style) node.style.cssText = style;
  tagLayer.append(node);
  return node;
}
const projected = new THREE.Vector3(), worldPoint = new THREE.Vector3();
function place(node, object, point, visible) {
  worldPoint.copy(point);
  if (object) object.localToWorld(worldPoint);
  projected.copy(worldPoint).project(camera);
  const onScreen = projected.z < 1 && Math.abs(projected.x) < 1.05 && Math.abs(projected.y) < 1.05;
  node.style.opacity = onScreen ? visible.toFixed(3) : 0;
  node.style.visibility = onScreen && visible > .003 ? 'visible' : 'hidden';
  node.style.left = `${((projected.x * .5 + .5) * 1280).toFixed(1)}px`;
  node.style.top = `${((-projected.y * .5 + .5) * 720).toFixed(1)}px`;
}

// The stack: one plate per layer, each with a soft shadow, a side label and its standalone exit.
const plates = LAYERS.map((layer, i) => {
  const group = new THREE.Group();
  group.position.y = level(i);
  scene.add(group);
  const shadow = panel(W, D, COLOR.shadow, COLOR.shadow, {fillAlpha: dark ? 0 : .05, edgeAlpha: 0, glowAlpha: 0, soft: 14});
  shadow.position.y = -2.5;
  shadow.renderOrder = 1;
  const top = panel(W, D, layer.rgb, layer.rgb, layer.planned ? {fillAlpha: .025, edgeAlpha: .75, glowAlpha: .02, dash: 9}
    : {fillAlpha: dark ? .05 : .075, edgeAlpha: .9, glowAlpha: .07});
  top.renderOrder = 2 + i;
  group.add(shadow, top);
  const content = new THREE.Group();
  content.position.y = 1;
  group.add(content);
  const label = tag('layer', `<small style="--c:var(${layer.color})">${layer.id}</small>${layer.name}${layer.planned ? ` <em>${tr('planned', '规划中')}</em>` : ''}`);
  const exit = layer.exit ? tag('exit', layer.exit, `--c:var(${layer.color})`) : null;
  return {layer, group, top, shadow, content, label, exit};
});
const layerRGB = i => plates[i].layer.rgb;

// L1 · Data: the loaders by signal type, and the replayed dataset's radio map in miniature.
const reference = DATA.reference, refCount = reference.xy.length;
const TYPE = {wifi: 'WiFi', ble: 'BLE', csi: 'CSI', multi: tr('IMU + WiFi', 'IMU + WiFi'), simulated: tr('Simulated', '仿真')};
const tiles = [];
Object.entries(DATA.catalog).forEach(([, ids], column) => ids.forEach((id, row) => {
  const verified = DATA.verified.includes(id);
  const mesh = panel(22, 22, layerRGB(0), layerRGB(0), {radius: 4, fillAlpha: verified ? .55 : .14, edgeAlpha: verified ? 1 : .6, glowAlpha: 0, width: .35});
  mesh.position.set(-128 + column * 28, 0, -66 + row * 26);
  plates[0].content.add(mesh);
  tiles.push({mesh, id, verified, row, column});
}));
const typeTags = [{node: tag('chip', Object.entries(DATA.catalog).map(([signal, ids]) => `${TYPE[signal] || signal} ×${ids.length}`).join(' · ')),
  at: new THREE.Vector3(-100, 0, 96)}];
const verifiedTile = tiles.find(tile => tile.verified);
const verifiedTag = {node: tag('chip on', `${DATA.dataset} ✓`), at: new THREE.Vector3(-128, 0, -66)};
const campus = new THREE.Group();
const MINI_FLOOR = 9;
campus.position.set(52, 0, -12);
campus.scale.setScalar(.42);
plates[0].content.add(campus);
const campusPoints = pointCloud(refCount, campus);
reference.xy.forEach((xy, i) => campusPoints.geometry.attributes.position.array.set([xy[0], reference.floor[i] * MINI_FLOOR + 1, -xy[1]], i * 3));

// L2 · Signals: the scan's full 520-value fingerprint as it enters the model.
const entry = DATA.featured[0];
const scanIndex = entry.index;
const count = entry.features.length;
const bars = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1).translate(0, .5, 0),
  new THREE.MeshStandardMaterial({roughness: .6, transparent: true}), count);
bars.frustumCulled = false;
plates[1].content.add(bars);
const barX = i => -136 + i * (272 / (count - 1));
const heard = entry.raw.map(v => v !== null);
const barColor = new THREE.Color(), strong = new THREE.Color(dark ? '#e8fffb' : '#063f39');
entry.features.forEach((v, i) => bars.setColorAt(i, heard[i] ? barColor.copy(layerRGB(1)).lerp(strong, clamp01(v * 1.2))
  : barColor.set(dark ? '#2b3a46' : '#d3dde2')));
const BAR_HEIGHT = 72;
const matrix = new THREE.Matrix4(), position = new THREE.Vector3(), rotation = new THREE.Quaternion(), size = new THREE.Vector3();
const axisTags2 = [[0, 'WAP001'], [count - 1, `WAP${count}`]].map(([i, text]) => ({node: tag('axis', text), at: new THREE.Vector3(barX(i), 0, 22)}));
const strongest = entry.raw.map((v, i) => [v, i]).filter(([v]) => v !== null).sort((a, b) => b[0] - a[0])[0];
const peakTag = {node: tag('chip on', `WAP${String(strongest[1] + 1).padStart(3, '0')} · ${strongest[0]} dBm`),
  at: new THREE.Vector3(barX(strongest[1]), Math.max(1.2, entry.features[strongest[1]] * BAR_HEIGHT) + 8, 0)};

// L3 · Methods: WKNN on the true floor around the scan, with its 5 nearest training fingerprints.
const scans = DATA.scans, S = scanIndex;
const truth = {xy: scans.truth[S], floor: scans.floor[S], building: scans.building[S]};
const result = {xy: scans.results.wknn.prediction[S], floor: scans.results.wknn.floor[S], error: scans.results.wknn.error[S]};
const refs = [...new Set(entry.neighbors.map(n => n.reference))];
const center = [0, 1].map(k => (truth.xy[k] + result.xy[k] + refs.reduce((sum, r) => sum + reference.xy[r][k], 0)) / (2 + refs.length));
const MAP_SCALE = 2.7, MAP_X = 42, HALF = [(W / 2 - 78) / MAP_SCALE, (D / 2 - 12) / MAP_SCALE];
const local = xy => new THREE.Vector3(MAP_X + (xy[0] - center[0]) * MAP_SCALE, 0, -(xy[1] - center[1]) * MAP_SCALE);
// The same fit / predict / evaluate interface covers these localizers; this scan uses WKNN.
const models = ['kNN', 'WKNN', 'SVM', 'RF', 'Ensemble'].map((name, j) => {
  const on = name === 'WKNN';
  const mesh = panel(52, 20, layerRGB(2), layerRGB(2), {radius: 5, fillAlpha: on ? .5 : .1, edgeAlpha: on ? 1 : .55, glowAlpha: 0, width: .35});
  mesh.position.set(-112, 0, -62 + j * 31);
  plates[2].content.add(mesh);
  return {mesh, on, node: tag(on ? 'chip on' : 'chip', name), at: new THREE.Vector3(-112, 0, -62 + j * 31)};
});
const nearby = reference.xy.map((_, r) => r).filter(r => reference.building[r] === truth.building && reference.floor[r] === truth.floor
  && Math.abs(reference.xy[r][0] - center[0]) < HALF[0] && Math.abs(reference.xy[r][1] - center[1]) < HALF[1]);
const mapPoints = pointCloud(nearby.length, plates[2].content);
nearby.forEach((r, j) => local(reference.xy[r]).setY(.8).toArray(mapPoints.geometry.attributes.position.array, j * 3));
const neighborPoints = pointCloud(refs.length, plates[2].content);
refs.forEach((r, j) => local(reference.xy[r]).setY(1.2).toArray(neighborPoints.geometry.attributes.position.array, j * 3));
function pin(c, parent) {
  const group = new THREE.Group();
  const head = new THREE.Mesh(new THREE.SphereGeometry(2.6, 32, 20), new THREE.MeshStandardMaterial({color: c, emissive: c,
    emissiveIntensity: dark ? .6 : .18, roughness: .4, transparent: true}));
  head.position.y = 14;
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(.35, .35, 14, 10).translate(0, 7, 0), basic(c, .9));
  const ring = new THREE.Mesh(new THREE.RingGeometry(2.6, 3.4, 48).rotateX(-Math.PI / 2), basic(c, .9));
  group.add(head, stem, ring);
  parent.add(group);
  return {group, parts: [head, stem, ring], base: [1, .9, .9]};
}
const truthPin = pin(COLOR.truth, plates[2].content), predictionPin = pin(COLOR.prediction, plates[2].content);
truthPin.group.position.copy(local(truth.xy));
predictionPin.group.position.copy(local(result.xy));
const links = entry.neighbors.map(() => rod(COLOR.match, COLOR.prediction, plates[2].content));
const errorRod = rod(COLOR.truth, COLOR.prediction, plates[2].content);
errorRod.material.uniforms.dash.value = 3;
const truthTag = tag('pin truth', tr(`<b>Truth</b> <i>floor ${truth.floor}</i>`, `<b>真值</b> <i>楼层 ${truth.floor}</i>`));
truthTag.style.transform = 'translate(calc(-100% - 12px),-50%)';
const predictionTag = tag('pin prediction', `<b>WKNN</b> <i>${tr('floor', '楼层')} ${result.floor}</i>`);
const errorTag = tag('distance', `${result.error.toFixed(2)} m`);
const nearestTag = tag('pin match', tr(`<b>${DATA.k} nearest</b> <i>of ${number(DATA.trainingFingerprints)}</i>`,
  `<b>最近 ${DATA.k} 条</b> <i>共 ${number(DATA.trainingFingerprints)} 条</i>`));
nearestTag.style.transform = 'translate(calc(-100% - 12px),-50%)';

// L4 · Evaluation: the recorded errors of all held-out scans, as a histogram under their cumulative share.
const errors = DATA.methods.wknn.errors, metrics = DATA.methods.wknn.metrics;
const LOW = .05, HIGH = Math.max(...errors) * 1.02, BINS = 42;
const logPosition = e => (Math.log(Math.max(e, LOW)) - Math.log(LOW)) / (Math.log(HIGH) - Math.log(LOW));
const logX = e => -136 + 272 * logPosition(e);
const bins = new Array(BINS).fill(0);
errors.forEach(e => bins[Math.min(BINS - 1, Math.floor(logPosition(e) * BINS))]++);
const bandColor = e => e <= 5 ? COLOR.accent : e <= 20 ? COLOR.match : COLOR.prediction;
const histogram = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1).translate(0, .5, 0),
  new THREE.MeshStandardMaterial({roughness: .55, transparent: true}), BINS);
histogram.frustumCulled = false;
plates[3].content.add(histogram);
const binCenter = b => Math.exp(Math.log(LOW) + (b + .5) / BINS * (Math.log(HIGH) - Math.log(LOW)));
bins.forEach((_, b) => histogram.setColorAt(b, bandColor(binCenter(b))));
const maxBin = Math.max(...bins);
const cdfPoints = errors.map((e, i) => new THREE.Vector3(logX(e), 2 + (i + 1) / errors.length * 40, -44));
const CDF_STEPS = 90;
const cdfRods = Array.from({length: CDF_STEPS}, () => rod(layerRGB(3), layerRGB(3), plates[3].content));
const cdfAt = f => cdfPoints[Math.min(errors.length - 1, Math.floor(f * (errors.length - 1)))];
const axisTags4 = [.1, 1, 5, 20, 200].filter(e => e < HIGH).map(e => ({node: tag('axis', `${e} m`), at: new THREE.Vector3(logX(e), 0, 36)}));

// L5 · Applications: the components, then one held-out ILC 2020 trace on its floor plan: the surveyor's
// waypoints, the WiFi fixes of L3 and the PDR + WiFi particle filter that keeps to the walls.
const APPS = DATA.apps;
const components = (zh ? ['卡尔曼 / RTS', '粒子滤波 + 地图', '航位推算', 'PDR + WiFi 融合', '流式推理', 'A* 导航']
  : ['Kalman / RTS', 'Particle filter', 'PDR', 'PDR + WiFi', 'Streaming', 'A* navigation']).map((name, j) => {
  const at = new THREE.Vector3(-117 + (j % 2) * 60, 0, -50 + Math.floor(j / 2) * 34);
  const mesh = panel(54, 24, layerRGB(4), layerRGB(4), {radius: 5, fillAlpha: .14, edgeAlpha: .85, glowAlpha: 0, width: .35});
  mesh.position.copy(at);
  plates[4].content.add(mesh);
  return {mesh, node: tag('chip', name), at};
});
const APP_SCALE = Math.min(150 / APPS.size[0], 150 / APPS.size[1]), APP_X = 66;
const appAt = (xy, y) => new THREE.Vector3(APP_X + xy[0] * APP_SCALE, y, -xy[1] * APP_SCALE);
const segments = (points, y, c) => points.slice(1).map((xy, j) => ({mesh: rod(c, c, plates[4].content), a: appAt(points[j], y), b: appAt(xy, y)}));
const wallColor = color('--faint');
const wallRods = APPS.walls.map(w => ({mesh: rod(wallColor, wallColor, plates[4].content), a: appAt([w[0], w[1]], .4), b: appAt([w[2], w[3]], .4)}));
const truthRods = segments(APPS.truth, 1.0, COLOR.truth);
const fusedRods = segments(APPS.fused, 1.8, layerRGB(4));
const fixPoints = pointCloud(APPS.fixes.length, plates[4].content);
APPS.fixes.forEach((xy, j) => appAt(xy, 1.4).toArray(fixPoints.geometry.attributes.position.array, j * 3));
// Labels at the extremes of each drawing, so they never cover one another: truth to the right,
// the fixes below, the fused track above.
const extreme = (points, score) => points.reduce((best, xy) => score(xy) > score(best) ? xy : best, points[0]);
const truthLabelAt = extreme(APPS.truth, xy => xy[0]), fixLabelAt = extreme(APPS.fixes, xy => -xy[1]);
const fusedLabelAt = extreme(APPS.fused, xy => xy[1]);
const truthTag5 = tag('pin truth', tr('<b>Truth</b> <i>waypoints</i>', '<b>真值</b> <i>路点</i>'));
const fixTag5 = tag('pin', `<b>WiFi WKNN</b> <i>${tr('fixes', '定位点')}</i>`);
const fusedTag5 = tag('pin prediction', `<b>PDR + WiFi</b> <i>${tr('+ floor plan', '+ 楼层地图')}</i>`);
fixTag5.style.transform = 'translate(-50%,12px)';
fusedTag5.style.transform = 'translate(-50%,calc(-100% - 10px))';

// Camera: the whole stack, then an elevator ride up through the layers, then the whole stack again.
const OPEN_UP = 115, OPEN_DOWN = 75;  // How far the stack opens above and below the active layer.
const full = () => ({target: new THREE.Vector3(0, 2 * GAP + 4, 0), distance: 1060, azimuth: .92, elevation: .4, shiftX: 175, shiftY: 14, cut: 2, open: 0});
const at = i => ({target: new THREE.Vector3(0, level(i) + 12, 0), distance: 610, azimuth: .96, elevation: .52, shiftX: 24, shiftY: -52, cut: i, open: 1});
function cameraKeys() {
  const keys = [[0, full()], [T.l1 - .4, full()]];
  ACT_KEYS.forEach((act, i) => keys.push([T[act] + .5, at(i)], [(i < 4 ? T[ACT_KEYS[i + 1]] : T.outro) - .1, at(i)]));
  keys.push([T.outro + 1.1, full()], [T.end, full()]);
  return keys;
}
function hermite(keys, t, get) {
  // Monotone cubic (Fritsch–Carlson): smooth through every key without overshooting it.
  let k = 0;
  while (k < keys.length - 2 && t > keys[k + 1][0]) k++;
  const [t0, a] = keys[k], [t1, b] = keys[k + 1];
  const secant = j => (get(keys[j + 1][1]) - get(keys[j][1])) / (keys[j + 1][0] - keys[j][0]);
  const slope = j => {
    if (j === 0 || j === keys.length - 1) return 0;
    const left = secant(j - 1), right = secant(j);
    return left * right <= 0 ? 0 : 2 / (1 / left + 1 / right);
  };
  const s = clamp01((t - t0) / (t1 - t0)), h = t1 - t0, s2 = s * s, s3 = s2 * s;
  return (2 * s3 - 3 * s2 + 1) * get(a) + (s3 - 2 * s2 + s) * h * slope(k) + (-2 * s3 + 3 * s2) * get(b) + (s3 - s2) * h * slope(k + 1);
}
let orbit = 0, zoom = 1;
const cameraTarget = new THREE.Vector3();
function placeCamera(t) {
  const keys = cameraKeys(), value = get => hermite(keys, t, get);
  cameraTarget.set(value(p => p.target.x), value(p => p.target.y), value(p => p.target.z));
  const distance = Math.exp(value(p => Math.log(p.distance))) * zoom;
  const azimuth = value(p => p.azimuth) + orbit, elevation = value(p => p.elevation);
  camera.position.set(cameraTarget.x + distance * Math.cos(elevation) * Math.cos(azimuth),
    cameraTarget.y + distance * Math.sin(elevation), cameraTarget.z + distance * Math.cos(elevation) * Math.sin(azimuth));
  camera.lookAt(cameraTarget);
  const shiftX = value(p => p.shiftX), shiftY = value(p => p.shiftY);  // Positive values move the subject right / down.
  const fullWidth = 1280 + 2 * Math.abs(shiftX), fullHeight = 720 + 2 * Math.abs(shiftY);
  camera.fov = 2 * Math.atan(Math.tan(15 * Math.PI / 180) * fullHeight / 720) * 180 / Math.PI;
  camera.aspect = fullWidth / fullHeight;
  camera.setViewOffset(fullWidth, fullHeight, Math.abs(shiftX) - shiftX, Math.abs(shiftY) - shiftY, 1280, 720);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
}

// Captions and the info card: recorded values only.
const floors = DATA.buildings.reduce((n, b) => n + b.floors.length, 0);
const heardCount = heard.filter(Boolean).length;
const loaders = Object.values(DATA.catalog).flat().length;
const row = (label, value, dot = '') => `<div><span>${dot ? `<i class="dot" style="background:var(${dot})"></i>` : ''}${label}</span><b>${value}</b></div>`;
const floorText = tr(result.floor === truth.floor ? `${result.floor} · correct` : `${result.floor} · truth ${truth.floor}`,
  result.floor === truth.floor ? `${result.floor} · 正确` : `${result.floor} · 真值 ${truth.floor}`);
const exits = `<div class="info-rows">${LAYERS.filter(l => l.exit).map(l => row(`${l.id} · ${l.name}`, l.exit.replace('→ ', ''), l.color)).join('')}</div>`;
const apps = APPS.methods, mean = name => `${apps[name].mean.toFixed(2)} m`;
const appsNote = tr(`${APPS.dataset} · ${APPS.testTraces} held-out traces, ${APPS.waypoints} waypoints · leave one trajectory out · the differences are within sampling error`,
  `${APPS.dataset} · ${APPS.testTraces} 条留出轨迹、${APPS.waypoints} 个路点 · 留一轨迹 · 差异在抽样误差之内`);
const measured = (DATA.catalog.wifi || []).length + (DATA.catalog.ble || []).length + (DATA.catalog.csi || []).length + (DATA.catalog.multi || []).length;
const TEXT = zh ? {
  intro: {kicker: 'IndoorLoc', headline: '无线室内定位的全栈工具库', sub: '从数据到应用共五层 · 每一层均可独立使用', card: ['五层架构', '', exits]},
  l1: {kicker: 'L1 <span>·</span> 数据', headline: `${loaders} 个数据集，一个加载接口`,
    sub: `${measured} 个实测数据集均带 SHA-256 校验 · 本片回放 ${DATA.dataset}`,
    card: ['L1 · 数据', `${loaders} 个加载器`, `<div class="info-rows">${Object.entries(DATA.catalog)
      .map(([signal, ids]) => row(TYPE[signal] || signal, `${ids.length} 个`)).join('')}
      ${row(DATA.dataset, `训练 ${number(DATA.trainingFingerprints)} · 测试 ${number(DATA.evaluationSamples)}`)}</div>
      <div class="info-note">${number(refCount)} 个测量位置 · ${DATA.buildings.length} 栋楼 · ${floors} 个楼层 · 缺失读数记为 NaN，坐标保持原始坐标系</div>`]},
  l2: {kicker: 'L2 <span>·</span> 信号', headline: `一次扫描成为 ${count} 维指纹`,
    sub: `test.X[${scanIndex}] 听到 ${count} 个 AP 中的 ${heardCount} 个`,
    card: ['L2 · 信号', `test.X[${scanIndex}]`, `<div class="info-rows">${row('听到的 AP', `${heardCount} / ${count}`)}
      ${row('最强信号', `${strongest[0]} dBm`)}${row('未听到', 'NaN → −104 dBm')}${row('变换', 'FillMissing(−104)')}</div>
      <div class="info-note">RSSI 表示 · 设备校准 · 增强 · CSI 相位净化 · 测距 · IMU · 地磁 · 可见光</div>`]},
  l3: {kicker: 'L3 <span>·</span> 方法', headline: '不同模型，同一接口',
    sub: `WKNN 在 ${number(DATA.trainingFingerprints)} 条指纹中找到最近的 ${DATA.k} 条`,
    card: ['L3 · 方法', `WKNN · k = ${DATA.k}`, `<div class="info-rows">${row('最近指纹', `${DATA.k} 条，来自 ${refs.length} 个位置`)}
      ${row('本次误差', `${result.error.toFixed(2)} m`)}${row('楼层', floorText)}</div>
      <div class="info-note">指纹（kNN · Horus · 高斯过程 · 森林 · 集成）· 模型驱动（三边 · Chan TDoA · MUSIC）· 深度 · 迁移，都用同样的 fit / localize / evaluate</div>`]},
  l4: {kicker: 'L4 <span>·</span> 评测', headline: '所有留出扫描，同一套协议',
    sub: `官方切分的 ${number(DATA.evaluationSamples)} 条扫描 · WKNN 运行记录`,
    card: ['L4 · 评测', `${number(DATA.evaluationSamples)} 条扫描`, `<div class="info-rows">${row('平均误差', `${metrics.mean_error.toFixed(2)} m`)}
      ${row('中位误差', `${metrics.median_error.toFixed(2)} m`)}${row('P90 误差', `${metrics.p90_error.toFixed(2)} m`)}
      ${row('楼层准确率', `${metrics.floor_accuracy.toFixed(2)} %`)}</div>
      <div class="info-note">柱：误差直方图（≤ 5 m · 5–20 m · &gt; 20 m）· 线：误差不超过该值的扫描占比</div>`]},
  l5: {kicker: 'L5 <span>·</span> 应用', headline: '真实手机轨迹上的跟踪与融合',
    sub: 'WiFi 定位点 → 行人航位推算 + 粒子滤波，粒子不穿墙',
    card: ['L5 · 应用', `${APPS.testTraces} 条留出轨迹`, `<div class="info-rows">${row('WiFi WKNN', mean('WiFi WKNN'))}
      ${row('+ 卡尔曼 RTS 平滑', mean('+ Kalman RTS'))}${row('PDR + WiFi + 地图', mean('PDR + WiFi + map'))}</div>
      <div class="info-note">${appsNote}</div>`]},
  outro: {kicker: 'IndoorLoc', headline: '五层架构，按需取用', sub: '画面中的每个数字都能用 <code>python3 -m examples.readme_demo</code> 重跑',
    card: ['单独使用某一层', '', exits]},
} : {
  intro: {kicker: 'IndoorLoc', headline: 'The full stack for wireless indoor localization.',
    sub: 'Five layers, from data to applications · each one usable on its own', card: ['Five layers', '', exits]},
  l1: {kicker: 'L1 <span>·</span> Data', headline: `${loaders} datasets, one loader.`,
    sub: `${measured} measured datasets, each sha256-verified · replaying ${DATA.dataset}`,
    card: ['L1 · Data', `${loaders} loaders`, `<div class="info-rows">${Object.entries(DATA.catalog)
      .map(([signal, ids]) => row(TYPE[signal] || signal, `${ids.length}`)).join('')}
      ${row(DATA.dataset, `${number(DATA.trainingFingerprints)} train · ${number(DATA.evaluationSamples)} test`)}</div>
      <div class="info-note">${number(refCount)} surveyed positions · ${DATA.buildings.length} buildings · ${floors} floors · missing readings are NaN, coordinates stay in the source frame</div>`]},
  l2: {kicker: 'L2 <span>·</span> Signals', headline: `One scan becomes a ${count}-value fingerprint.`,
    sub: `test.X[${scanIndex}] hears ${heardCount} of ${count} access points`,
    card: ['L2 · Signals', `test.X[${scanIndex}]`, `<div class="info-rows">${row('Access points heard', `${heardCount} / ${count}`)}
      ${row('Strongest', `${strongest[0]} dBm`)}${row('Not heard', 'NaN → −104 dBm')}${row('Transform', 'FillMissing(−104)')}</div>
      <div class="info-note">RSSI representations · device calibration · augmentation · CSI phase sanitizing · ranging · IMU · magnetic · VLC</div>`]},
  l3: {kicker: 'L3 <span>·</span> Methods', headline: 'Any model, one interface.',
    sub: `WKNN matches the ${DATA.k} nearest of ${number(DATA.trainingFingerprints)} fingerprints`,
    card: ['L3 · Methods', `WKNN · k = ${DATA.k}`, `<div class="info-rows">${row('Nearest fingerprints', `${DATA.k}, at ${refs.length} positions`)}
      ${row('Error for this scan', `${result.error.toFixed(2)} m`)}${row('Floor', floorText)}</div>
      <div class="info-note">Fingerprinting (kNN · Horus · GP · forests · ensembles) · model-based (trilateration · Chan TDoA · MUSIC) · deep · transfer: the same fit / localize / evaluate</div>`]},
  l4: {kicker: 'L4 <span>·</span> Evaluation', headline: 'Every held-out scan, one protocol.',
    sub: `${number(DATA.evaluationSamples)} scans of the official split · recorded WKNN run`,
    card: ['L4 · Evaluation', `${number(DATA.evaluationSamples)} scans`, `<div class="info-rows">${row('Mean error', `${metrics.mean_error.toFixed(2)} m`)}
      ${row('Median error', `${metrics.median_error.toFixed(2)} m`)}${row('P90 error', `${metrics.p90_error.toFixed(2)} m`)}
      ${row('Floor accuracy', `${metrics.floor_accuracy.toFixed(2)} %`)}</div>
      <div class="info-note">Bars: error histogram (≤ 5 m · 5–20 m · &gt; 20 m) · line: share of scans within each error</div>`]},
  l5: {kicker: 'L5 <span>·</span> Applications', headline: 'Tracking and fusion on real phone traces.',
    sub: 'WiFi fixes → dead reckoning + a particle filter that keeps to the walls',
    card: ['L5 · Applications', `${APPS.testTraces} held-out traces`, `<div class="info-rows">${row('WiFi WKNN', mean('WiFi WKNN'))}
      ${row('+ Kalman RTS smoother', mean('+ Kalman RTS'))}${row('PDR + WiFi + floor plan', mean('PDR + WiFi + map'))}</div>
      <div class="info-note">${appsNote}</div>`]},
  outro: {kicker: 'IndoorLoc', headline: 'Five layers. Use one, or all of them.',
    sub: 'Every number shown reruns with <code>python3 -m examples.readme_demo</code>', card: ['Use a single layer', '', exits]},
};
const LINE = {l1: 1, l2: 2, l3: 3, l4: 4, l5: 5};
$('code-index').textContent = scanIndex;
document.querySelectorAll('[data-line]').forEach(node => {
  const i = Number(node.dataset.line);
  if (i) node.style.setProperty('--c', `var(${LAYERS[i - 1].color})`);
});

const bufferSize = new THREE.Vector2();
let currentAct = '';
function renderAt(seconds) {
  const t = ((seconds % T.end) + T.end) % T.end;
  const act = actAt(t);
  const active = layerOf(act);
  placeCamera(t);
  const keys = cameraKeys(), cut = hermite(keys, t, p => p.cut), open = hermite(keys, t, p => p.open);
  plates.forEach((p, i) => p.group.position.y = level(i) + open * (OPEN_UP * smooth(clamp01(i - cut)) - OPEN_DOWN * smooth(clamp01(cut - i))));
  scene.updateMatrixWorld();
  const clear = 1 - ramp(t, CLEAR, CLEAR + .8);
  const shown = i => ramp(t, T[ACT_KEYS[i]] + .1, T[ACT_KEYS[i]] + .9) * clear;
  const dim = i => active < 0 || active === i ? 1 : i < active ? .3 : .5;

  // Plates: the active layer is full strength; a light sweep rises through the stack at the start.
  plates.forEach((p, i) => {
    const sweep = t < T.l1 ? Math.exp(-(((t - .3) / 1.6 * 5 - i) ** 2) / .5) : 0;
    const focus = active < 0 ? .85 : i === active ? 1 : .45;
    p.top.material.uniforms.opacity.value = lerp(focus, 1, sweep) * (1 + .6 * sweep);
    place(p.label, p.group, new THREE.Vector3(-W / 2 - 6, 0, D / 2), active < 0 ? .95 : i === active ? 1 : .5);
    if (p.exit) place(p.exit, p.group, new THREE.Vector3(W / 2 + 6, 0, -D / 2 + 14),
      Math.max(i === active ? ramp(t, T[act] + 1.2, T[act] + 1.8) : 0, within(t, T.outro + 1.0, T.end - .3, .4, .4)));
  });

  // L1: tiles, then the radio map of the verified dataset.
  const s1 = shown(0);
  tiles.forEach(tile => {
    const delay = tile.row * .06 + tile.column * .1;
    const pop = ramp(t, T.l1 + .1 + delay, T.l1 + .5 + delay) * clear;
    tile.mesh.material.uniforms.opacity.value = pop * (tile.verified ? 1 : lerp(1, .6, 1 - dim(0)));
    tile.mesh.position.y = (1 - pop) * 10 + (tile.verified ? 2 * ramp(t, T.l1 + .9, T.l1 + 1.3) : 0);
  });
  typeTags.forEach(({node, at}) => place(node, plates[0].group, at, s1 * (active === 0 ? 1 : 0)));
  place(verifiedTag.node, plates[0].content, verifiedTag.at.clone().setY(4), ramp(t, T.l1 + 1.0, T.l1 + 1.5) * clear * (active === 0 ? 1 : 0));
  verifiedTag.node.style.transform = 'translate(20px,-50%)';
  const bloom = ramp(t, T.l1 + 1.0, T.l1 + 2.2) * clear;
  const ca = campusPoints.geometry.attributes;
  for (let i = 0; i < refCount; i++) {
    const a = clamp01((bloom - (reference.xy[i][0] + 200) / 400 * .45) / .55);
    layerRGB(0).toArray(ca.aColor.array, i * 3);
    ca.aAlpha.array[i] = .85 * a * dim(0);
    ca.aSize.array[i] = 2.2;
  }
  ca.aColor.needsUpdate = ca.aAlpha.needsUpdate = ca.aSize.needsUpdate = true;

  // L2: the 520 inputs grow; heard access points stand out.
  const s2 = ramp(t, T.l2 + .2, T.l2 + 1.6) * clear;
  entry.features.forEach((v, i) => {
    const grow = clamp01(s2 * 1.3 - i / count * .3);
    const height = heard[i] ? Math.max(1.2, v * BAR_HEIGHT) : .7;
    matrix.compose(position.set(barX(i), 0, 0), rotation, size.set(.42, Math.max(height * grow, .01), heard[i] ? 5 : 3));
    bars.setMatrixAt(i, matrix);
  });
  bars.instanceMatrix.needsUpdate = true;
  bars.visible = s2 > .001;
  bars.material.opacity = clear * lerp(.6, 1, dim(1) === 1 ? 1 : 0);
  axisTags2.forEach(({node, at}) => place(node, plates[1].content, at, s2 * (active === 1 ? 1 : 0)));
  place(peakTag.node, plates[1].content, peakTag.at, ramp(t, T.l2 + 1.3, T.l2 + 1.8) * (active === 1 ? 1 : 0));

  // L3: the model chips, nearby positions, the 5 nearest, the WKNN estimate and the truth.
  const s3 = shown(2);
  models.forEach(({mesh, on, node, at}, j) => {
    const pop = ramp(t, T.l3 + .1 + j * .08, T.l3 + .5 + j * .08) * clear;
    mesh.material.uniforms.opacity.value = pop * (on ? 1 : .8) * dim(2);
    place(node, plates[2].content, at, pop * (active === 2 ? 1 : 0));
  });
  const ma = mapPoints.geometry.attributes, na = neighborPoints.geometry.attributes;
  for (let j = 0; j < nearby.length; j++) {
    layerRGB(2).toArray(ma.aColor.array, j * 3);
    ma.aAlpha.array[j] = .7 * s3 * dim(2); ma.aSize.array[j] = 2.4;
  }
  const matched = ramp(t, T.l3 + .7, T.l3 + 1.2) * clear;
  for (let j = 0; j < refs.length; j++) {
    COLOR.match.toArray(na.aColor.array, j * 3);
    na.aAlpha.array[j] = matched * dim(2); na.aSize.array[j] = 5; na.aHalo.array[j] = .9;
  }
  ma.aColor.needsUpdate = ma.aAlpha.needsUpdate = ma.aSize.needsUpdate = true;
  na.aColor.needsUpdate = na.aAlpha.needsUpdate = na.aSize.needsUpdate = na.aHalo.needsUpdate = true;
  const tip = local(result.xy).setY(2);
  const grow3 = ramp(t, T.l3 + 1.1, T.l3 + 1.8);
  entry.neighbors.forEach((n, j) => setRod(links[j], local(reference.xy[n.reference]).setY(1.4), tip, .5, grow3, .9 * matched * dim(2)));
  const drop = ramp(t, T.l3 + 1.6, T.l3 + 2.1) * clear, reveal = ramp(t, T.l3 + 2.2, T.l3 + 2.7) * clear;
  for (const [item, visible] of [[predictionPin, drop * dim(2)], [truthPin, reveal * dim(2)]])
    item.parts.forEach((part, j) => { part.material.opacity = item.base[j] * visible; part.visible = visible > .003; });
  predictionPin.group.position.y = (1 - drop) * 20;
  const errorGrow = ramp(t, T.l3 + 2.5, T.l3 + 3.0);
  setRod(errorRod, local(truth.xy).setY(2.5), local(result.xy).setY(2.5), .4, errorGrow, .95 * clear * dim(2));
  const focus3 = active === 2 ? 1 : 0;
  place(truthTag, plates[2].content, local(truth.xy).setY(16), reveal * focus3);
  place(predictionTag, plates[2].content, local(result.xy).setY(16), drop * focus3);
  place(errorTag, plates[2].content, local(truth.xy).lerp(local(result.xy), .5).setY(2).add(new THREE.Vector3(0, 0, 12)), errorGrow * focus3);
  const nearestAt = refs.reduce((sum, r) => sum.add(local(reference.xy[r])), new THREE.Vector3()).divideScalar(refs.length);
  place(nearestTag, plates[2].content, nearestAt.setY(6).add(new THREE.Vector3(-14, 0, 0)), within(t, T.l3 + .8, T.l3 + 2.0, .3, .4) * focus3);

  // L4: histogram bars rise, then the cumulative line is drawn across them.
  const s4 = ramp(t, T.l4 + .2, T.l4 + 1.4) * clear;
  bins.forEach((n, b) => {
    const grow = clamp01(s4 * 1.4 - b / BINS * .4);
    matrix.compose(position.set(-136 + 272 * (b + .5) / BINS, 0, 10), rotation, size.set(272 / BINS * .78, Math.max(n / maxBin * 40 * grow, .01), 26));
    histogram.setMatrixAt(b, matrix);
  });
  histogram.instanceMatrix.needsUpdate = true;
  histogram.visible = s4 > .001;
  histogram.material.opacity = clear * (dim(3) === 1 ? 1 : .6);
  const draw = ramp(t, T.l4 + 1.0, T.l4 + 2.6) * clear;
  cdfRods.forEach((mesh, j) => {
    const visible = clamp01(draw * CDF_STEPS - j);
    setRod(mesh, cdfAt(j / CDF_STEPS), cdfAt((j + 1) / CDF_STEPS), .55, Math.max(visible, .001), visible > 0 ? .95 * clear * dim(3) : 0);
  });
  axisTags4.forEach(({node, at}) => place(node, plates[3].content, at, s4 * (active === 3 ? 1 : 0)));

  // L5: component chips, the floor plan, the waypoints, then the fixes arrive and the fused track follows them.
  const focus5 = active === 4 ? 1 : 0;
  components.forEach(({mesh, node, at}, j) => {
    const pop = ramp(t, T.l5 + .1 + j * .07, T.l5 + .5 + j * .07) * clear;
    mesh.material.uniforms.opacity.value = pop * dim(4);
    place(node, plates[4].content, at, pop * focus5);
  });
  const plan = ramp(t, T.l5 + .3, T.l5 + 1.0) * clear;
  wallRods.forEach(({mesh, a, b}) => setRod(mesh, a, b, .28, 1, .55 * plan * dim(4)));
  const walk = ramp(t, T.l5 + .6, T.l5 + 1.5);
  truthRods.forEach(({mesh, a, b}, j) => {
    const grow = clamp01(walk * truthRods.length - j);
    setRod(mesh, a, b, .7, Math.max(grow, .001), grow > 0 ? .95 * clear * dim(4) : 0);
  });
  const run = ramp(t, T.l5 + 1.3, T.l5 + 3.1);
  const fa = fixPoints.geometry.attributes;
  for (let j = 0; j < APPS.fixes.length; j++) {
    const on = clamp01((run * 1.05 - j / APPS.fixes.length) * 12);
    COLOR.reference.toArray(fa.aColor.array, j * 3);
    fa.aAlpha.array[j] = .9 * on * clear * dim(4); fa.aSize.array[j] = 3.2; fa.aHalo.array[j] = .5;
  }
  fa.aColor.needsUpdate = fa.aAlpha.needsUpdate = fa.aSize.needsUpdate = fa.aHalo.needsUpdate = true;
  fusedRods.forEach(({mesh, a, b}, j) => {
    const grow = clamp01(run * fusedRods.length - j);
    setRod(mesh, a, b, .75, Math.max(grow, .001), grow > 0 ? clear * dim(4) : 0);
  });
  place(truthTag5, plates[4].content, appAt(truthLabelAt, 4), ramp(t, T.l5 + 1.0, T.l5 + 1.5) * clear * focus5);
  place(fixTag5, plates[4].content, appAt(fixLabelAt, 4), ramp(t, T.l5 + 1.8, T.l5 + 2.3) * clear * focus5);
  place(fusedTag5, plates[4].content, appAt(fusedLabelAt, 4), ramp(t, T.l5 + 2.4, T.l5 + 2.9) * clear * focus5);

  const pixels = renderer.getDrawingBufferSize(bufferSize).y / (2 * Math.tan(camera.fov * Math.PI / 360));
  clouds.forEach(cloud => { cloud.material.uniforms.scale.value = pixels; cloud.material.uniforms.perUnit.value = 3.2 * renderer.getPixelRatio(); });
  renderer.render(scene, camera);

  // Captions, card and code.
  if (act !== currentAct) {
    const text = TEXT[act];
    $('kicker').innerHTML = text.kicker;
    $('headline').innerHTML = text.headline;
    $('subline').innerHTML = text.sub;
    $('info-title').textContent = text.card[0];
    $('info-badge').textContent = text.card[1];
    $('info-body').innerHTML = text.card[2];
    $('info-card').style.borderColor = active >= 0 ? css(LAYERS[active].color) : '';
    document.querySelectorAll('[data-line]').forEach(node => node.classList.toggle('active', Number(node.dataset.line) === LINE[act]));
    currentAct = act;
  }
  const bounds = {intro: [0, T.l1], l1: [T.l1, T.l2], l2: [T.l2, T.l3], l3: [T.l3, T.l4], l4: [T.l4, T.l5], l5: [T.l5, T.outro], outro: [T.outro, T.end]}[act];
  const textShown = within(t, bounds[0] + (act === 'intro' ? .1 : 0), bounds[1], .35, act === 'outro' ? .45 : .3);
  $('caption').style.opacity = textShown.toFixed(3);
  const cardShown = active >= 0 ? textShown : 0;
  $('info-card').style.opacity = cardShown.toFixed(3);
  $('info-card').style.visibility = cardShown > .003 ? 'visible' : 'hidden';
  $('info-card').style.transform = `translateY(${((1 - cardShown) * 10).toFixed(1)}px)`;
  window.__sceneState = {time: t, act, index: scanIndex, error: result.error, layer: active};
}

// Playback, interaction and the deterministic export hook.
let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches, manual = false, dirty = true;
let start = performance.now(), pausedAt = playing ? 0 : T.outro + 1.6, lastFrame = 0;
function resize() {
  const scale = fitStage();
  const ratio = devicePixelRatio * scale;  // Exports supersample through the device scale; viewing caps it.
  renderer.setPixelRatio(manual ? ratio : Math.max(1, Math.min(ratio, 2)));
  renderer.setSize(1280, 720, false);
  dirty = true;
}
function now() { return playing ? (performance.now() - start) / 1000 : pausedAt; }
function tick(time) {
  if (!manual && !document.hidden && (dirty || (playing && time - lastFrame >= 1000 / 30))) {
    renderAt(now()); lastFrame = time; dirty = false;
  }
  requestAnimationFrame(tick);
}
function syncPlayButton() {
  $('play').textContent = playing ? tr('Pause', '暂停') : tr('Play', '播放');
  $('play').setAttribute('aria-label', playing ? tr('Pause animation', '暂停动画') : tr('Play animation', '播放动画'));
}
function setPlaying(value) {
  if (value === playing) return;
  if (value) start = performance.now() - pausedAt * 1000; else pausedAt = now();
  playing = value;
  syncPlayButton();
  dirty = true;
}
$('play').addEventListener('click', () => setPlaying(!playing));
let dragX = null;
renderer.domElement.addEventListener('pointerdown', e => { dragX = e.clientX; renderer.domElement.setPointerCapture(e.pointerId); });
renderer.domElement.addEventListener('pointermove', e => { if (dragX === null) return; orbit += (e.clientX - dragX) * .006; dragX = e.clientX; dirty = true; });
renderer.domElement.addEventListener('pointerup', () => dragX = null);
renderer.domElement.addEventListener('pointercancel', () => dragX = null);
renderer.domElement.addEventListener('wheel', e => { e.preventDefault(); zoom = Math.max(.45, Math.min(1.8, zoom * Math.exp(e.deltaY * .001))); dirty = true; }, {passive: false});
window.addEventListener('keydown', e => { if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); setPlaying(!playing); } });
window.addEventListener('resize', resize);
document.addEventListener('visibilitychange', () => { if (document.hidden && playing) pausedAt = now(); else if (playing) start = performance.now() - pausedAt * 1000; });
window.__renderScene = seconds => {
  if (!manual) { manual = true; stage.classList.add('exporting'); resize(); }
  renderAt(seconds);
  return window.__sceneState;
};
window.__sceneDuration = T.end;
window.__scenePosterTime = T.outro + 1.6;  // The whole stack with every layer filled in.
syncPlayButton();
resize();
renderAt(now());
stage.classList.add('ready');
window.__sceneReady = true;
requestAnimationFrame(tick);
