import {el, node, api, feedback, formatTime, formatSize} from './api.js';
import {mountMaintenance} from './maintenance.js';
import {createVideoDeletion, trashIcon} from './video-delete.js';

const PAGE_SIZE = 50;
let catalog = [], items = [], page = 1, mode = 'browse', applied = {}, limited = false;
let request = null, generation = 0, previewURL = null, qwenLoaded = null;
const natural = el('natural'), reference = el('reference');
const labels = {text:'语义片段', object:'物体图片', face:'人物', person_scene:'人物与场景', person_object:'人物与物体'};

const openDeletion = createVideoDeletion(async (item,result) => {
  cancelSearch();
  catalog = catalog.filter(v => v.id !== item.id); items = items.filter(v => v.id !== item.id); render();
  document.dispatchEvent(new Event('video-library-changed'));
  el('search-status').textContent = result.cleanup_pending ? '视频已从检索库移除，部分历史文件正在重试清理。' : `已删除「${item.name}」及对应检索索引。`;
  try { await loadCatalog(); } catch(e) { feedback('search-error','删除已完成，列表刷新失败：'+e.message,true); }
});

function syncInputs() {
  const hasImage = Boolean(reference.files[0]);
  natural.disabled = hasImage;
  reference.disabled = Boolean(natural.value.trim());
  reference.closest('.image-picker').classList.toggle('unavailable', reference.disabled);
  el('clear-reference').hidden = !hasImage;
  reference.closest('.image-input').classList.toggle('has-image', hasImage);
}
function clearReference() {
  reference.value = ''; if (previewURL) URL.revokeObjectURL(previewURL); previewURL = null;
  el('image-preview').hidden = true; el('image-preview').removeAttribute('src');
  el('image-prompt').textContent = '▧　选择参考图片'; syncInputs();
}
reference.addEventListener('change', () => {
  if (previewURL) URL.revokeObjectURL(previewURL);
  const file = reference.files[0];
  if (!file) return clearReference();
  if (file.size > 10 * 1024 ** 2) { clearReference(); feedback('search-error', '参考图片最大 10 MB', true); return; }
  previewURL = URL.createObjectURL(file); el('image-preview').src = previewURL;
  el('image-preview').hidden = false; el('image-prompt').textContent = file.name;
  feedback('search-error'); syncInputs();
});
natural.addEventListener('input', syncInputs);
el('person').addEventListener('change', syncInputs);
el('clear-reference').onclick = event => { event.preventDefault(); clearReference(); reference.focus(); };

function play(item) {
  const dialog = el('video-dialog'), player = dialog.querySelector('video');
  dialog.querySelector('h3').textContent = item.name;
  el('play-caption').textContent = item.kind ? `${labels[item.kind]}命中 · 从 ${formatTime(item.start)} 开始播放` : `时长 ${formatTime(item.duration)} · ${item.width} × ${item.height}`;
  player.onloadedmetadata = () => { player.currentTime = Math.min(item.start || 0, player.duration || Infinity); player.play().catch(() => {}); };
  player.src = item.url; dialog.showModal();
}
el('close-video').onclick = () => el('video-dialog').close();
el('video-dialog').addEventListener('close', () => {
  const player = el('video-dialog').querySelector('video'); player.pause(); player.removeAttribute('src'); player.load();
});
function render() {
  const pages = Math.max(1, Math.ceil(items.length / PAGE_SIZE)); page = Math.min(page, pages);
  el('list-title').textContent = mode === 'browse' ? (applied.filename ? '筛选视频' : '全部视频') : '查询结果';
  el('count').textContent = `${items.length} 个视频`;
  const grid = el('video-grid'); grid.replaceChildren();
  if (!items.length) grid.append(node('p', '没有找到视频，请调整条件或新增视频。', 'empty'));
  for (const item of items.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)) {
    const card = node('article', undefined, 'video-card');
    const open = node('button', undefined, 'video-open'); open.type = 'button';
    open.title = `播放 ${item.name}${item.kind ? ' · ' + formatTime(item.start) : ''}`;
    const thumb = node('div', undefined, 'thumbnail');
    const img = node('img'); img.src = new URL(item.thumbnail, location.origin + '/demo/').href;
    img.alt = ''; img.loading = 'lazy';
    thumb.append(img, node('span', item.kind ? formatTime(item.start) : formatTime(item.duration), 'duration'), node('span', '▶', 'play'));
    const info = node('div', undefined, 'card-info');
    const title = node('h3', item.name); title.title = item.name; info.append(title);
    open.append(thumb, info); open.onclick = () => play(item);
    const footer = node('div', undefined, 'card-footer');
    const meta = node('div', undefined, 'card-meta');
    meta.append(node('span', item.kind ? labels[item.kind] : `${item.width} × ${item.height}`), node('span', item.kind ? `${formatTime(item.start)}–${formatTime(item.end)}` : formatSize(item.size_in_bytes)));
    const remove = node('button', undefined, 'delete-video-button'); remove.type = 'button';
    remove.innerHTML = trashIcon + '<span>删除</span>'; remove.setAttribute('aria-label',`删除视频：${item.name}`);
    remove.onclick = () => openDeletion(item); footer.append(meta,remove);
    card.append(open,footer); grid.append(card);
  }
  el('range').textContent = items.length ? `${(page - 1) * PAGE_SIZE + 1}–${Math.min(page * PAGE_SIZE, items.length)} / ${items.length} 条${limited ? ' · 当前返回的候选结果' : ''}` : '0 条';
  const nav = el('pages'); nav.replaceChildren();
  function pageButton(label, target, disabled, current = false) {
    const b = node('button', label, current ? 'current' : ''); b.type = 'button'; b.disabled = disabled;
    if (current) b.setAttribute('aria-current', 'page');
    b.setAttribute('aria-label', typeof label === 'number' ? `第 ${label} 页` : label);
    b.onclick = () => { page = target; render(); el('list-title').scrollIntoView({block:'nearest'}); }; nav.append(b);
  }
  pageButton('‹', page - 1, page === 1);
  const visiblePages = new Set([1, pages, page - 1, page, page + 1]);
  let previous = 0;
  for (const n of [...visiblePages].filter(n => n >= 1 && n <= pages).sort((a,b) => a-b)) {
    if (previous && n - previous > 1) nav.append(node('span','…'));
    pageButton(n, n, false, n === page); previous = n;
  }
  pageButton('›', page + 1, page === pages);
}
async function loadCatalog(refresh = false) {
  const version = generation;
  const data = await api('/mvp-api/videos'); catalog = data.videos;
  if (version !== generation) return;
  if (mode === 'browse' && !request) {
    items = catalog.filter(v => v.name.toLocaleLowerCase().includes((applied.filename || '').toLocaleLowerCase())); render();
  } else if (refresh) el('search-status').textContent = '新视频已入库，重新查询可更新当前结果。';
}
function setSearchBusy(busy) {
  el('search-button').disabled = busy;
  el('search-label').textContent = busy ? '查询中…' : '查询';
}
function cancelSearch() {
  generation++; request?.abort(); request = null;
  setSearchBusy(false);
  el('video-grid').removeAttribute('aria-busy');
}
el('query-form').onsubmit = async event => {
  event.preventDefault(); cancelSearch(); feedback('search-error');
  const input = {query:natural.value.trim(), person:el('person').value.trim(), filename:el('filename').value.trim(), image:reference.files[0]?.name};
  if (input.query && input.image) { feedback('search-error', '自然语言与参考图片请分开查询', true); return; }
  const form = new FormData();
  for (const key of ['query','person','filename']) form.set(key,input[key]);
  if (reference.files[0]) form.set('photo',reference.files[0]);
  const current = generation, controller = new AbortController(); request = controller;
  setSearchBusy(true);
  el('search-status').textContent = input.query && qwenLoaded === false ? '正在查询，首次加载自然语言模型可能需要一些时间…' : '正在查询…'; el('video-grid').setAttribute('aria-busy','true');
  try {
    const data = await api('/mvp-api/library/search',{method:'POST',body:form,signal:controller.signal});
    if (current !== generation) return;
    if (data.mode === 'text' || (data.mode === 'person_scene' && data.items.length)) qwenLoaded = true;
    items = data.items; mode = data.mode; limited = data.limited; applied = input; page = 1; render();
    el('search-status').textContent = '';
  } catch (error) {
    if (current === generation && error.name !== 'AbortError') { feedback('search-error', error.message, true); el('search-status').textContent = '查询未完成，仍显示上次结果。'; }
  } finally {
    if (current === generation) { request = null; setSearchBusy(false); el('video-grid').removeAttribute('aria-busy'); }
  }
};
el('reset').onclick = async () => {
  cancelSearch(); el('query-form').reset(); clearReference();
  applied = {}; mode = 'browse'; limited = false; page = 1; items = catalog;
  feedback('search-error'); el('search-status').textContent = ''; render();
  try { await loadCatalog(); } catch (e) { feedback('search-error', '视频库读取失败：' + e.message, true); }
};

async function checkIndex() {
  try {
    const state = await api('/mvp-api/actions/status');
    qwenLoaded = state.model_loaded;
    el('index-status').hidden = state.status === 'ready';
    el('index-status').replaceChildren(node('span', state.status === 'failed' ? '自然语言索引处理失败，请重试或查看上传任务。' : '自然语言索引正在处理，新视频完成后即可检索。'));
    if (state.status === 'failed') {
      const retry = node('button','重试','outline-button'); retry.onclick = async () => {
        retry.disabled = true;
        try { await api('/mvp-api/actions/retry',{method:'POST'}); await checkIndex(); }
        catch (e) { feedback('search-error', e.message, true); retry.disabled = false; }
      }; el('index-status').append(retry);
    }
  } catch (e) { feedback('index-status', '暂时无法读取索引状态：' + e.message, true); }
}
const cleanURL = new URL(location.href); cleanURL.searchParams.delete('variant'); cleanURL.searchParams.delete('modal'); history.replaceState(null,'',cleanURL);
syncInputs();
try { await loadCatalog(); } catch (e) { feedback('search-error','视频库读取失败：' + e.message, true); }
try {
  await mountMaintenance({onPeopleChanged:names => {
    const select = el('person'), selected = select.value;
    const all = node('option', '全部人物'); all.value = '';
    select.replaceChildren(all, ...names.map(name => { const option = node('option', name); option.value = name; return option; }));
    select.value = names.includes(selected) ? selected : '';
    syncInputs();
  }, onLibraryChanged:() => loadCatalog(true).catch(e => feedback('search-error',e.message,true))});
} catch (e) { feedback('search-error','维护窗口加载失败，请刷新页面：' + e.message,true); }
checkIndex(); setInterval(checkIndex, 15000);
