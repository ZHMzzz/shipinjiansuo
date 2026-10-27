(() => {
  const el = id => document.getElementById(id);
  let videos = [], jobs = [], pollTimer = null, completionKey = '', uploading = false, polling = false, pollErrorMessage = '';
  const terminal = new Set(['completed', 'failed']);
  const formatTime = n => `${Math.floor(n / 60)}:${Math.floor(n % 60).toString().padStart(2, '0')}`;
  const formatSize = n => n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GB` : `${(n / 1024 ** 2).toFixed(1)} MB`;
  function node(tag, text, cls) { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; }
  async function api(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `请求失败 (${response.status})`);
    return data;
  }
  function notice(message, isError = false) {
    el('upload-notice').textContent = message;
    el('upload-notice').classList.toggle('notice-error', isError);
  }
  function playVideo(video) {
    const player = el('player');
    el('player-title').textContent = video.name;
    el('player-caption').textContent = `视频库播放 · 时长 ${formatTime(video.duration)} · ${video.width} × ${video.height}`;
    player.src = video.url;
    player.onloadedmetadata = () => { player.currentTime = 0; player.play().catch(() => {}); };
    el('player-dialog').showModal();
  }
  function renderVideos() {
    const query = el('library-filter').value.trim().toLowerCase();
    const filtered = videos.filter(v => v.name.toLowerCase().includes(query));
    el('video-count').textContent = `(${videos.length})`;
    const grid = el('video-library-grid');
    grid.replaceChildren();
    if (!filtered.length) { grid.append(node('p', query ? '没有匹配此名称的视频。' : '还没有已入库视频，请先上传。', 'empty')); return; }
    for (const video of filtered) {
      const card = node('button', undefined, 'result library-video');
      card.type = 'button';
      card.title = `从头播放：${video.name}`;
      const picture = node('div', undefined, 'thumb');
      const img = node('img'); img.src = video.thumbnail; img.alt = video.name; img.loading = 'lazy';
      picture.append(img, node('span', '已入库', 'rank'), node('span', formatTime(video.duration), 'time'));
      const info = node('div', undefined, 'result-info');
      const title = node('h3', video.name); title.title = video.name;
      const meta = node('p'); meta.append(node('span', formatSize(video.size_in_bytes)), node('span', '从头播放 ↗'));
      const features = node('div', undefined, 'video-capabilities');
      for (const [key, label] of [['owlv2', '物体'], ['insightface', '人脸']]) {
        const available = Object.entries(video.features).some(([model, count]) => model.includes(key) && count > 0);
        features.append(node('span', available ? `${label}可检索` : `未检出${label}特征`, available ? '' : 'muted'));
      }
      features.append(node('span',video.action_segments ? `图文 / 动作可检索 · ${video.action_segments} 段` : 'Qwen 片段待补建',video.action_segments ? '' : 'muted'));
      info.append(title, meta, features); card.append(picture, info); card.onclick = () => playVideo(video); grid.append(card);
    }
  }
  async function loadVideos() {
    try {
      const data = await api('/mvp-api/videos'); videos = data.videos;
      el('library-summary').textContent = `${data.count} 个视频 · 总时长 ${formatTime(data.total_duration)} · 点击任一视频直接播放`;
      el('library-error').hidden = true; renderVideos();
      el('library').textContent = `本机视频库已连接 · ${data.count} 段素材 / ${videos.reduce((n,v)=>n+v.action_segments,0)} 个 Qwen 片段 / 可定位播放`;
    } catch (error) { el('library-error').hidden = false; el('library-error').textContent = `视频库读取失败：${error.message}`; }
  }
  function renderJobs() {
    const container = el('upload-jobs'); container.replaceChildren();
    const active = jobs.filter(j => !terminal.has(j.status)).length;
    el('job-count').hidden = !active; el('job-count').textContent = active;
    if (!jobs.length) { container.append(node('p', '暂无上传任务。新视频处理完成后会自动出现在下方。', 'help')); return; }
    for (const job of jobs) {
      const row = node('article', undefined, `job-row ${job.status}`); row.dataset.jobId = job.id;
      const heading = node('div', undefined, 'job-heading');
      const name = node('strong', job.filename); name.title = job.filename;
      heading.append(name, node('span', job.stage, 'job-stage'));
      const progress = node('progress'); progress.max = 100; progress.value = job.progress;
      progress.setAttribute('aria-label', `${job.filename} 处理进度`);
      const details = node('div', undefined, 'job-details');
      const pct = terminal.has(job.status) ? (job.status === 'completed' ? '100%' : '') : `约 ${Math.floor(job.progress)}%`;
      details.append(node('span', job.message), node('span', pct));
      row.append(heading, progress, details);
      if (job.status === 'failed') {
        const retry = node('button', '重试处理', 'secondary'); retry.type = 'button';
        retry.onclick = async () => { retry.disabled = true; try { await api(`/mvp-api/jobs/${job.id}/retry`, {method:'POST'}); await pollJobs(); } catch (e) { notice(e.message, true); retry.disabled = false; } };
        row.append(retry);
      } else if (job.status === 'completed') {
        const view = node('button', '播放视频', 'secondary'); view.type = 'button';
        view.onclick = async () => { await loadVideos(); const v = videos.find(v => v.id === job.media_id); if (v) playVideo(v); };
        row.append(view);
      }
      container.append(row);
    }
  }
  async function pollJobs() {
    if (polling) return;
    polling = true;
    clearTimeout(pollTimer);
    try {
      const data = await api('/mvp-api/jobs'); jobs = data.jobs; renderJobs();
      if (pollErrorMessage && el('upload-notice').textContent === pollErrorMessage) notice('');
      pollErrorMessage = '';
      const key = jobs.filter(j => j.status === 'completed').map(j => j.id).sort().join(',');
      if (key !== completionKey) { completionKey = key; await loadVideos(); }
    } catch (e) { if (!el('library-panel').hidden) { pollErrorMessage = `任务状态暂时不可用：${e.message}`; notice(pollErrorMessage, true); } }
    const pending = jobs.some(j => !terminal.has(j.status));
    polling = false;
    pollTimer = setTimeout(pollJobs, pending || !el('library-panel').hidden ? 2000 : 10000);
  }
  window.refreshVideoLibrary = loadVideos;
  window.showSearch = () => {
    el('library-panel').hidden = true; document.querySelector('.search-area').hidden = false;
    el('library-tab').classList.remove('active'); history.replaceState(null, '', location.pathname);
  };
  async function showLibrary() {
    document.querySelector('.search-area').hidden = true; el('library-panel').hidden = false;
    document.querySelectorAll('nav [data-mode]').forEach(button => button.classList.remove('active'));
    el('library-tab').classList.add('active'); history.replaceState(null, '', '#library');
    await Promise.all([loadVideos(), pollJobs()]);
  }
  el('library-tab').onclick = showLibrary;
  el('refresh-library').onclick = () => Promise.all([loadVideos(), pollJobs()]);
  el('library-filter').oninput = renderVideos;
  el('video-upload').onchange = () => {
    const file = el('video-upload').files[0];
    el('selected-video').textContent = file ? `${file.name} · ${formatSize(file.size)}` : '尚未选择文件';
  };
  el('upload-form').onsubmit = event => {
    event.preventDefault(); if (uploading) return;
    const file = el('video-upload').files[0];
    if (!file) return notice('请先选择视频文件', true);
    if (file.size > 1024 ** 3) return notice('单个视频最大 1GB，请选择较小文件', true);
    uploading = true; el('upload-button').disabled = true; el('video-upload').disabled = true;
    el('upload-transfer').hidden = false; el('transfer-progress').value = 0; notice('正在上传视频…');
    const form = new FormData(); form.append('video', file);
    const xhr = new XMLHttpRequest(); xhr.open('POST', '/mvp-api/uploads');
    xhr.upload.onprogress = event => { if (event.lengthComputable) { const value = Math.round(event.loaded/event.total*100); el('transfer-progress').value = value; el('transfer-label').textContent = value === 100 ? '上传完成，正在登记任务…' : `上传中 ${value}%`; } };
    xhr.onload = async () => {
      try {
        const data = JSON.parse(xhr.responseText);
        if (xhr.status >= 400) throw new Error(data.detail || '上传失败');
        notice(data.duplicate ? `此文件已有任务：${data.job.stage}，未重复入库。` : '上传成功，后台开始处理。可继续检索已入库的视频。');
        el('video-upload').value = ''; el('selected-video').textContent = '尚未选择文件';
        await pollJobs();
      } catch (e) { notice(e.message, true); }
    };
    xhr.onerror = () => notice('连接中断，上传未确认。请刷新任务列表后再重试。', true);
    xhr.onloadend = () => { uploading = false; el('upload-button').disabled = false; el('video-upload').disabled = false; el('upload-transfer').hidden = true; };
    xhr.send(form);
  };
  pollJobs();
  if (location.hash === '#library') showLibrary();
})();
