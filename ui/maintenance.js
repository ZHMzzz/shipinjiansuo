import {el, node, api, feedback, errorMessage, formatSize} from './api.js';

export async function mountMaintenance({onPeopleChanged, onLibraryChanged}) {
  const response = await fetch('/mvp/maintenance.html', {cache:'no-store'});
  if (!response.ok) throw new Error('无法加载维护窗口');
  // This is a local static template; user content is always rendered with textContent.
  const template = document.createElement('template'); template.innerHTML = await response.text(); document.body.append(template.content);
  const uploadModal = el('upload-modal'), peopleModal = el('people-modal');
  for (const button of document.querySelectorAll('[data-close]')) button.onclick = () => el(button.dataset.close).close();
  el('open-upload').onclick = () => { uploadModal.showModal(); pollJobs(); };
  el('open-people').onclick = () => { peopleModal.showModal(); loadPeople().catch(e => feedback('person-feedback', e.message, true)); };
  let videoFile = null, uploading = false, polling = false, timer, completionKey = null;
  let photoURL = null, personBusy = false;
  const terminal = new Set(['completed','failed']);
  const submitPerson = el('person-form').querySelector('[type=submit]');

  function chooseVideo(file) {
    if (uploading) return;
    if (file && file.size > 1024 ** 3) { feedback('upload-feedback','视频最大 1 GB',true); return; }
    videoFile = file || null; el('new-video-file').value = '';
    el('video-selection').hidden = !videoFile; el('start-upload').disabled = !videoFile;
    if (videoFile) { el('selected-video-name').textContent = file.name; el('selected-video-size').textContent = formatSize(file.size); }
    el('upload-transfer').hidden = true; uploadStep(0); feedback('upload-feedback');
  }
  el('new-video-file').onchange = event => chooseVideo(event.target.files[0]);
  el('clear-video').onclick = () => chooseVideo(null);
  const zone = el('video-dropzone');
  for (const type of ['dragenter','dragover']) zone.addEventListener(type, e => { e.preventDefault(); if (!uploading) zone.classList.add('dragging'); });
  for (const type of ['dragleave','drop']) zone.addEventListener(type, e => { e.preventDefault(); zone.classList.remove('dragging'); });
  zone.addEventListener('drop', e => {
    if (e.dataTransfer.files.length > 1) return feedback('upload-feedback','请一次选择一个视频',true);
    chooseVideo(e.dataTransfer.files[0]);
  });
  function uploadStep(step) {
    uploadModal.querySelectorAll('.upload-flow span').forEach((item,i) => item.classList.toggle('flow-current',i === step));
  }
  function lockUpload(busy) {
    uploading = busy; el('new-video-file').disabled = busy; el('clear-video').disabled = busy;
    el('start-upload').disabled = busy || !videoFile; el('start-upload').textContent = busy ? '正在上传…' : '开始上传';
    zone.classList.toggle('unavailable',busy);
    uploadModal.querySelector('.modal-note').textContent = busy ? '关闭窗口不会中断上传，请勿刷新页面' : '上传后自动处理并加入视频库';
  }
  el('upload-form').onsubmit = async event => {
    event.preventDefault(); if (uploading || !videoFile) return;
    const file = videoFile; lockUpload(true); uploadStep(1); feedback('upload-feedback');
    el('upload-transfer').hidden = false; el('upload-progress').value = 0;
    el('upload-percentage').textContent = '0%'; el('upload-stage').textContent = '上传文件';
    try {
      const result = await new Promise((resolve,reject) => {
        const xhr = new XMLHttpRequest(); xhr.open('POST','/mvp-api/uploads'); xhr.responseType = 'json'; xhr.timeout = 30 * 60 * 1000;
        xhr.upload.onprogress = e => {
          if (!e.lengthComputable) return;
          const progress = Math.round(e.loaded / e.total * 100); el('upload-progress').value = progress;
          el('upload-percentage').textContent = progress + '%';
          el('upload-stage').textContent = progress === 100 ? '文件已传输，正在确认入库任务…' : '上传文件';
        };
        xhr.onload = () => xhr.status >= 200 && xhr.status < 300 ? resolve(xhr.response) : reject(new Error(errorMessage(xhr.response,xhr.status)));
        xhr.onerror = () => reject(new Error('网络连接中断，请重试上传；相同文件不会重复入库'));
        xhr.ontimeout = () => reject(new Error('上传超时，请重试'));
        const form = new FormData(); form.set('video',file); xhr.send(form);
      });
      videoFile = null; el('video-selection').hidden = true; el('upload-transfer').hidden = true; uploadStep(2);
      feedback('upload-feedback',result.duplicate ? '此视频已有上传记录，已显示原有处理任务。' : '上传成功，正在后台处理。完成后会自动加入视频库。');
      await pollJobs();
    } catch (e) { feedback('upload-feedback',e.message,true); }
    finally { lockUpload(false); }
  };
  function renderJobs(jobs) {
    const list = el('recent-uploads'); list.replaceChildren();
    if (!jobs.length) { list.append(node('p','暂无上传记录','subtle')); return; }
    const visible = [...jobs.filter(j => !terminal.has(j.status)), ...jobs.filter(j => terminal.has(j.status)).slice(0,5)];
    for (const job of visible) {
      const row = node('article',undefined,'job-entry'); row.dataset.jobId = job.id;
      const heading = node('div',undefined,'recent-job'); const title = node('strong',job.filename); title.title = job.filename;
      heading.append(title,node('span',job.stage,'job-badge ' + (job.status === 'completed' ? 'done' : job.status === 'failed' ? 'error' : '')));
      row.append(heading);
      if (!terminal.has(job.status)) {
        const progress = node('progress'); progress.max = 100; progress.value = job.progress;
        progress.setAttribute('aria-label',`${job.filename} 处理进度`); row.append(progress);
      }
      const detail = node('p', job.message, 'job-detail'); row.append(detail);
      if (job.status === 'failed') {
        const retry = node('button','重试处理','outline-button'); retry.type = 'button';
        retry.onclick = async () => {
          retry.disabled = true;
          try { await api(`/mvp-api/jobs/${encodeURIComponent(job.id)}/retry`,{method:'POST'}); await pollJobs(); }
          catch (e) { feedback('upload-feedback',e.message,true); retry.disabled = false; }
        }; row.append(retry);
      }
      list.append(row);
    }
  }
  async function pollJobs() {
    if (polling) return; polling = true; clearTimeout(timer);
    let active = false;
    try {
      const {jobs} = await api('/mvp-api/jobs'); renderJobs(jobs);
      active = jobs.some(j => !terminal.has(j.status));
      const key = jobs.filter(j => j.status === 'completed').map(j => j.id).sort().join(',');
      if (key !== completionKey) await onLibraryChanged();
      completionKey = key;
    } catch (e) {
      if (uploadModal.open) el('recent-uploads').replaceChildren(node('p','任务状态读取失败，将自动重试：' + e.message,'error modal-feedback'));
    } finally { polling = false; timer = setTimeout(pollJobs, active || uploadModal.open ? 2500 : 10000); }
  }

  function lockPerson(busy) {
    personBusy = busy; submitPerson.disabled = busy; el('new-person-name').disabled = busy; el('new-person-photo').disabled = busy;
    submitPerson.textContent = busy ? '正在登记…' : '新增人名';
  }
  function previewPhoto() {
    if (photoURL) URL.revokeObjectURL(photoURL); photoURL = null;
    const file = el('new-person-photo').files[0];
    if (file?.size > 10 * 1024 ** 2) { el('new-person-photo').value = ''; feedback('person-feedback','照片最大 10 MB',true); return previewPhoto(); }
    el('new-portrait').hidden = !file; el('portrait-empty').hidden = Boolean(file); el('replace-portrait').hidden = !file;
    if (file) { photoURL = URL.createObjectURL(file); el('new-portrait').src = photoURL; }
    else el('new-portrait').removeAttribute('src');
  }
  el('new-person-photo').onchange = () => { feedback('person-feedback'); previewPhoto(); };
  async function loadPeople() {
    const names = await api('/mvp-api/people'); onPeopleChanged(names); el('person-count').textContent = names.length;
    const list = el('registered-people'); list.replaceChildren();
    if (!names.length) list.append(node('p','还没有登记人物','subtle empty-people'));
    for (const name of names) {
      const row = node('div',undefined,'registered-person');
      const avatar = node('img'); avatar.src = `/mvp-api/people/${encodeURIComponent(name)}/photo`; avatar.alt = `${name}的登记照片`; avatar.loading = 'lazy';
      const info = node('div'); info.append(node('strong',name));
      const controls = node('span',undefined,'person-actions');
      function deleteButton() {
        controls.replaceChildren(); const remove = node('button','删除','delete-person'); remove.type = 'button';
        remove.onclick = () => {
          controls.replaceChildren(node('span','确认删除？','subtle'));
          const confirm = node('button','确认删除','delete-person danger'), cancel = node('button','取消','delete-person');
          confirm.type = cancel.type = 'button'; cancel.onclick = deleteButton;
          confirm.onclick = async () => {
            confirm.disabled = cancel.disabled = true;
            try { await api(`/mvp-api/people/${encodeURIComponent(name)}`,{method:'DELETE'}); await loadPeople(); feedback('person-feedback',`已删除「${name}」`); }
            catch (e) { feedback('person-feedback',e.message,true); confirm.disabled = cancel.disabled = false; }
          }; controls.append(confirm,cancel); confirm.focus();
        }; controls.append(remove);
      }
      deleteButton(); row.append(avatar,info,controls); list.append(row);
    }
  }
  el('person-form').onsubmit = async event => {
    event.preventDefault(); if (personBusy) return;
    const name = el('new-person-name').value.trim(), photo = el('new-person-photo').files[0];
    if (!name || !photo) return feedback('person-feedback','请输入姓名并选择一张人脸照片',true);
    const form = new FormData(); form.set('name',name); form.set('photo',photo);
    lockPerson(true); feedback('person-feedback','正在检测照片中的人脸…');
    try {
      await api('/mvp-api/people',{method:'POST',body:form});
      el('person-form').reset(); previewPhoto(); await loadPeople(); feedback('person-feedback',`已登记「${name}」，可通过姓名检索视频。`);
    } catch (e) { feedback('person-feedback',e.message,true); }
    finally { lockPerson(false); }
  };
  el('open-upload').disabled = false; el('open-people').disabled = false;
  await loadPeople().catch(e => feedback('person-feedback','人物列表读取失败：' + e.message,true));
  document.addEventListener('video-library-changed', pollJobs);
  pollJobs();
}
