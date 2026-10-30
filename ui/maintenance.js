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
  let videos = [], uploading = false, polling = false, timer, completionKey = null;
  let photoURL = null, personBusy = false, selectedPhotos = [];
  const expandedPeople = new Set();
  const terminal = new Set(['completed','failed']);
  const submitPerson = el('person-form').querySelector('[type=submit]');

  const pendingVideos = () => videos.filter(item => item.file);
  function renderSelection() {
    el('video-selection').hidden = !videos.length;
    el('selected-video-summary').textContent = `已选择 ${videos.length} 个视频 · ${formatSize(videos.reduce((sum,item) => sum + item.size, 0))}`;
    const list = el('selected-video-list'); list.replaceChildren();
    for (const item of videos) {
      const row = node('div',undefined,'selected-file');
      const info = node('div'); const name = node('strong',item.name); name.title = item.name;
      info.append(name,node('span',`${formatSize(item.size)} · ${item.message}`,item.status === 'failed' ? 'error' : ''));
      row.append(node('span','▶','file-symbol'),info);
      if (item.status === 'failed') {
        const retry = node('button','重试','outline-button'); retry.type = 'button'; retry.disabled = uploading;
        retry.setAttribute('aria-label',`重试上传 ${item.name}`); retry.onclick = () => uploadVideos([item]); row.append(retry);
      }
      const remove = node('button','×'); remove.type = 'button'; remove.disabled = uploading;
      remove.setAttribute('aria-label',`移除 ${item.name}`);
      remove.onclick = () => { videos = videos.filter(video => video !== item); renderSelection(); };
      row.append(remove); list.append(row);
    }
    const pending = pendingVideos();
    el('clear-video').disabled = uploading;
    el('start-upload').disabled = uploading || !pending.length;
    el('start-upload').textContent = uploading ? '正在上传…' : pending.length ? `上传 ${pending.length} 个视频` : '开始上传';
  }
  function chooseVideos(files) {
    if (uploading) return;
    const rejected = [];
    for (const file of files) {
      if (file.size > 1024 ** 3) { rejected.push(`${file.name}：超过 1 GB`); continue; }
      const key = JSON.stringify([file.name,file.size,file.lastModified]);
      if (videos.some(item => item.key === key)) continue;
      videos.push({file,key,name:file.name,size:file.size,status:'pending',message:'等待上传'});
    }
    el('new-video-file').value = ''; renderSelection();
    el('upload-transfer').hidden = true; uploadStep(0);
    feedback('upload-feedback',rejected.length ? `以下文件未加入：${rejected.join('；')}` : '',Boolean(rejected.length));
  }
  el('new-video-file').onchange = event => chooseVideos(event.target.files);
  el('clear-video').onclick = () => {
    if (uploading) return;
    videos = []; renderSelection(); el('upload-transfer').hidden = true; uploadStep(0); feedback('upload-feedback');
  };
  const zone = el('video-dropzone');
  for (const type of ['dragenter','dragover']) zone.addEventListener(type, e => { e.preventDefault(); if (!uploading) zone.classList.add('dragging'); });
  for (const type of ['dragleave','drop']) zone.addEventListener(type, e => { e.preventDefault(); zone.classList.remove('dragging'); });
  zone.addEventListener('drop', e => chooseVideos(e.dataTransfer.files));
  function uploadStep(step) {
    uploadModal.querySelectorAll('.upload-flow span').forEach((item,i) => item.classList.toggle('flow-current',i === step));
  }
  function lockUpload(busy) {
    uploading = busy; el('new-video-file').disabled = busy; renderSelection();
    zone.classList.toggle('unavailable',busy);
    uploadModal.querySelector('.modal-note').textContent = busy ? '关闭窗口不会中断上传，请勿刷新页面' : '上传后自动处理并加入视频库';
  }
  async function uploadVideos(batch) {
    if (uploading || !batch.length) return;
    lockUpload(true); uploadStep(1); feedback('upload-feedback');
    let succeeded = 0, duplicates = 0, failed = 0;
    try {
      for (const [index,item] of batch.entries()) {
        item.status = 'uploading'; item.message = '正在上传'; renderSelection();
        const label = `第 ${index + 1}/${batch.length} 个：${item.name}`;
        el('upload-transfer').hidden = false; el('upload-progress').value = 0;
        el('upload-percentage').textContent = '0%'; el('upload-stage').textContent = label;
        try {
          const result = await new Promise((resolve,reject) => {
            const xhr = new XMLHttpRequest(); xhr.open('POST','/mvp-api/uploads'); xhr.responseType = 'json'; xhr.timeout = 30 * 60 * 1000;
            xhr.upload.onprogress = e => {
              if (!e.lengthComputable) return;
              const progress = Math.round(e.loaded / e.total * 100); el('upload-progress').value = progress;
              el('upload-percentage').textContent = progress + '%';
              el('upload-stage').textContent = label + (progress === 100 ? ' · 正在确认入库任务…' : '');
            };
            xhr.onload = () => {
              if (xhr.status < 200 || xhr.status >= 300) return reject(new Error(errorMessage(xhr.response,xhr.status)));
              if (!xhr.response?.job?.id) return reject(new Error('服务返回了无法读取的任务，请重试'));
              resolve(xhr.response);
            };
            xhr.onerror = () => reject(new Error('网络连接中断，请重试上传；相同文件不会重复入库'));
            xhr.ontimeout = () => reject(new Error('上传超时，请重试'));
            xhr.onabort = () => reject(new Error('上传已中断，请重试'));
            const form = new FormData(); form.set('video',item.file); xhr.send(form);
          });
          if (result.duplicate) duplicates++; else succeeded++;
          item.status = 'submitted'; item.file = null;
          item.message = result.duplicate ? '已有上传记录，请查看最近上传' : '上传成功，后台处理中';
          void pollJobs();
        } catch (e) { failed++; item.status = 'failed'; item.message = e.message; }
        renderSelection();
      }
      el('upload-transfer').hidden = true; uploadStep(failed ? 0 : 2);
      feedback('upload-feedback',`本次上传：成功 ${succeeded} 个，已有记录 ${duplicates} 个，失败 ${failed} 个。` +
        (failed ? '失败文件可单独重试，也可点击上传按钮重试全部失败文件。' : '后台处理状态见「最近上传」。'),failed > 0);
    } finally { lockUpload(false); }
  }
  el('upload-form').onsubmit = event => { event.preventDefault(); void uploadVideos(pendingVideos()); };
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
    submitPerson.textContent = busy ? '正在保存…' : '保存照片';
    peopleModal.querySelectorAll('[data-person-action]').forEach(button => { button.disabled = busy || button.dataset.unavailable === 'true'; });
  }
  function previewPhoto() {
    if (photoURL) URL.revokeObjectURL(photoURL); photoURL = null;
    const file = selectedPhotos[0];
    el('new-portrait').hidden = !file; el('portrait-empty').hidden = Boolean(file); el('replace-portrait').hidden = !file;
    if (file) { photoURL = URL.createObjectURL(file); el('new-portrait').src = photoURL; }
    else el('new-portrait').removeAttribute('src');
    el('replace-portrait').textContent = `已选 ${selectedPhotos.length} 张 · 重新选择`;
    const list = el('selected-portraits'); list.replaceChildren(); list.hidden = !selectedPhotos.length;
    selectedPhotos.forEach((photo,index) => {
      const row = node('div',undefined,'selected-portrait'); const name = node('span',photo.name); name.title = photo.name;
      const remove = node('button','×'); remove.type = 'button'; remove.dataset.personAction = '';
      remove.setAttribute('aria-label',`移除待登记照片 ${photo.name}`);
      remove.onclick = () => { if (personBusy) return; selectedPhotos.splice(index,1); previewPhoto(); };
      row.append(name,remove); list.append(row);
    });
  }
  el('new-person-photo').onchange = event => {
    const files = Array.from(event.target.files); event.target.value = '';
    if (!files.length) return;
    if (files.length > 10) return feedback('person-feedback','每次最多选择 10 张照片，请重新选择。',true);
    const tooLarge = files.find(file => file.size > 10 * 1024 ** 2);
    if (tooLarge) return feedback('person-feedback',`${tooLarge.name} 超过 10 MB，请重新选择。`,true);
    selectedPhotos = files; feedback('person-feedback'); previewPhoto();
  };
  async function loadPeople() {
    const people = await api('/mvp-api/people?details=true');
    const names = people.map(person => person.name); onPeopleChanged(names); el('person-count').textContent = names.length;
    el('registered-person-names').replaceChildren(...names.map(name => { const option = node('option'); option.value = name; return option; }));
    const list = el('registered-people'); list.replaceChildren();
    if (!people.length) list.append(node('p','还没有登记人物','subtle empty-people'));
    for (const person of people) {
      const {name,photos,photo_count:count} = person, base = `/mvp-api/people/${encodeURIComponent(name)}`;
      const entry = node('article',undefined,'person-entry'); entry.dataset.personName = name;
      const row = node('div',undefined,'registered-person');
      const avatar = node('img'); avatar.src = photos[0].url; avatar.alt = `${name}的登记照片`; avatar.loading = 'lazy';
      const info = node('div'); info.append(node('strong',name),node('small',`${count} 张照片`));
      const controls = node('span',undefined,'person-actions');
      const add = node('button','添加照片','person-link'); add.type = 'button'; add.dataset.personAction = '';
      add.dataset.unavailable = String(count >= 10); add.disabled = count >= 10;
      add.setAttribute('aria-label',`为${name}添加照片`);
      add.onclick = () => {
        if (personBusy) return;
        el('new-person-name').value = name;
        feedback('person-feedback',`为「${name}」追加照片，已有 ${count} 张。`);
        el('person-form').scrollIntoView({block:'nearest'}); el('new-person-photo').click();
      };
      const toggle = node('button','查看照片','person-link'); toggle.type = 'button';
      const gallery = node('div',undefined,'person-photos'); gallery.hidden = !expandedPeople.has(name);
      const setExpanded = () => {
        toggle.textContent = gallery.hidden ? '查看照片' : '收起照片'; toggle.setAttribute('aria-expanded',String(!gallery.hidden));
      };
      toggle.setAttribute('aria-label',`查看${name}的登记照片`);
      toggle.onclick = () => { gallery.hidden = !gallery.hidden; if (gallery.hidden) expandedPeople.delete(name); else expandedPeople.add(name); setExpanded(); };
      setExpanded();
      const deletion = node('span',undefined,'person-delete-controls');
      function deleteButton() {
        deletion.replaceChildren(); const remove = node('button','删除人物','delete-person'); remove.type = 'button'; remove.dataset.personAction = '';
        remove.onclick = () => {
          if (personBusy) return;
          deletion.replaceChildren(node('span',`删除全部 ${count} 张？`,'subtle'));
          const confirm = node('button','确认删除','delete-person danger'), cancel = node('button','取消','delete-person');
          confirm.type = cancel.type = 'button'; confirm.dataset.personAction = cancel.dataset.personAction = ''; cancel.onclick = deleteButton;
          confirm.onclick = async () => {
            if (personBusy) return; lockPerson(true);
            try { await api(base,{method:'DELETE'}); expandedPeople.delete(name); await loadPeople(); feedback('person-feedback',`已删除「${name}」及其全部登记照片`); }
            catch (e) { feedback('person-feedback',e.message,true); }
            finally { lockPerson(false); }
          }; deletion.append(confirm,cancel); confirm.focus();
        }; deletion.append(remove);
      }
      deleteButton(); controls.append(add,toggle); row.append(avatar,info,controls);
      for (const [index,photo] of photos.entries()) {
        const card = node('div',undefined,'person-photo');
        const img = node('img'); img.src = photo.url; img.alt = `${name}的第 ${index + 1} 张登记照片`; img.loading = 'lazy';
        const remove = node('button','删除此照片','delete-person'); remove.type = 'button'; remove.dataset.personAction = '';
        remove.dataset.unavailable = String(count === 1); remove.disabled = count === 1;
        remove.setAttribute('aria-label',`删除${name}的第 ${index + 1} 张照片`);
        if (count === 1) remove.title = '至少保留一张照片；可使用删除人物移除全部登记';
        let confirming = false;
        remove.onclick = async () => {
          if (personBusy) return;
          if (!confirming) { confirming = true; remove.textContent = '确认删除'; return; }
          lockPerson(true);
          try { await api(photo.url,{method:'DELETE'}); await loadPeople(); feedback('person-feedback',`已删除「${name}」的一张照片`); }
          catch (e) { feedback('person-feedback',e.message,true); confirming = false; remove.textContent = '删除此照片'; }
          finally { lockPerson(false); }
        };
        remove.onblur = () => { if (!personBusy) { confirming = false; remove.textContent = '删除此照片'; } };
        card.append(img,remove); gallery.append(card);
      }
      entry.append(row,gallery,deletion); list.append(entry);
    }
    lockPerson(personBusy);
  }
  el('person-form').onsubmit = async event => {
    event.preventDefault(); if (personBusy) return;
    const name = el('new-person-name').value.trim();
    if (!name || !selectedPhotos.length) return feedback('person-feedback','请输入姓名并选择人脸照片',true);
    const form = new FormData(); form.set('name',name); selectedPhotos.forEach(photo => form.append('photos',photo));
    lockPerson(true); feedback('person-feedback','正在逐张检测人脸，全部通过后保存…');
    try {
      const result = await api('/mvp-api/people',{method:'POST',body:form});
      selectedPhotos = []; previewPhoto(); expandedPeople.add(name); await loadPeople();
      feedback('person-feedback',`「${name}」新增 ${result.added} 张，现有 ${result.photo_count} 张照片。` + (result.duplicates ? `跳过 ${result.duplicates} 张重复文件。` : ''));
    } catch (e) { feedback('person-feedback',e.message,true); }
    finally { lockPerson(false); }
  };
  el('open-upload').disabled = false; el('open-people').disabled = false;
  await loadPeople().catch(e => feedback('person-feedback','人物列表读取失败：' + e.message,true));
  document.addEventListener('video-library-changed', pollJobs);
  pollJobs();
}
