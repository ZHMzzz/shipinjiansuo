/* Keep native WISE object ranking; only promote strong screenshot evidence.
 * Scores from the two routes are deliberately never averaged or compared.
 */
globalThis.ImageSearch = (() => {
  const traceThreshold = 0.90; // Conservative local heuristic, not a calibrated probability.
  function merge(objects, trace) {
    const a = objects?.video_results, b = trace?.video_results;
    const verified = (b?.unmerged_windows || []).filter(h => Number.isFinite(h.distance) && h.distance >= traceThreshold)
      .map(h => ({...h, match_kind: 'trace'}));
    const related = (a?.unmerged_windows || []).map(h => ({...h, match_kind: 'object'}));
    const seen = new Set();
    const hits = [...verified, ...related].filter(h => {
      const key = String(h.media_id);
      if (seen.has(key)) return false;
      seen.add(key); return true;
    });
    return {score_kind: 'image', video_results: {videos: {...a?.videos, ...b?.videos}, unmerged_windows: hits,
      merged_windows: hits, total: hits.length}};
  }
  async function run(file, negative, onPartial, request = fetch) {
    let objects, trace;
    const warnings = [];
    async function post(url, field) {
      const form = new FormData(); form.append(field, file);
      const response = await request(url, {method: 'POST', body: form});
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `请求失败 (${response.status})`);
      return data;
    }
    const params = new URLSearchParams({search_in: 'video', feature_extractor_id: 'transformers/owlv2/google/owlv2-base-patch16-ensemble', start: '0', end: '100', thumbnails_to_send: '1', add_prefix: 'false'});
    if (negative) params.append('negative_text_queries', negative);
    try { objects = await post('/demo/search?' + params, 'image_file_queries'); }
    catch (error) { warnings.push('物体检索暂不可用：' + error.message); }
    if (objects) onPartial(merge(objects, null));
    try { trace = await post('/mvp-api/trace', 'photo'); }
    catch (error) { warnings.push('截图比对暂不可用：' + error.message); }
    if (!objects && !trace) throw new Error(warnings.join('；'));
    return {...merge(objects, trace), warnings};
  }
  return {merge, run};
})();
