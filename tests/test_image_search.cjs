const {test} = require('node:test');
const assert = require('node:assert/strict');
require('../ui/image-search.js');
const payload = hits => ({video_results: {videos: {}, unmerged_windows: hits}});
const objects = payload([{media_id: 21, distance: .99}, {media_id: 22, distance: .98}, {media_id: 21, distance: .97}]);
test('strong screenshot replaces duplicate video; unrelated trace scores do not reorder OWL', () => {
 const trace = payload([{media_id: 22, distance: .98, thumbnail_ts: 30}, {media_id: 99, distance: .89}]);
 const hits = ImageSearch.merge(objects, trace).video_results.unmerged_windows;
 assert.deepEqual(hits.map(h => h.media_id), [22, 21]);
 assert.equal(hits[0].thumbnail_ts, 30);
 assert.equal(hits[0].match_kind, 'trace');
 assert.equal(hits[1].distance, .99);
 assert.equal(objects.video_results.unmerged_windows.length, 3);
});
test('weak, nonfinite or absent screenshot evidence preserves native object order', () => {
 const hits = ImageSearch.merge(objects, payload([{media_id: 99, distance: NaN}, {media_id: 22, distance: .8}])).video_results.unmerged_windows;
 assert.deepEqual(hits.map(h => h.media_id), [21, 22]);
 assert.ok(hits.every(h => h.match_kind === 'object'));
});
test('object results arrive before trace; trace failure retains them with warning', async () => {
 let calls = 0, partial = false;
 const result = await ImageSearch.run(new Blob(['test']), '汽车', data => {partial=true; assert.equal(data.video_results.total, 2);}, async(url, options) => {
  calls++;
  if(calls === 1) {
   assert.ok(url.includes('owlv2')); assert.ok(url.includes('negative_text_queries='));
   assert.ok(options.body.has('image_file_queries'));
   return {ok: true, json: async()=>objects};
  }
  assert.ok(partial); assert.equal(url, '/mvp-api/trace'); assert.ok(options.body.has('photo'));
  throw new Error('offline');
 });
 assert.equal(result.video_results.total, 2); assert.equal(result.warnings.length, 1);
});
test('object failure still permits screenshot match; both failures report error', async () => {
 let calls=0;
 const request=async()=>{if(++calls===1)throw new Error('offline');return {ok:true,json:async()=>payload([{media_id:22,distance:.98}])};};
 const result=await ImageSearch.run(new Blob(['test']), '', ()=>{}, request);
 assert.equal(result.video_results.unmerged_windows[0].match_kind,'trace');assert.equal(result.warnings.length,1);
 await assert.rejects(ImageSearch.run(new Blob(['test']), '', ()=>{}, async()=>{throw new Error('offline');}),/物体检索.*截图比对/);
});
