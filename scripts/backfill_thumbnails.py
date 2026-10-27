"""Repair thumbnails missing due to the upstream enum-key bug, without re-embedding."""
import sqlite3
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with sqlite3.connect(ROOT/'projects/demo/metadata/internal.db') as db:
    media = db.execute('SELECT media.id, source_collections.location, media.path FROM media JOIN source_collections ON source_collections.id=media.source_collection_id').fetchall()
    times = {mid:[r[0] for r in db.execute('SELECT DISTINCT timestamp FROM vectors WHERE media_id=? ORDER BY timestamp',(mid,))] for mid,_,_ in media}
with sqlite3.connect(ROOT/'projects/demo/thumbs.db') as db:
    count=0
    for mid,location,name in media:
        source=Path(location)/name
        if not source.is_absolute(): source=ROOT/source
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run([str(ROOT/'.media-runtime/bin/ffmpeg'),'-v','error','-i',str(source),'-vf','fps=2,scale=320:-2','-q:v','3',str(Path(temp)/'%06d.jpg')],check=True)
            frames=sorted(Path(temp).glob('*.jpg'))
            for t in times[mid]:
                index=round(t*2)
                if index>=len(frames): raise RuntimeError(f'Missing frame {name} @ {t}')
                if db.execute('SELECT 1 FROM thumbnails WHERE media_id=? AND timestamp=?',(mid,t)).fetchone(): continue
                db.execute('INSERT INTO thumbnails(media_id,timestamp,content) VALUES (?,?,?)',(mid,t,frames[index].read_bytes()))
                count+=1
    print(f'Added {count} thumbnails at indexed timestamps')
