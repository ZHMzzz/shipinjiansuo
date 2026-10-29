"""Measure current production ingestion; clone only the untimed test fixture."""
import asyncio
import os
from pathlib import Path
import shutil
import subprocess

os.environ['FRAME_RECHECK_PRODUCTION']='1'
os.environ.setdefault('FRAME_RECHECK_OUTPUT','current-upload-timing-r2-20261028')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import native_recheck

original=shutil.copytree

def fixture_copy(source,destination,*args,**kwargs):
    target=Path(destination)
    # Only the initial experiment fixture is clone-on-write. IngestionManager's
    # measured copy to its new release directory still calls normal copytree.
    if (target.parts[-4:]==('projects','releases','initial','demo')
            and Path(source).resolve()==(native_recheck.ROOT/'projects/demo').resolve()):
        target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists()
        subprocess.run(['/bin/cp','-cR',str(source),str(target)],check=True)
        return str(target)
    return original(source,destination,*args,**kwargs)

if __name__=='__main__':
    shutil.copytree=fixture_copy
    try:asyncio.run(native_recheck.main())
    finally:shutil.copytree=original
