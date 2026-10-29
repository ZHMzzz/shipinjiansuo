"""Dedicated indexing child. Query processes never install these adapters."""
import logging
import os
import sys


def configure(argv):
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s %(message)s')
    value=os.environ.get('VIDEO_FRAME_ACCELERATION','1')
    if value not in ('0','1'):
        raise ValueError('VIDEO_FRAME_ACCELERATION must be 0 or 1')
    # --help is usable without loading models or requiring accelerator hardware.
    if value=='1' and not any(arg in ('-h','--help') for arg in argv):
        if __package__:
            from .frame_acceleration import enable
        else:
            from frame_acceleration import enable
        enable()
    else:
        logging.getLogger(__name__).info('Frame acceleration disabled; original WISE extraction')


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    configure(argv)
    from wise._main.extract_features import main as extract
    extract(['extract-features',*argv])


if __name__=='__main__':
    main()
