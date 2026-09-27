"""Export notebooks to PDF (nbconvert `webpdf`, rendered by Chromium; no LaTeX needed).

Usage (from the project root):

    task export_pdf                              # every notebook in notebooks/
    task export_pdf notebooks/3_*.ipynb          # selected notebooks

The PDFs are written to documents/notebooks/. They show the outputs saved in each
notebook; the notebooks are not executed. The first run downloads Chromium (Playwright).
"""

import asyncio
import sys
from pathlib import Path

from nbconvert.nbconvertapp import NbConvertApp

ROOT_PATH = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_PATH / 'documents' / 'notebooks'
DEFAULT_NOTEBOOKS = str(ROOT_PATH / 'notebooks' / '*.ipynb')


class WebPDFExportApp(NbConvertApp):
    """nbconvert with the default Windows event loop restored after initialization.

    On Windows, nbconvert switches asyncio to the selector event loop, which cannot
    start subprocesses. The webpdf exporter needs a subprocess (Chromium), so the export
    fails with `NotImplementedError` unless the proactor loop is restored.
    """

    def initialize(self, argv=None):
        super().initialize(argv)
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


if __name__ == '__main__':
    notebooks = sys.argv[1:] or [DEFAULT_NOTEBOOKS]
    WebPDFExportApp.launch_instance(
        argv=[
            '--to',
            'webpdf',
            '--allow-chromium-download',
            '--output-dir',
            str(OUTPUT_DIR),
            *notebooks,
        ]
    )
