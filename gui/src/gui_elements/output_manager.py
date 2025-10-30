import streamlit as st
import io
from contextlib import contextmanager
import contextlib
import re

@contextmanager
def st_capture(output_func):
    with io.StringIO() as stdout, contextlib.redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield

class TqdmToStreamlit(io.StringIO):
    """
    A custom file-like object that redirects tqdm's output to Streamlit's st.progress
    and st.text widgets.
    """
    # def __init__(self, progress_bar, text_element):
    #     super().__init__()
    #     self.progress_bar = progress_bar
    #     self.text_element = text_element
    #     self.last_progress = 0
    
    def __init__(self, text_element):
        super().__init__()
        self.text_element = text_element

    def write(self, buf):
        # We can use this code if we want to have a nicer progress bar.
        # match = re.search(r"(\d+)%\|", buf)
        # if match:
        #     progress = int(match.group(1))
        #     if progress > self.last_progress:
        #         self.progress_bar.progress(progress)
        #         self.text_element.text(buf.strip())
        #         self.last_progress = progress
        self.text_element.text(buf.strip())

    def flush(self):
        pass # No-op

class QueueIO(io.StringIO):
    """
    A custom file-like object that writes to a queue.
    Used to capture stdout/stderr.
    """
    def __init__(self, q):
        self.queue = q
    
    def write(self, buf):
        self.queue.put(buf)

    def flush(self):
        pass