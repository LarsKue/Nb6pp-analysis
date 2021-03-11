from time import perf_counter
from time import sleep

class Timer:
    def __init__(self):
        self.start = None
        self.end = None
        self.duration = None

    def begin(self):
        self.start = perf_counter()

    def stop(self):
        self.end = perf_counter()
        self.duration = self.end - self.start

    def __enter__(self):
        self.begin()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.end = perf_counter()
        self.duration = self.end - self.start

class EasyTimer(Timer):
    def __init__(self, message=None):
        super().__init__()
        if message is None:
            self.message = "Execution took {:.2f} ms."
        else:
            self.message = message

    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)
        print(self.message.format(1e3 * self.duration))