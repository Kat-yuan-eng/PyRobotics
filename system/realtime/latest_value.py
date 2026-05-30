import threading
import copy


# === Phase 1: Latest Value Container ===

class LatestResult:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = None
        self._seq = 0

    def write(self, data):
        with self._lock:
            self._data = data
            self._seq += 1

    def read(self):
        with self._lock:
            return copy.deepcopy(self._data), self._seq
