"""Read-only device observations; wall time never determines a kernel transition."""
import subprocess
import threading

class Telemetry:
    def __init__(self):
        self.samples=[]
        self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self.sample,daemon=True,name='tk-device-observer')

    def sample(self):
        while not self.stop_event.is_set():
            try:
                fields=subprocess.check_output(['nvidia-smi','--id=0',
                    '--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu',
                    '--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
                self.samples.append(dict(usedMiB=int(fields[0]),freeMiB=int(fields[1]),
                    utilizationPercent=int(fields[2]),temperatureC=int(fields[3])))
            except (OSError,ValueError,subprocess.SubprocessError):return
            self.stop_event.wait(1)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join()
