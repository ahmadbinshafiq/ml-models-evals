import os
import psutil
import time

LOGS_DIR = "/app/logs"
BASE_MONITORING_FILENAME = "logs_"


def get_monitoring_filename(monitoring_filename_prefix):
    """creates a new log file on each run"""
    i = 0
    while True:
        filename = monitoring_filename_prefix + str(i) + ".log"
        if not os.path.isfile(filename):
            return filename
        i += 1


def monitor():
    os.makedirs(LOGS_DIR, exist_ok=True)
    filename = get_monitoring_filename(f"{LOGS_DIR}/{BASE_MONITORING_FILENAME}")
    filename = os.path.abspath(filename)

    print(os.getcwd(), os.listdir(), flush=True)
    print(os.path.abspath(filename), flush=True)
    print(f"monitoring system usage to {filename}", flush=True)

    with open(filename, "w") as f:
        while True:
            ram_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage("/").percent
            cpu_percent = psutil.cpu_percent()
            f.write(
                f"ram usage: {ram_percent}%,  disk usage: {disk_percent}%,  cpu usage: {cpu_percent}%\n"
            )
            f.flush()
            time.sleep(0.25)

            print(
                f"ram usage: {ram_percent}%,  disk usage: {disk_percent}%,  cpu usage: {cpu_percent}%\n",
                flush=True,
            )


if __name__ == "__main__":
    monitor()
