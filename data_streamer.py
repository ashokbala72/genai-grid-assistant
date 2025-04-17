import csv
import time
import os

def stream_scada_logs():
    with open('data/scada_sim.csv', 'r', encoding='utf-8', errors='ignore') as file:
        reader = csv.DictReader(file)
        print("Headers:", reader.fieldnames)  # Debug print

        for row in reader:
            print("Headers:", reader.fieldnames)
            log = f"{row['timestamp']} - {row['location']} - {row['event']}"
            with open("data/live_feed.txt", "a", encoding='utf-8') as live:
                live.write(log + "\n")
            print("Streamed:", log)
            time.sleep(2)

if __name__ == "__main__":
    if os.path.exists("data/live_feed.txt"):
        os.remove("data/live_feed.txt")
    stream_scada_logs()
