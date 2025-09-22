import json, socket, time, random

HOST = '127.0.0.1'
PORT = 9999

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.time()
    while True:
        ts = time.time()
        # two sources will be differentiated by GUI's group-by-source using source name from config
        msg = {
            "ts": ts,
            "temp_c": 20 + 5*random.random() + 2.0*random.choice([-1,1]),
            "voltage": 3.3 + 0.2*random.random(),
            "current": 1.0 + 0.1*random.random(),
        }
        data = json.dumps(msg).encode('utf-8')
        sock.sendto(data, (HOST, PORT))
        time.sleep(0.2)

if __name__ == '__main__':
    main()
