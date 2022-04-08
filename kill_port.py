import os
import sys
import signal
import pdb
def kill_process(*pids):
  for pid in pids:
    a = os.kill(pid, signal.SIGKILL)
    print('已杀死pid为%s的进程,　返回值是:%s' % (pid, a))

def get_pid(*ports):
	#其中\"为转义"
    pids = []
    print(ports)
    for port in ports:
        msg = os.popen('lsof -i:{}'.format(port)).read()
        msg = msg.split('\n')[1:-1]
        for m in msg:
            m = m.replace('  ', ' ')
            m = m.replace('  ', ' ')
            tokens = m.split(' ')
            pids.append(int(tokens[1]))
    return pids

if __name__ == "__main__":
    # 杀死占用端口号的ps进程
    ports = sys.argv[1:]
    kill_process(*get_pid(*ports))

