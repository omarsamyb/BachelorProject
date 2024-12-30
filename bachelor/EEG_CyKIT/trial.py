import os
import time
import signal
import subprocess

if __name__ == '__main__':
    filename = "OmarSamy_1_awake"
    command = (['python', 'epocStream.py', filename])
    streamer = subprocess.Popen(command, shell=True)

    time.sleep(320)
    # CSV Header (Sensor Order) for EPOC + F3,  FC5, AF3, F7, T7, P7, O1, O2, P8, T8, F8, AF4, FC6, F4, ns
    # AF3, F7, F3, FC5, T7, P7, 01, 02, P8, T8, FC6, F4, F8 and AF4
    # Put the triggers / events here

    os.kill(streamer.pid, signal.CTRL_C_EVENT)
