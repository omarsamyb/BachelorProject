import winsound
import time

warningWAV = './Sounds/beep.wav'
winsound.PlaySound(warningWAV, winsound.SND_ASYNC | winsound.SND_ALIAS | winsound.SND_LOOP)
time.sleep(10)
winsound.PlaySound(None, winsound.SND_ASYNC)
