import RPi.GPIO as gpio
import time
import sys
import signal
import traceback
import subprocess
from picamera import PiCamera

def signal_handler(signal, frame): # ctrl + c -> exit program
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

gpio.setmode(gpio.BCM)
trig = 4 # 7th
echo = 17 # 6th

gpio.setup(trig, gpio.OUT)
gpio.setup(echo, gpio.IN)

camera = PiCamera()
i = 0

time.sleep(0.5)
print ('-----------------------------------------------------------------sonar start')
try :
    while True :
        time.sleep(1)
        gpio.output(trig, False)
        time.sleep(0.1)
        gpio.output(trig, True)
        time.sleep(0.00001)
        gpio.output(trig, False)
        while gpio.input(echo) == 0 :
            pulse_start = time.time()
        while gpio.input(echo) == 1 :
            pulse_end = time.time()
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17000
        if pulse_duration >=0.01746:
            print('time out')
            continue
        elif distance > 300 or distance==0:
            print('out of range')
            continue
        distance = round(distance, 3)
        if distance < 50.0:
            imgpath = '/home/pi/Desktop/test/image%s.jpg' % i
            camera.start_preview()
            time.sleep(4)
            camera.capture(imgpath)
            camera.stop_preview()
            bashCommand = 'curl -X POST -F image=@'+imgpath+' http://192.168.88.254:5000/readtext'
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            i = i + 1
            
        
        print ('Distance : %f cm'%distance)
        
except (KeyboardInterrupt, SystemExit):
    traceback.print_exc()
    gpio.cleanup()
    sys.exit(0)
except:
    traceback.print_exc()
    gpio.cleanup()
