import glob, os, subprocess, time, cv2
from picamera2 import Picamera2
from libcamera import controls

def find_vcm():
    for p in sorted(glob.glob('/sys/class/video4linux/v4l-subdev*')):
        if open(p + '/name').read().strip().startswith('ak7375'):
            return '/dev/' + os.path.basename(p)
    raise SystemExit('ak7375 subdev not found')

vcm = find_vcm()
def code():
    r = subprocess.run(['v4l2-ctl', '-d', vcm, '--get-ctrl=focus_absolute'], capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()

picam2 = Picamera2(1)
try:
    picam2.configure(picam2.create_still_configuration(main={'size': (1920, 1080), 'format': 'RGB888'}))
    picam2.start(); time.sleep(2)
    md = picam2.capture_metadata()
    picam2.set_controls({'AeEnable': False, 'ExposureTime': md['ExposureTime'],
                         'AnalogueGain': md['AnalogueGain'], 'AfMode': controls.AfModeEnum.Manual})
    time.sleep(1)
    steps = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10]
    for lp in steps + steps[-2::-1]:
        picam2.set_controls({'LensPosition': lp}); time.sleep(1.0)
        for _ in range(4): picam2.capture_array()
        g = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_BGR2GRAY)
        h, w = g.shape; roi = g[h//4:3*h//4, w//4:3*w//4]
        m = picam2.capture_metadata()
        print(f"req {lp:5} meta {m.get('LensPosition')}  {code()}  sharp {cv2.Laplacian(roi, cv2.CV_64F).var():.1f}  mean {roi.mean():.0f}")
finally:
    picam2.close()