#!/bin/bash
# SQUADRONE — set up a fresh Raspberry Pi 5 for survey.py / b.py / payload_test.py
# Run from the folder that has requirements.txt and the scripts:   bash setup_pi.sh
# Every step prints OK / PROBLEM so you can see exactly what failed.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
ok(){ echo -e "  \e[32mOK\e[0m  $*"; }
bad(){ echo -e "  \e[31mPROBLEM\e[0m  $*"; PROBLEMS=$((PROBLEMS+1)); }
PROBLEMS=0

echo "== 1. System packages (apt) =="
sudo apt update
sudo apt install -y python3-picamera2 python3-numpy python3-venv libzbar0 v4l-utils \
  && ok "apt packages" || bad "apt install failed"

echo "== 2. Python venv (with system packages so picamera2 is visible) =="
if [ ! -d "$DIR/.venv" ]; then
  python3 -m venv --system-site-packages "$DIR/.venv" && ok "created $DIR/.venv"
else
  grep -q "include-system-site-packages = true" "$DIR/.venv/pyvenv.cfg" \
    && ok "existing .venv sees system packages" \
    || bad ".venv exists WITHOUT system packages — delete it and re-run (rm -rf .venv)"
fi
"$DIR/.venv/bin/pip" install --upgrade pip >/dev/null
"$DIR/.venv/bin/pip" uninstall -y opencv-python opencv-python-headless >/dev/null 2>&1
"$DIR/.venv/bin/pip" install -r "$DIR/requirements.txt" && ok "pip requirements" \
  || bad "pip install failed"

echo "== 3. Check imports =="
"$DIR/.venv/bin/python" - <<'EOF'
import importlib
for mod, why in [("numpy", "maths"), ("cv2", "vision"), ("picamera2", "camera"),
                 ("pymavlink", "Pixhawk link"), ("pyzbar.pyzbar", "fallback QR")]:
    try:
        m = importlib.import_module(mod)
        print(f"  OK  {mod:14s} {getattr(m, '__version__', '')}  ({why})")
    except Exception as e:
        print(f"  PROBLEM  {mod:14s} {e}  ({why})")
import cv2
print("  OK  WeChatQRCode available" if hasattr(cv2, "wechat_qrcode_WeChatQRCode")
      else "  PROBLEM  no WeChatQRCode in cv2 -> opencv-contrib-python not active")
EOF

echo "== 4. WeChat QR model files (models/ next to survey.py) =="
for f in detect.prototxt detect.caffemodel sr.prototxt sr.caffemodel; do
  [ -f "$DIR/models/$f" ] && ok "models/$f" || bad "models/$f missing — copy the models folder from the old Pi"
done

echo "== 5. Camera (ArduCam IMX519) boot settings =="
CFG=/boot/firmware/config.txt; [ -f $CFG ] || CFG=/boot/config.txt
grep -qE '^\s*dtoverlay=imx519' $CFG && ok "dtoverlay=imx519 in $CFG" \
  || bad "add 'dtoverlay=imx519' under [all] in $CFG (and camera_auto_detect=0), then reboot"
grep -qE '^\s*camera_auto_detect=1' $CFG && bad "set camera_auto_detect=0 in $CFG"
rpicam-hello --list-cameras -t 1 2>&1 | grep -qi imx519 && ok "IMX519 detected" \
  || bad "IMX519 not detected (cable / config / reboot)"

echo "== 6. Serial port to the Pixhawk (/dev/ttyAMA0 @ 921600) =="
grep -qE '^\s*enable_uart=1' $CFG && ok "enable_uart=1" || bad "add 'enable_uart=1' to $CFG"
grep -q "console=serial0" /boot/firmware/cmdline.txt 2>/dev/null \
  && bad "serial console is ON — run: sudo raspi-config -> Interface -> Serial: login shell NO, hardware YES"
[ -e /dev/ttyAMA0 ] && ok "/dev/ttyAMA0 exists" || bad "/dev/ttyAMA0 missing (UART not enabled?)"
groups | grep -qw dialout && ok "user in dialout group" \
  || bad "run: sudo usermod -aG dialout $USER  (then log out/in)"

echo
[ $PROBLEMS -eq 0 ] && echo -e "\e[32mALL GOOD\e[0m — run scripts with: $DIR/.venv/bin/python survey.py" \
  || echo -e "\e[31m$PROBLEMS PROBLEM(S)\e[0m above — fix them (reboot after config.txt changes) and re-run"