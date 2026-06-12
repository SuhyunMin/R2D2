#!/usr/bin/env python3
"""
디버그 래퍼: adr_integrated.py를 subprocess로 실행하고 전체 출력을 파일에 저장.
"""
import subprocess
import sys
import os

LOG_PATH = "/tmp/adr_full_debug.log"
script_dir = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(script_dir, "adr_integrated.py")

# 원래 스크립트에 전달할 인자 (--enable-berthing 등)
args = sys.argv[1:]

cmd = [sys.executable, "-u", target] + args
print(f"[DEBUG] 실행: {' '.join(cmd)}")
print(f"[DEBUG] 로그 저장: {LOG_PATH}")

with open(LOG_PATH, "w") as log_file:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        log_file.write(line)
    proc.wait()

print(f"\n[DEBUG] 종료 코드: {proc.returncode}")
print(f"[DEBUG] 전체 로그: {LOG_PATH}")

# 마지막 100줄 요약
with open(LOG_PATH, "r") as f:
    lines = f.readlines()
print(f"[DEBUG] 총 {len(lines)}줄 출력됨")
if len(lines) > 100:
    print("[DEBUG] === 마지막 100줄 ===")
    for line in lines[-100:]:
        print(line, end="")
