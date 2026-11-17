"""Launch Abbas agent API service on the user's interactive desktop (WinSta0\\Default)."""

import ctypes
import os
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


def main():
    is_win = sys.platform == "win32"
    if is_win:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        h_desk = user32.GetThreadDesktop(kernel32.GetCurrentThreadId())
        buf = ctypes.create_unicode_buffer(256)
        user32.GetUserObjectInformationW(h_desk, 2, buf, 512, None)
        current_desk = buf.value
    else:
        current_desk = "Default"

    # If already on Default desktop (e.g. user ran it in standard CMD / PowerShell terminal), run uvicorn directly
    if current_desk == "Default" or not is_win:
        import uvicorn

        uvicorn.run("services.agent_api.main:app", host="127.0.0.1", port=8000)
        return

    # If in an isolated desktop (e.g. Antigravity task runner), re-launch on WinSta0\Default
    print(f"Current desktop is '{current_desk}'. Relaunching uvicorn on WinSta0\\Default...", flush=True)
    si = STARTUPINFOW()
    si.cb = ctypes.sizeof(STARTUPINFOW)
    si.lpDesktop = "WinSta0\\Default"
    si.dwFlags = 0x00000100  # STARTF_USESTDHANDLES
    si.hStdInput = kernel32.GetStdHandle(-10)
    si.hStdOutput = kernel32.GetStdHandle(-11)
    si.hStdError = kernel32.GetStdHandle(-12)
    pi = PROCESS_INFORMATION()

    cmd = f'"{sys.executable}" -m uvicorn services.agent_api.main:app --host 127.0.0.1 --port 8000'
    res = kernel32.CreateProcessW(
        None, cmd, None, None, True, 0, None, str(ROOT), ctypes.byref(si), ctypes.byref(pi)
    )
    if not res:
        err = kernel32.GetLastError()
        print(f"Failed to spawn on WinSta0\\Default (err {err}); running in current process...", file=sys.stderr)
        import uvicorn

        uvicorn.run("services.agent_api.main:app", host="127.0.0.1", port=8000)
        return

    kernel32.WaitForSingleObject(pi.hProcess, 0xFFFFFFFF)
    code = wintypes.DWORD()
    kernel32.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
    sys.exit(code.value)


if __name__ == "__main__":
    main()
