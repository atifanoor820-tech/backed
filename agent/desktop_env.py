"""Helpers to ensure desktop tools and child processes attach to the user's interactive desktop."""

import ctypes
import os
import subprocess
import sys

if sys.platform == "win32":
    from ctypes import wintypes
else:  # Linux hosts (Vercel): same-size stand-ins so the module imports; every helper below returns early.
    class wintypes:  # noqa: N801
        DWORD = ctypes.c_uint32
        WORD = ctypes.c_uint16
        LPWSTR = ctypes.c_wchar_p
        HANDLE = ctypes.c_void_p


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


def ensure_default_desktop() -> bool:
    """Assign calling thread to WinSta0\\Default so window enumerations and GUI operations target the real screen."""
    if sys.platform != "win32":
        return True
    try:
        user32 = ctypes.windll.user32
        DESKTOP_ALL = 0x01FF
        h_desk = user32.OpenDesktopW("Default", 0, False, DESKTOP_ALL)
        if h_desk:
            return bool(user32.SetThreadDesktop(h_desk))
    except Exception:
        pass
    return False


def bring_to_front(hwnd: int) -> None:
    """Restore and bring a window to the foreground."""
    if sys.platform != "win32" or not hwnd:
        return
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        user32.ShowWindow(hwnd, 9)  # 9 = SW_RESTORE
        cur_tid = kernel32.GetCurrentThreadId()
        fore_hwnd = user32.GetForegroundWindow()
        fore_tid = user32.GetWindowThreadProcessId(fore_hwnd, None)
        if fore_tid and fore_tid != cur_tid:
            user32.AttachThreadInput(cur_tid, fore_tid, True)
            user32.SetForegroundWindow(hwnd)
            user32.AttachThreadInput(cur_tid, fore_tid, False)
        else:
            user32.SetForegroundWindow(hwnd)
    except Exception:
        try:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass


def spawn_on_desktop(cmd: list[str] | str, env: dict | None = None) -> None:
    """Spawn a process explicitly targeting the interactive desktop WinSta0\\Default."""
    if sys.platform != "win32":
        subprocess.Popen(cmd, close_fds=True, env=env)
        return

    cmd_str = subprocess.list2cmdline(cmd) if isinstance(cmd, list) else cmd
    si = STARTUPINFOW()
    si.cb = ctypes.sizeof(STARTUPINFOW)
    si.lpDesktop = "WinSta0\\Default"
    pi = PROCESS_INFORMATION()

    res = ctypes.windll.kernel32.CreateProcessW(
        None, cmd_str, None, None, False, 0, None, None, ctypes.byref(si), ctypes.byref(pi)
    )
    if res:
        ctypes.windll.kernel32.CloseHandle(pi.hProcess)
        ctypes.windll.kernel32.CloseHandle(pi.hThread)
    else:
        # Fallback to standard subprocess.Popen
        subprocess.Popen(cmd, close_fds=True, env=env)
