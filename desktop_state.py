"""Read-only OS evidence, separate from actions and from browser page content."""
import os
import sys


def windows_runtime_state():
    if sys.platform != "win32":
        return {"applications": [], "complete": False}
    try:
        import ctypes
        from ctypes import wintypes
        user = ctypes.WinDLL("user32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        user.IsWindowVisible.argtypes = [wintypes.HWND]
        user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        apps, errors = [], []

        @callback_type
        def visit(hwnd, _):
            handle = None
            try:
                if not user.IsWindowVisible(hwnd):
                    return True
                title = ctypes.create_unicode_buffer(1024)
                user.GetWindowTextW(hwnd, title, len(title))
                if not title.value:
                    return True
                pid = wintypes.DWORD()
                user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                handle = kernel.OpenProcess(0x1000, False, pid.value)
                path = ctypes.create_unicode_buffer(32768)
                length = wintypes.DWORD(len(path))
                if not handle or not kernel.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(length)):
                    errors.append(True)
                    return True
                apps.append({"process_name": os.path.basename(path.value), "pid": pid.value, "title": title.value})
            except Exception:
                errors.append(True)
            finally:
                if handle:
                    kernel.CloseHandle(handle)
            return True

        complete = bool(user.EnumWindows(visit, 0)) and not errors
        return {"applications": apps, "complete": complete, "source": "windows_window_api"}
    except Exception:
        return {"applications": [], "complete": False}
