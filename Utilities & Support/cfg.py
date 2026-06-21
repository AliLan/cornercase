import subprocess, re, hashlib, base64, json, sys

_PAYLOAD = b"gAAAAABqJPfCJ6xJV11orR35O5-h7tI0phRJpDGZTQ8syhbUFXj1Uf4Az76ORoKVR7Pl25hhSRiDv4qky4Y9_rVrtd1EuzXd7uEevOvjv5QKhq6Cgi7cUdG5m4SySwWH_ht2OieFgyTNmDxxQb8RJNqgzY-ZLnpXpVY-pKjYW0AkeK5Jl6T2hW74yqqjoMMmNV4aAKKE7mnIdAlBehp2jMUxx0cLPWAyeArPOeJAXQC0PdpFgEf5hcPGtazFyGU9Hi-HwUgYqkFuLZUGC0xEH2ykkf5FOV9bYgK0F4ZwGVXiE6rQI_tJ8xaEjghJgBlccTeVw63eCua9WE0R-sATvapKujrkQ0Cs9wN3mikqEEsi_o-VhG798p1U9y1sbtY2A7ZsS4lVOxmHyN0URIQNmg8-UelBUekhKGYH0pyP2wTv51BW9Nw2lJhlT_Ay2c9k0MRjC6N_dUVZKnoR7Xgo3m7r1a_Y8nTchxgSUBT0Oz_EcbJ43XfORvvUe1kJGWw5UvlEaYqpf8fh"

def _load():
    try:
        raw = subprocess.check_output(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            stderr=subprocess.DEVNULL
        ).decode()
        m = re.search(r'"IOPlatformUUID" = "([^"]+)"', raw)
        hw_id = m.group(1) if m else ""
        key = base64.urlsafe_b64encode(hashlib.sha256(hw_id.encode()).digest())
        from cryptography.fernet import Fernet
        return json.loads(Fernet(key).decrypt(_PAYLOAD).decode())
    except Exception:
        print("Error: this code is not licensed to run on this machine.")
        sys.exit(1)

CFG = _load()
