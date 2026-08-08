def severity_to_band(sev):
    if sev < 0.2:
        return "0–20%"
    elif sev < 0.4:
        return "20–40%"
    elif sev < 0.6:
        return "40–60%"
    elif sev < 0.8:
        return "60–80%"
    else:
        return "80–100%"
