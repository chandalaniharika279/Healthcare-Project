import pyreadr
print("Reading RData file...")
result = pyreadr.read_r("D:/Nandini/data/5v_cleandf.rdata")
df = list(result.values())[0]
print("RData file read successfully.")
df.to_csv("D:/Nandini/hospital_triage.csv", index=False)

print("Converted to CSV")
