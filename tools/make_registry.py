"""Generate the sample vehicle registry (data/registry.csv).

* ~600 random but realistic Indian registrations
* the ground-truth vehicles of the synthetic test clips
* deliberate "decoys" - plates 1-2 characters away from the ground truth,
  some with the same colour/model - so ranking and explanations are exercised.
The vehicle in clip07 (UP32ZZ9087) is intentionally NOT registered.
"""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
random.seed(7)

MODELS = {
    "car": [("Maruti Suzuki", "Swift"), ("Maruti Suzuki", "Baleno"), ("Maruti Suzuki", "WagonR"),
            ("Maruti Suzuki", "Alto"), ("Hyundai", "i20"), ("Hyundai", "Grand i10"),
            ("Tata", "Tiago"), ("Tata", "Altroz"), ("Honda", "City"), ("Honda", "Amaze"),
            ("Maruti Suzuki", "Dzire"), ("Maruti Suzuki", "Ciaz"), ("Hyundai", "Verna"),
            ("Skoda", "Slavia"), ("Volkswagen", "Virtus")],
    "suv": [("Hyundai", "Creta"), ("Hyundai", "Venue"), ("Kia", "Seltos"), ("Kia", "Sonet"),
            ("Tata", "Nexon"), ("Tata", "Harrier"), ("Mahindra", "XUV700"), ("Mahindra", "Scorpio"),
            ("Mahindra", "Thar"), ("Maruti Suzuki", "Brezza"), ("Toyota", "Fortuner")],
    "van": [("Maruti Suzuki", "Eeco"), ("Maruti Suzuki", "Ertiga"), ("Toyota", "Innova Crysta")],
    "two-wheeler": [("Honda", "Activa"), ("Hero", "Splendor"), ("Bajaj", "Pulsar"), ("TVS", "Jupiter"),
                    ("Royal Enfield", "Classic 350"), ("TVS", "Apache")],
    "auto-rickshaw": [("Bajaj", "RE"), ("Piaggio", "Ape"), ("TVS", "King"), ("Mahindra", "Treo")],
    "truck": [("Tata", "407"), ("Ashok Leyland", "Dost"), ("Eicher", "Pro 2049"), ("Tata", "LPT 1613"),
              ("Mahindra", "Bolero Pickup")],
    "bus": [("Ashok Leyland", "Viking"), ("Tata", "Starbus")],
}
TYPE_WEIGHTS = {"car": 34, "suv": 18, "van": 6, "two-wheeler": 26, "auto-rickshaw": 7, "truck": 7, "bus": 2}
COLOURS = {
    "default": ["white"] * 8 + ["silver"] * 5 + ["grey"] * 4 + ["black"] * 4 + ["red"] * 3 +
               ["blue"] * 3 + ["maroon", "brown", "beige", "orange", "green", "yellow"],
    "auto-rickshaw": ["green"] * 5 + ["yellow"] * 3 + ["black"] * 2 + ["blue"],
    "truck": ["orange"] * 3 + ["yellow"] * 2 + ["white"] * 2 + ["blue", "red", "green", "brown"],
    "bus": ["white", "red", "blue", "orange", "yellow"],
}
STATES = {"MH": ["01", "02", "04", "12", "14", "43"], "KA": ["01", "03", "05", "51", "53"],
          "DL": ["1C", "2C", "3C", "4C", "8C"], "TN": ["01", "07", "09", "22", "38"],
          "GJ": ["01", "05", "18"], "HR": ["26", "55", "51"], "UP": ["14", "16", "32"],
          "TS": ["07", "09"], "WB": ["02", "06"], "KL": ["07", "11"], "RJ": ["14", "27"]}
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"  # I and O are not issued in series
FIRST = ["Aarav", "Vihaan", "Priya", "Ananya", "Rohan", "Kavya", "Arjun", "Meera", "Ishaan", "Sneha",
         "Rahul", "Pooja", "Aditya", "Neha", "Karthik", "Divya", "Siddharth", "Fatima", "Imran", "Gurpreet"]
LAST = ["Sharma", "Patel", "Reddy", "Iyer", "Khan", "Singh", "Nair", "Gupta", "Desai", "Rao",
        "Mehta", "Das", "Kulkarni", "Joshi", "Menon", "Chopra", "Banerjee", "Pillai", "Verma", "Shaikh"]
CITY = {"MH": "Pune", "KA": "Bengaluru", "DL": "New Delhi", "TN": "Chennai", "GJ": "Ahmedabad",
        "HR": "Gurugram", "UP": "Lucknow", "TS": "Hyderabad", "WB": "Kolkata", "KL": "Kochi", "RJ": "Jaipur"}


def rand_plate():
    st = random.choice(list(STATES))
    rto = random.choice(STATES[st])
    series = "".join(random.choice(LETTERS) for _ in range(random.choice([1, 2, 2, 2])))
    return f"{st}{rto}{series}{random.randint(1, 9999):04d}"


def row(reg, btype, make, model, colour, year=None, note=""):
    st = reg[:2] if reg[:2].isalpha() else "MH"
    return {"reg_no": reg, "make": make, "model": model, "body_type": btype, "colour": colour,
            "year": year or random.randint(2010, 2025),
            "fuel": "Petrol" if btype in ("two-wheeler",) else random.choice(["Petrol", "Diesel", "CNG", "Petrol"]),
            "owner": f"{random.choice(FIRST)} {random.choice(LAST)}",
            "city": CITY.get(st, "Mumbai"), "status": note or "Active"}


# Ground truth for the synthetic test clips + decoys
SPECIAL = [
    # clip01 / clip08 : white Swift
    ("MH12DE4521", "car", "Maruti Suzuki", "Swift", "white"),
    ("MH12DE4527", "car", "Maruti Suzuki", "Swift", "white"),
    ("MH12DF4521", "car", "Maruti Suzuki", "Baleno", "silver"),
    ("MH14DE4521", "suv", "Tata", "Nexon", "white"),
    ("MH12DE4321", "car", "Hyundai", "i20", "red"),
    # clip02 : red Creta, mud over 2 characters
    ("KA05MN7788", "suv", "Hyundai", "Creta", "red"),
    ("KA05MH7788", "suv", "Hyundai", "Creta", "white"),
    ("KA05AB7788", "car", "Maruti Suzuki", "Swift", "red"),
    ("KA05MN7783", "suv", "Tata", "Nexon", "blue"),
    ("KA03MN7788", "two-wheeler", "Honda", "Activa", "red"),
    # clip03 : black Honda City (low light)
    ("DL3CAF0912", "car", "Honda", "City", "black"),
    ("DL3CAF0972", "car", "Hyundai", "Verna", "black"),
    ("DL8CAF0912", "suv", "Kia", "Seltos", "grey"),
    ("DL3CAE0912", "car", "Honda", "City", "white"),
    # clip04 : auto-rickshaw (two-line yellow plate)
    ("TN09BX3307", "auto-rickshaw", "Bajaj", "RE", "green"),
    ("TN09BX3807", "auto-rickshaw", "Bajaj", "RE", "green"),
    ("TN07BX3307", "auto-rickshaw", "Piaggio", "Ape", "yellow"),
    ("TN09BX3301", "car", "Maruti Suzuki", "Dzire", "white"),
    # clip05 : blue Activa
    ("GJ01KQ5604", "two-wheeler", "Honda", "Activa", "blue"),
    ("GJ01KQ5684", "two-wheeler", "TVS", "Jupiter", "grey"),
    ("GJ01KR5604", "two-wheeler", "Honda", "Activa", "black"),
    ("GJ05KQ5604", "car", "Hyundai", "Grand i10", "blue"),
    # clip06 : orange truck
    ("HR55AJ8016", "truck", "Tata", "LPT 1613", "orange"),
    ("HR55AJ8018", "truck", "Tata", "407", "yellow"),
    ("HR55AU8016", "truck", "Eicher", "Pro 2049", "white"),
    ("HR26AJ8016", "car", "Honda", "Amaze", "silver"),
    # clip07 (UP32ZZ9087 silver Scorpio) is deliberately absent; a few look-alikes that should NOT fit
    ("UP32ZX4410", "suv", "Mahindra", "Scorpio", "silver"),
    ("UP14ZZ1187", "car", "Maruti Suzuki", "Dzire", "silver"),
]

rows = [row(*s) for s in SPECIAL]
rows[0]["status"] = "Flagged - red-light violation"
seen = {r["reg_no"] for r in rows}
types = list(TYPE_WEIGHTS)
while len(rows) < 600:
    reg = rand_plate()
    if reg in seen or reg.startswith("UP32ZZ"):
        continue
    seen.add(reg)
    bt = random.choices(types, weights=[TYPE_WEIGHTS[t] for t in types])[0]
    mk, md = random.choice(MODELS[bt])
    col = random.choice(COLOURS.get(bt, COLOURS["default"]))
    rows.append(row(reg, bt, mk, md, col))
random.shuffle(rows)

out = ROOT / "data" / "registry.csv"
out.parent.mkdir(exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} vehicles -> {out}")
