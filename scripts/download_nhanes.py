"""Download official NHANES August 2021-August 2023 component files."""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"
FILES = ["DEMO_L", "BMX_L", "BPXO_L", "GLU_L", "INS_L", "SMQ_L", "ALQ_L", "PAQ_L", "DIQ_L"]


def main():
    output = Path("data/raw")
    output.mkdir(parents=True, exist_ok=True)
    for stem in FILES:
        target = output / f"{stem}.xpt"
        if target.exists():
            print(f"exists: {target}")
            continue
        url = f"{BASE}/{stem}.XPT"
        print(f"download: {url}")
        urlretrieve(url, target)


if __name__ == "__main__":
    main()

