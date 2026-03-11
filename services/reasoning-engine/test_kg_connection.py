import requests


def check_kg_health():
    url = "http://localhost:8001/health"  # Assuming they run on 8001
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ Connection Successful! Knowledge Graph is ALIVE.")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ KG is there but returned error: {response.status_code}")
    except Exception:
        print("❌ Could not find the KG service. Is it running?")


if __name__ == "__main__":
    check_kg_health()
