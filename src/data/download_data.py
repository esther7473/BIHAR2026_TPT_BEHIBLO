import pandas as pd
import requests
from src.common.common import CONFIG
from src.data.database import save_weather_data, get_last_timestamp 

URL       = CONFIG["api"]["url"]
LATITUDE  = CONFIG["api"]["latitude"]
LONGITUDE = CONFIG["api"]["longitude"]
START     = CONFIG["api"]["start_date"]
END       = CONFIG["api"]["end_date"]
VARIABLE  = CONFIG["api"]["variable"]


def fetch_temperature(start_date=None, end_date=None):
    params = {
        "latitude":   LATITUDE,
        "longitude":  LONGITUDE,
        "start_date": start_date or START,
        "end_date":   end_date   or END,
        "hourly":     VARIABLE
    }
    response = requests.get(URL, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame({
        "timestamp":   data["hourly"]["time"],    
        "temperature": data["hourly"][VARIABLE]
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")                    

    save_weather_data(df)
    print(f" {len(df)} observations sauvegardées ({params['start_date']} --- {params['end_date']})")
    return df


def fetch_recent_data(end_date=END):
    last_date = get_last_timestamp()

    if last_date is None:
        start_date = CONFIG["api"]["start_date"]
        print("BDD vide --- téléchargement complet")
    else:
        start_date = (last_date + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
        print(f"Dernière date en BDD : {last_date}")

    if start_date >= end_date:
        print(" Données déjà à jour")
        return None

    return fetch_temperature(start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    from src.data.database import init_db
    init_db()
    fetch_recent_data()