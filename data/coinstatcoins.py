import requests
import pandas as pd

url = "https://openapiv1.coinstats.app/coins?limit=500"

headers = {
    "accept": "application/json",
    "X-API-KEY": "G9bXc1n6Gp6xgMBpPrdZi6R7BJ9R0HlUIg2n599/+EI="
}

response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    
    # Access the list of coins inside the 'result' key
    if 'result' in data:
        coins = data['result']
        
        # Extract the relevant data
        symbols = []
        prices = []
        ranks = []
        
        for coin in coins:
            symbols.append(coin['symbol'])
            prices.append(coin['price'])
            ranks.append(coin['rank'])
        
        # Create a DataFrame
        df = pd.DataFrame({
            'Symbol': symbols,
            'Price': prices,
            'Rank': ranks
        })
        
        # Save the DataFrame to a CSV file
        output_file_path = '/Users/cyh/Desktop/coin_stats.csv'
        df.to_csv(output_file_path, index=False)
        
        print(f"Data has been saved to {output_file_path}")
    else:
        print("'result' key not found in the response")
else:
    print(f"Failed to retrieve data: {response.status_code}")
